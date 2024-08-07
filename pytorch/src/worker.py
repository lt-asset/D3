import argparse
import copy
from copyreg import pickle
import os
from datetime import datetime
import sys
import time
import traceback
from typing import Any, Callable, Type, Dict, List, Optional, Union
import pickle
import socket
from contextlib import closing
import multiprocessing
from dlrm_model_generation import TestModel
from constants import NUM_MODELS
from test_utils import convert_awaitable_to_tensor, dump_pred_result, dump_intermediate_result, dump_intermediate_result_sequential_lzma, dump_pred_result_lzma, dump_intermediate_result_lzma
from freeze_one_layer import random_pick_n_index, freeze_model_n_layer

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_sharding import create_test_sharder
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestSparseNNBase,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.inference.modules import quantize_embeddings
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim import RowWiseAdagrad
from torchrec import distributed as trec_dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

def may_modify_local_batches(
    inputs,
    world_size,
    global_batch_size,
):
    batch_size = int(global_batch_size/world_size)
    global_input = inputs[0]
    global_float = global_input.float_features
    global_label = global_input.label

    global_idlist_lengths = global_input.idlist_features.lengths()
    global_idlist_indices = global_input.idlist_features.values()
    global_idscore_lengths = global_input.idscore_features.lengths()
    global_idscore_indices = global_input.idscore_features.values()
    global_idscore_weights = global_input.idscore_features.weights()

    features = global_input.idlist_features.keys()
    weighted_features = global_input.idscore_features.keys()

    length_per_feature = int(len(global_idlist_lengths)/len(features))
    length_per_weighted_feature = int(len(global_idscore_lengths)/len(weighted_features))
        
    # Split global batch into local batches.
    local_inputs = []
    for r in range(world_size):
        local_idlist_lengths = []
        local_idlist_indices = []
        local_idscore_lengths = []
        local_idscore_indices = []
        local_idscore_weights = []

        begin = 0
        end = 0
        for k in range(len(features)):
            lengths = global_idlist_lengths[k*length_per_feature: (k + 1)*length_per_feature]
            sum = lengths.sum()
            end = begin + sum
            indices = global_idlist_indices[begin:end]
            local_idlist_lengths.append(
                lengths[r * batch_size : (r + 1) * batch_size]
            )
            lengths_cumsum = [0] + lengths.view(world_size, -1).sum(dim=1).cumsum(
                dim=0
            ).tolist()
            local_idlist_indices.append(
                indices[lengths_cumsum[r] : lengths_cumsum[r + 1]]
            )
            begin = end

        begin = 0
        end = 0
        for k in range(len(weighted_features)):
            lengths = global_idscore_lengths[k*length_per_weighted_feature: (k + 1)*length_per_weighted_feature]
            sum = lengths.sum()
            end = begin + sum
            indices = global_idscore_indices[begin:end]
            weights = global_idscore_weights[begin:end]
            local_idscore_lengths.append(
                lengths[r * batch_size : (r + 1) * batch_size]
            )
            lengths_cumsum = [0] + lengths.view(world_size, -1).sum(dim=1).cumsum(
                dim=0
            ).tolist()
            local_idscore_indices.append(
                indices[lengths_cumsum[r] : lengths_cumsum[r + 1]]
            )
            local_idscore_weights.append(
                weights[lengths_cumsum[r] : lengths_cumsum[r + 1]]
            )
            begin = end

        local_idlist_kjt = KeyedJaggedTensor(
            keys=features,
            values=torch.cat(local_idlist_indices),
            lengths=torch.cat(local_idlist_lengths),
        )

        local_idscore_kjt = (
            KeyedJaggedTensor(
                keys=weighted_features,
                values=torch.cat(local_idscore_indices),
                lengths=torch.cat(local_idscore_lengths),
                weights=torch.cat(local_idscore_weights),
            )
            if local_idscore_indices
            else None
        )

        local_input = ModelInput(
            float_features=global_float[r * batch_size : (r + 1) * batch_size],
            idlist_features=local_idlist_kjt,
            idscore_features=local_idscore_kjt,
            label=global_label[r * batch_size : (r + 1) * batch_size],
        )
        local_inputs.append(local_input)
    
    return(
        copy.deepcopy(global_input),
        local_inputs,
    )

def load_or_gen_layer_freeze_index(path, params, n=1):
    # generate a random shuffle of the order of params
    if os.path.exists(path):
        with open(path, "rb") as f:
            layer_freeze_index_order = pickle.load(f)
    else:
        initial_order = list(range(len(params)))
        layer_freeze_index_order = random_pick_n_index(initial_order, len(initial_order))
        with open(path, "wb") as f:
            pickle.dump(layer_freeze_index_order, f)
    layer_freeze_index = layer_freeze_index_order[:n]
    return layer_freeze_index


def load_model_and_input(world_size, global_batch_size, model_class, seed, model_state_dict_path, input_path, freeze_n = 0):
    if not os.path.exists(model_state_dict_path):
        raise Exception(model_state_dict_path + "doesn't exists")
    if model_state_dict_path.endswith('.onnx'):
        import onnx
        from onnx2pytorch import ConvertModel
        onnx_model = onnx.load(model_state_dict_path)
        device = torch.device("cpu")
        model = ConvertModel(onnx_model, experimental=True)
        # print(model)
        model = model.to(device)
        # model = None

        with open(input_path, 'rb') as f:
            ((train_input, train_label), (test_input, test_label)) = pickle.load(f)
        train_input = torch.tensor(train_input, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.float32)
        test_input = torch.tensor(test_input, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.float32)
        # Load test data
        inputs = ((train_input, train_label), (test_input, test_label))
        # return(model, input)
    else:
        if seed >= int(0.25*NUM_MODELS) and seed < int(0.5*NUM_MODELS):
            model = model_class(seed=seed - int(0.25*NUM_MODELS))
        else:
            model = model_class(seed=seed)

        print("model loaded from", model_state_dict_path)
        model.load_state_dict(torch.load(model_state_dict_path))

        if not os.path.exists(input_path):
            raise Exception(model_state_dict_path + "doesn't exists")
        with open(input_path, "rb") as f:
            inputs = pickle.load(f)
        new_inputs = []
        for input in inputs:
            new_inputs.append(may_modify_local_batches(input, world_size, global_batch_size))
        inputs = new_inputs
        # return (model, inputs)

    # Freeze n layers if seed is between NUM_MODELS/4 and NUM_MODELS/2 or between 3*NUM_MODELS/4 and NUM_MODELS
    if (seed >= int(0.25*NUM_MODELS) and seed < int(0.5*NUM_MODELS)) or (seed >= int(0.75*NUM_MODELS)):
        params = list(model.named_parameters())
        freeze_index = load_or_gen_layer_freeze_index(model_state_dict_path+".freeze", params, n=1)
        freeze_model_n_layer(model, freeze_index)
    return (model, inputs)

def copy_state_dict(
    loc: Dict[str, Union[torch.Tensor, ShardedTensor]],
    glob: Dict[str, torch.Tensor],
) -> None:
    for name, tensor in loc.items():
        assert name in glob
        global_tensor = glob[name]
        if isinstance(global_tensor, ShardedTensor):
            global_tensor = global_tensor.local_shards()[0].tensor
        if isinstance(tensor, ShardedTensor):
            for local_shard in tensor.local_shards():
                assert global_tensor.ndim == local_shard.tensor.ndim
                shard_meta = local_shard.metadata
                t = global_tensor.detach()
                if t.ndim == 1:
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0]
                    ]
                elif t.ndim == 2:
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0],
                        shard_meta.shard_offsets[1] : shard_meta.shard_offsets[1]
                        + local_shard.tensor.shape[1],
                    ]
                else:
                    raise ValueError("Tensors with ndim > 2 are not supported")
                local_shard.tensor.copy_(t)
        else:
            tensor.copy_(global_tensor)

def sharding_single_rank_test(
    rank: int,
    world_size: int,
    model,
    inputs,
    # embedding_groups: Dict[str, List[str]],
    sharders: List[ModuleSharder[nn.Module]],
    backend: str,
    dense_optim: EmbOptimType = torch.optim.SGD,
    sparse_optim: Optional[EmbOptimType] = RowWiseAdagrad,
    dense_optim_params: Optional[Dict[str, Any]] = {"lr": 1},
    sparse_optim_params: Optional[Dict[str, Any]] = {"lr": 1},
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    num_iters = 1,
    save_dir: str = "/results",
    result_file_str: Optional[str] = None,
    log_file: Optional[str] = None,
    quantization=False,
    seed = None,
    input_seed = None,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        trec_dist.comm_ops.set_gradient_division(False)
        # print(trec_dist.comm_ops.GRADIENT_DIVISION)
        
        try:
            log = open(log_file, "a")
            saved_stdout = sys.stdout
            saved_stderr = sys.stderr
            sys.stdout = log
            sys.stderr = log
            model = model.to(ctx.device)

            global_model = copy.deepcopy(model)

            global_model = global_model.to(ctx.device)
            global_input_train = inputs[0][0].to(ctx.device)
            global_input_eval = inputs[1][0].to(ctx.device)

            local_model = copy.deepcopy(model)

            if constraints is not None:
                constraints_table = {
                    table.name: constraints
                    for table in global_model.tables + global_model.weighted_tables
                }
                constraints = constraints_table
                
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    world_size, ctx.device.type, local_world_size=ctx.local_size
                ),
                constraints=constraints,
            )
            plan: ShardingPlan = planner.collective_plan(local_model, sharders, ctx.pg)

            local_model = DistributedModelParallel(
                local_model,
                env=ShardingEnv.from_process_group(ctx.pg),
                plan=plan,
                sharders=sharders,
                device=ctx.device,
            )

            local_optim_dense = KeyedOptimizerWrapper(
                dict(local_model.named_parameters()),
                lambda params: dense_optim(params, **dense_optim_params),
            )

            local_opt = CombinedOptimizer([local_model.fused_optimizer, local_optim_dense])

            # Load model state from the global model.
            copy_state_dict(local_model.state_dict(), global_model.state_dict())

            # Run a single training step of the sharded model.
            local_input_train = inputs[0][1][rank].to(ctx.device)
            local_input_eval = inputs[1][1][rank].to(ctx.device)
            for _ in range(num_iters):
                local_pred, (dense_r, sparse_r, sparse_weighted_r, over_r) = gen_full_pred_after_one_step(local_model, local_opt, local_input_train, local_input_eval)

            # record the local prediction
            all_local_pred = []
            for _ in range(world_size):
                all_local_pred.append(torch.empty_like(local_pred))
            dist.all_gather(all_local_pred, local_pred, group=ctx.pg)

            # record the local model's layer output
            all_dense_r = []
            for _ in range(world_size):
                all_dense_r.append(torch.empty_like(dense_r))
            dist.all_gather(all_dense_r, dense_r, group=ctx.pg)

            # print(sparse_r.to_dict())
            sparse_r_dict = sparse_r.to_dict()
            all_sparse_r_dict = {}
            for key in sparse_r_dict:
                all_sparse_r_dict[key] = []
                for _ in range(world_size):
                    all_sparse_r_dict[key].append(torch.empty_like(sparse_r_dict[key]))
                dist.all_gather(all_sparse_r_dict[key], sparse_r_dict[key].contiguous(), group=ctx.pg)

            sparse_weighted_r_dict = sparse_weighted_r.to_dict()
            all_sparse_weighted_r_dict = {}
            for key in sparse_weighted_r_dict:
                all_sparse_weighted_r_dict[key] = []
                for _ in range(world_size):
                    all_sparse_weighted_r_dict[key].append(torch.empty_like(sparse_weighted_r_dict[key]))
                dist.all_gather(all_sparse_weighted_r_dict[key], sparse_weighted_r_dict[key].contiguous(), group=ctx.pg)

            all_over_r = []
            for _ in range(world_size):
                all_over_r.append(torch.empty_like(over_r))
            dist.all_gather(all_over_r, over_r, group=ctx.pg)

            global_opt = dense_optim(global_model.parameters(), **dense_optim_params)
            
            for _ in range(num_iters):
                global_pred, (global_dense_r, global_sparse_r, global_sparse_weighted_r, global_over_r) = gen_full_pred_after_one_step(global_model, global_opt, global_input_train, global_input_eval)

            # Compare predictions of sharded vs unsharded models.
            if rank == 0:
                result_list = []
                result_list.append(global_pred)
                result_list.append(torch.cat(all_local_pred))
                if seed is not None and input_seed is not None and result_file_str is not None:
                    dump_pred_result_lzma(result_list, save_dir, seed, input_seed, result_file_str)
                print("global_pred vs local: ", global_pred, torch.cat(all_local_pred))
                print("Linf: ", torch.max(torch.abs(global_pred - torch.cat(all_local_pred))))
                

                all_local_pred = torch.cat(all_local_pred)

                try:
                    torch.testing.assert_close(global_pred, all_local_pred)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    with open(log_file, "a") as f:
                        f.write(str(e) + "\n")
                        f.write(traceback.format_exc() + "\n")
                    if seed is not None and input_seed is not None and result_file_str is not None:
                        # dump the intermediate layer output result
                        local_dense_r = torch.cat(all_dense_r)
                        local_sparse_r = {}
                        for key in all_sparse_r_dict:
                            local_sparse_r[key] = torch.cat(all_sparse_r_dict[key])
                        local_sparse_weighted_r = {}
                        for key in all_sparse_weighted_r_dict:
                            local_sparse_weighted_r[key] = torch.cat(all_sparse_weighted_r_dict[key])
                        local_over_r = torch.cat(all_over_r)
                        
                        intermediate_list = [global_dense_r, global_sparse_r, global_sparse_weighted_r, global_over_r, local_dense_r, local_sparse_r, local_sparse_weighted_r, local_over_r]
                        dump_intermediate_result_lzma(intermediate_list, save_dir, seed, input_seed, result_file_str)

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            with open(log_file, "a") as f:
                f.write(str(e) + "\n")
                f.write(traceback.format_exc() + "\n")
        finally:
            log.flush()
            log.close()
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr


def gen_full_pred_after_one_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    input_train: ModelInput,
    input_eval: ModelInput,
) -> torch.Tensor:
    # Run a single training step of the global model.
    opt.zero_grad()
    model.train(True)
    loss, _ = model(input_train)
    loss.backward()
    opt.step()

    # Run a forward pass of the global model.
    with torch.no_grad():
        model.train(False)
        full_pred, intermediate_list = model(input_eval, print_intermediate_layer=True)
        return full_pred, intermediate_list


def gather_shard_tensors(shard_tensor, world_size, ctx, use_empty_like=False):
    if use_empty_like:
        return gather_shard_empty_like(shard_tensor, world_size, ctx)
    else:
        return gather_shard_zeros(shard_tensor, world_size, ctx)

def gather_shard_zeros(shard_tensor, world_size, ctx):
    if shard_tensor.dtype != torch.float32:
        print("shard_tensor dtype is not float32")
    all_shard_tensor = []
    for _ in range(world_size):
        all_shard_tensor.append(torch.zeros(shard_tensor.size()).to(ctx.device))
    dist.all_gather(all_shard_tensor, shard_tensor.contiguous(), group=ctx.pg)
    return torch.cat(all_shard_tensor)

def gather_shard_empty_like(shard_tensor, world_size, ctx):
    all_shard_tensor = []
    for _ in range(world_size):
        all_shard_tensor.append(torch.empty_like(shard_tensor))
    dist.all_gather(all_shard_tensor, shard_tensor.contiguous(), group=ctx.pg)
    return torch.cat(all_shard_tensor)

def gather_all_intermediate_tensors(local_attention, world_size, ctx):
    all_local_attention = {}
    for key in local_attention.keys():
        if isinstance(local_attention[key], torch.Tensor):
            all_local_attention[key] = gather_shard_tensors(local_attention[key], world_size, ctx)
        elif isinstance(local_attention[key], tuple) or isinstance(local_attention[key], list):
            all_local_attention[key] = []
            for i in range(len(local_attention[key])):
                all_local_attention[key].append(gather_shard_tensors(local_attention[key][i], world_size, ctx))
    return all_local_attention

def get_intermediate_list(model):
    activation = {}
    hooks = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                print(name, output.size())
            else:
                print(name, len(output))
            activation[name] = output
        return hook
    for name, module in model.named_modules():
        print(name, module)
        hooks[name] = module.register_forward_hook(get_activation(name))
    return activation, hooks

def sharding_single_rank_test_sequential(
    rank: int,
    world_size: int,
    model,
    inputs,
    # embedding_groups: Dict[str, List[str]],
    sharders: List[ModuleSharder[nn.Module]],
    backend: str,
    dense_optim: EmbOptimType = torch.optim.SGD,
    sparse_optim: Optional[EmbOptimType] = RowWiseAdagrad,
    dense_optim_params: Optional[Dict[str, Any]] = {"lr": 1},
    sparse_optim_params: Optional[Dict[str, Any]] = {"lr": 1},
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    num_iters = 1,
    save_dir: str = "/results",
    result_file_str: Optional[str] = None,
    log_file: Optional[str] = None,
    quantization=False,
    seed = None,
    input_seed = None,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        try:
            log = open(log_file, "a")
            saved_stdout = sys.stdout
            saved_stderr = sys.stderr
            sys.stdout = log
            sys.stderr = log
            model = model.to(ctx.device)

            global_model = copy.deepcopy(model)
            global_model = global_model.to(ctx.device)

            global_input_train = inputs[0]

            global_input_train = inputs[0][0].to(ctx.device)
            global_input_eval = inputs[1][0].to(ctx.device)
            global_label_train = inputs[0][1].to(ctx.device)
            global_label_eval = inputs[1][1].to(ctx.device)

            local_model = copy.deepcopy(model)
              
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    world_size, ctx.device.type, local_world_size=ctx.local_size
                ),
                constraints=constraints,
            )
            plan: ShardingPlan = planner.collective_plan(local_model, sharders, ctx.pg)

            local_model = DistributedModelParallel(
                local_model,
                env=ShardingEnv.from_process_group(ctx.pg),
                plan=plan,
                sharders=sharders,
                device=ctx.device,
            )

            local_optim_dense = KeyedOptimizerWrapper(
                dict(local_model.named_parameters()),
                lambda params: dense_optim(params, **dense_optim_params),
            )

            # print("local model optimizer: ", local_model.fused_optimizer)
            local_opt = CombinedOptimizer([local_model.fused_optimizer, local_optim_dense])

            # Load model state from the global model.
            copy_state_dict(local_model.state_dict(), global_model.state_dict())
            
            # Run a single training step of the sharded model.
            local_batch_size = len(global_input_train)/world_size
            start = int(local_batch_size * ctx.rank)
            end = int(local_batch_size * (ctx.rank + 1))
            print(start, end)
            local_input_train = global_input_train[start:end].to(ctx.device)
            local_input_eval = global_input_eval[start:end].to(ctx.device)
            local_label_train = global_label_train[start:end].to(ctx.device)
            local_label_eval = global_label_eval[start:end].to(ctx.device)

            local_attentions, local_hooks = get_intermediate_list(local_model)
            for _ in range(num_iters):
                local_pred = gen_full_pred_after_one_step_sequential(local_model, local_opt, local_input_train, local_label_train, local_input_eval, local_label_eval)
            all_local_pred = gather_shard_tensors(local_pred, world_size, ctx)
            # all_local_pred = []
            # for _ in range(world_size):
            #     all_local_pred.append(torch.empty_like(local_pred))
            # dist.all_gather(all_local_pred, local_pred, group=ctx.pg)

            all_local_attentions = gather_all_intermediate_tensors(local_attentions, world_size, ctx)
            
            # Run second training step of the unsharded model.
            global_opt = dense_optim(global_model.parameters(), **dense_optim_params)
            global_attentions, global_hooks = get_intermediate_list(global_model)
            
            for _ in range(num_iters):
                global_pred = gen_full_pred_after_one_step_sequential(
                    global_model, global_opt, global_input_train, global_label_train, global_input_eval, global_label_eval
                )
            
            if rank == 0:
                result_list = []
                result_list.append(global_pred)
                result_list.append(all_local_pred)
                if seed is not None and input_seed is not None and result_file_str is not None:
                    dump_pred_result_lzma(result_list, save_dir, seed, input_seed, result_file_str)
                print("global_pred vs local: ", global_pred, all_local_pred)
                print("Linf: ", torch.max(torch.abs(global_pred - all_local_pred)))
                
                try:
                    torch.testing.assert_close(global_pred, all_local_pred)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    with open(log_file, "a") as f:
                        f.write(str(e) + "\n")
                        f.write(traceback.format_exc() + "\n")
                    if seed is not None and input_seed is not None and result_file_str is not None:
                        # dump the intermediate layer output result
                        dump_intermediate_result_sequential_lzma(local_attentions, save_dir, seed, input_seed, result_file_str, "local")
                        dump_intermediate_result_sequential_lzma(global_attentions, save_dir, seed, input_seed, result_file_str, "global")
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            with open(log_file, "a") as f:
                f.write(str(e) + "\n")
                f.write(traceback.format_exc() + "\n")
        finally:
            log.flush()
            log.close()
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr


def gen_full_pred_after_one_step_sequential(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    input_train,
    label_train,
    input_eval,
    label_eval,
) -> torch.Tensor:
    # Run a single training step of the global model.
    criterion = torch.nn.CrossEntropyLoss()
    opt.zero_grad()
    model.train(True)
    outputs = model(input_train)
    loss = criterion(outputs, label_train)
    # print(loss)
    loss.backward()
    opt.step()

    # Run a forward pass of the global model.
    with torch.no_grad():
        model.train(False)
        full_pred = model(input_eval)
        return full_pred


def get_free_port() -> int:
    if socket.has_ipv6:
        family = socket.AF_INET6
        address = "localhost6"
    else:
        family = socket.AF_INET
        address = "localhost4"
    with socket.socket(family, socket.SOCK_STREAM) as s:
        try:
            s.bind((address, 0))
            s.listen(0)
            with closing(s):
                return s.getsockname()[1]
        except socket.gaierror:
            if address == "localhost6":
                address = "::1"
            else:
                address = "127.0.0.1"
            s.bind((address, 0))
            s.listen(0)
            with closing(s):
                return s.getsockname()[1]
        except Exception as e:
            raise Exception(
                f"Binding failed with address {address} while getting free port {e}"
            )

def setUp():
    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())
    os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def run_multi_process_test(
    callable: Callable[
        ...,
        None,
    ],
    timeout: int,
    world_size: int,
    # pyre-ignore
    **kwargs,
) -> None:
    setUp()

    ctx = multiprocessing.get_context("forkserver")
    processes = []
    for rank in range(world_size):
        kwargs["rank"] = rank
        kwargs["world_size"] = world_size
        p = ctx.Process(
            target=callable,
            kwargs=kwargs,
        )
        p.start()
        processes.append(p)
    
    TIMEOUT = timeout 
    start = time.time()
    while time.time() - start <= TIMEOUT:
        if not any(p.is_alive() for p in processes):
            # All the processes are done, break now.
            break

        time.sleep(.1)  # Just to avoid hogging the CPU
    else:
        # We only enter this if we didn't 'break' above.
        print("timed out, killing all processes")
        for p in processes:
            p.terminate()

    for p in processes:
        p.join()


def main_test(
    sharders: List[ModuleSharder[nn.Module]],
    backend: str = "gloo",
    world_size: int = 2,
    local_size: Optional[int] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    model_class: Type[TestSparseNNBase] = TestModel,
    dense_optim = torch.optim.SGD,
    sparse_optim = RowWiseAdagrad,
    dense_optim_params: Optional[Dict[str, Any]] = {"lr": 10},
    sparse_optim_params: Optional[Dict[str, Any]] = {"lr": 0.1},
    test_case_name: Optional[str] = None,
    quantization: bool = False,
    model_state_dict_path = None,
    input_path = None,
    seed = None,
    input_seed = None,
    global_batch_size = 2400,
    timeout = 120,
) -> None:
    log_file_dir = os.path.join("/results", "logs", "log_{}".format(seed), "log_{}_{}".format(seed, input_seed))
    os.makedirs(log_file_dir, exist_ok=True)
    log_file = os.path.join(log_file_dir, "log_{}_{}_{}".format(seed, input_seed, test_case_name))
    with open(log_file, "w") as f:
        start = datetime.now()
        f.write("start time: {}\n".format(start))
    try:
        (model, inputs) = load_model_and_input(world_size, global_batch_size, model_class, seed, model_state_dict_path, input_path)
        if model_state_dict_path.endswith(".onnx"):
            run_multi_process_test(
                callable=sharding_single_rank_test_sequential,
                timeout = timeout,
                world_size=world_size,
                local_size=local_size,
                model=model,
                inputs=inputs,
                sharders=sharders,
                backend=backend,
                dense_optim=dense_optim,
                sparse_optim=sparse_optim,
                dense_optim_params=dense_optim_params,
                sparse_optim_params=sparse_optim_params,
                constraints=constraints,
                result_file_str=test_case_name,
                log_file=log_file,
                quantization=quantization,
                seed = seed,
                input_seed = input_seed,
            )
        else:
            run_multi_process_test(
                callable=sharding_single_rank_test,
                timeout = timeout,
                world_size=world_size,
                local_size=local_size,
                model=model,
                inputs=inputs,
                sharders=sharders,
                backend=backend,
                dense_optim=dense_optim,
                sparse_optim=sparse_optim,
                dense_optim_params=dense_optim_params,
                sparse_optim_params=sparse_optim_params,
                constraints=constraints,
                result_file_str=test_case_name,
                log_file=log_file,
                quantization=quantization,
                seed = seed,
                input_seed = input_seed,
            )
    except Exception as e:
        print(e)
        traceback.print_exc()
        with open(log_file, "a") as f:
            f.write("error: {}\n".format(e))
            f.write("traceback: {}\n".format(traceback.format_exc()))

    with open(log_file, "a") as f:
        end = datetime.now()
        f.write("end time: {}\n".format(end))
        total = end - start
        f.write("total time: {}\n".format(total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", action='store_true', help='whether to do quantization')
    parser.add_argument("--sharder_type", type=str, default=None)
    parser.add_argument("--sharding_type", type=str, default=None)
    parser.add_argument("--kernel_type", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--model_state_dict_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1, help='model seed')
    parser.add_argument("--input_seed", type=int, default=1, help='input seed/index')
    parser.add_argument("--batch_size", type=int, default=2400, help='global batch size')
    args = parser.parse_args()
    config = vars(args)


    quantization = config["quantization"]
    sharder_type = config["sharder_type"]
    sharding_type = config["sharding_type"]
    kernel_type = config["kernel_type"]
    if quantization:
        # continue
        sharders = [QuantEmbeddingBagCollectionSharder()]
    else:
        sharders = [create_test_sharder(sharder_type, sharding_type, kernel_type)]
    backend = config["backend"]
    world_size = config["world_size"]
    batch_size = config["batch_size"]

    test_case_name = str(sharder_type) + "#" + str(sharding_type) + "#" + backend + "#" + str(world_size) + "#" + str(quantization)

    main_test(
        sharders = sharders,
        backend = backend,
        world_size = world_size,
        model_class = TestModel,
        dense_optim = torch.optim.SGD,
        sparse_optim = RowWiseAdagrad,
        dense_optim_params = {"lr": 10},
        sparse_optim_params = {"lr": 10},
        test_case_name = test_case_name,
        quantization = quantization,
        model_state_dict_path = config["model_state_dict_path"],
        input_path = config["input_path"],
        seed = config["seed"],
        input_seed = config["input_seed"],
        global_batch_size = batch_size
    )

        
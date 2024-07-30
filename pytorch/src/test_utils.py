import os
import pickle
import torchrec
import lzma
# use lzma.open instead of open
import torch

def convert_awaitable_to_tensor(item):
    if isinstance(item, torchrec.distributed.embeddingbag.EmbeddingBagCollectionAwaitable):
        return item.wait()
    return item

def dump_pred_result(result_list, save_dir, seed, input_seed, result_file_str):
    result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", f"rank_0")
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir, exist_ok=True)
    # dump the result
    global_pred, local_pred = result_list
    with open(os.path.join(result_file_dir, "global_pred.p"), "wb") as f:
        pickle.dump(global_pred, f)
    with open(os.path.join(result_file_dir, "local_pred.p"), "wb") as f:
        pickle.dump(local_pred, f)

def dump_pred_result_lzma(result_list, save_dir, seed, input_seed, result_file_str):
    result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", f"rank_0")
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir, exist_ok=True)
    # dump the result
    global_pred, local_pred = result_list
    with lzma.open(os.path.join(result_file_dir, "global_pred.xz"), "wb") as f:
        pickle.dump(global_pred, f)
    with lzma.open(os.path.join(result_file_dir, "local_pred.xz"), "wb") as f:
        pickle.dump(local_pred, f)

def load_pred_result(save_dir, seed, input_seed, result_file_str):
    result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", f"rank_0")
    # print("in load_pred_result: ", result_file_dir)
    with open(os.path.join(result_file_dir, "global_pred.p"), "rb") as f:
        global_pred = pickle.load(f)
    with open(os.path.join(result_file_dir, "local_pred.p"), "rb") as f:
        local_pred = pickle.load(f)
    return [global_pred, local_pred]

def load_pred_result_lzma(save_dir, seed, input_seed, result_file_str):
    result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", f"rank_0")
    with lzma.open(os.path.join(result_file_dir, "global_pred.xz"), "rb") as f:
        global_pred = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, "local_pred.xz"), "rb") as f:
        local_pred = pickle.load(f)
    return [global_pred, local_pred]

def dump_intermediate_result(intermediate_list, save_dir, seed, input_seed, result_file_str):
    result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", "rank_0_intermediate")
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir, exist_ok=True)
    # dump the result
    # dump the intermediate layer output result
    global_dense_r, global_sparse_r, global_sparse_weighted_r, global_over_r, local_dense_r, local_sparse_r, local_sparse_weighted_r, local_over_r = intermediate_list
    with open(os.path.join(result_file_dir, f"global_dense_r.p"), "wb") as f:
        pickle.dump(global_dense_r, f)
    with open(os.path.join(result_file_dir, f"global_sparse_r.p"), "wb") as f:
        pickle.dump(global_sparse_r, f)
    with open(os.path.join(result_file_dir, f"global_sparse_weighted_r.p"), "wb") as f:
        pickle.dump(global_sparse_weighted_r, f)
    with open(os.path.join(result_file_dir, f"global_over_r.p"), "wb") as f:
        pickle.dump(global_over_r, f)
    with open(os.path.join(result_file_dir, f"local_dense_r.p"), "wb") as f:
        pickle.dump(local_dense_r, f)
    with open(os.path.join(result_file_dir, f"local_sparse_r.p"), "wb") as f:
        pickle.dump(local_sparse_r, f)
    with open(os.path.join(result_file_dir, f"local_sparse_weighted_r.p"), "wb") as f:
        pickle.dump(local_sparse_weighted_r, f)
    with open(os.path.join(result_file_dir, f"local_over_r.p"), "wb") as f:
        pickle.dump(local_over_r, f)

def dump_intermediate_result_lzma(intermediate_list, save_dir, seed, input_seed, result_file_str):
    result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", "rank_0_intermediate")
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir, exist_ok=True)
    # dump the result
    # dump the intermediate layer output result
    global_dense_r, global_sparse_r, global_sparse_weighted_r, global_over_r, local_dense_r, local_sparse_r, local_sparse_weighted_r, local_over_r = intermediate_list
    with lzma.open(os.path.join(result_file_dir, f"global_dense_r.xz"), "wb") as f:
        pickle.dump(global_dense_r, f)
    with lzma.open(os.path.join(result_file_dir, f"global_sparse_r.xz"), "wb") as f:
        pickle.dump(global_sparse_r, f)
    with lzma.open(os.path.join(result_file_dir, f"global_sparse_weighted_r.xz"), "wb") as f:
        pickle.dump(global_sparse_weighted_r, f)
    with lzma.open(os.path.join(result_file_dir, f"global_over_r.xz"), "wb") as f:
        pickle.dump(global_over_r, f)
    with lzma.open(os.path.join(result_file_dir, f"local_dense_r.xz"), "wb") as f:
        pickle.dump(local_dense_r, f)
    with lzma.open(os.path.join(result_file_dir, f"local_sparse_r.xz"), "wb") as f:
        pickle.dump(local_sparse_r, f)
    with lzma.open(os.path.join(result_file_dir, f"local_sparse_weighted_r.xz"), "wb") as f:
        pickle.dump(local_sparse_weighted_r, f)
    with lzma.open(os.path.join(result_file_dir, f"local_over_r.xz"), "wb") as f:
        pickle.dump(local_over_r, f)


def load_intermediate_result(save_dir, seed, input_seed, result_file_str):
    result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", "rank_0_intermediate")
    # print(result_file_dir)
    with open(os.path.join(result_file_dir, f"global_dense_r.p"), "rb") as f:
        global_dense_r = pickle.load(f)
    with open(os.path.join(result_file_dir, f"global_sparse_r.p"), "rb") as f:
        global_sparse_r = pickle.load(f)
    with open(os.path.join(result_file_dir, f"global_sparse_weighted_r.p"), "rb") as f:
        global_sparse_weighted_r = pickle.load(f)
    with open(os.path.join(result_file_dir, f"global_over_r.p"), "rb") as f:
        global_over_r = pickle.load(f)
    with open(os.path.join(result_file_dir, f"local_dense_r.p"), "rb") as f:
        local_dense_r = pickle.load(f)
    with open(os.path.join(result_file_dir, f"local_sparse_r.p"), "rb") as f:
        local_sparse_r = pickle.load(f)
    with open(os.path.join(result_file_dir, f"local_sparse_weighted_r.p"), "rb") as f:
        local_sparse_weighted_r = pickle.load(f)
    with open(os.path.join(result_file_dir, f"local_over_r.p"), "rb") as f:
        local_over_r = pickle.load(f)
    return [global_dense_r, global_sparse_r, global_sparse_weighted_r, global_over_r, local_dense_r, local_sparse_r, local_sparse_weighted_r, local_over_r]

def load_intermediate_result_lzma(save_dir, seed, input_seed, result_file_str):
    result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", "rank_0_intermediate")
    with lzma.open(os.path.join(result_file_dir, f"global_dense_r.xz"), "rb") as f:
        global_dense_r = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, f"global_sparse_r.xz"), "rb") as f:
        global_sparse_r = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, f"global_sparse_weighted_r.xz"), "rb") as f:
        global_sparse_weighted_r = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, f"global_over_r.xz"), "rb") as f:
        global_over_r = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, f"local_dense_r.xz"), "rb") as f:
        local_dense_r = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, f"local_sparse_r.xz"), "rb") as f:
        local_sparse_r = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, f"local_sparse_weighted_r.xz"), "rb") as f:
        local_sparse_weighted_r = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, f"local_over_r.xz"), "rb") as f:
        local_over_r = pickle.load(f)
    return [global_dense_r, global_sparse_r, global_sparse_weighted_r, global_over_r, local_dense_r, local_sparse_r, local_sparse_weighted_r, local_over_r]

def dump_intermediate_result_sequential_lzma(intermediate_dict, save_dir, seed, input_seed, result_file_str, folder=None):
    if folder is None:
        result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", "rank_0_intermediate")
    else:
        result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", "rank_0_intermediate", folder)
    # print(result_file_dir)
    # print("if dir exists", os.path.exists(result_file_dir))
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir, exist_ok=True)
    # print("if dir exists", os.path.exists(result_file_dir))
    # for key, item in intermediate_dict.items():
    #     with lzma.open(os.path.join(result_file_dir, f"{key}.xz"), "wb") as f:
    #         pickle.dump(item, f)
    with lzma.open(os.path.join(result_file_dir, "rank_0_intermediate.xz"), "wb") as f:
        pickle.dump(intermediate_dict, f)

def load_intermediate_result_sequential_lzma(save_dir, seed, input_seed, result_file_str, folder=None):
    if folder is None:
        result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", "rank_0_intermediate")
    else:
        result_file_dir = os.path.join(save_dir, "outputs", f"output_{seed}", f"output_{seed}_{input_seed}", f"output_{seed}_{input_seed}_{result_file_str}", "rank_0_intermediate", folder)
    # intermediate_dict = {}
    # for file in os.listdir(result_file_dir):
    #     with lzma.open(os.path.join(result_file_dir, file), "rb") as f:
    #         intermediate_dict[file.split(".")[0]] = pickle.load(f)
    with lzma.open(os.path.join(result_file_dir, "rank_0_intermediate.xz"), "rb") as f:
        intermediate_dict = pickle.load(f)
    return intermediate_dict

def compare_tensor(tensor_1, tensor_2):
    diff_dense = torch.max(torch.abs(tensor_1.cpu() - tensor_2.cpu()))
    diff_dense_max_1 = torch.max(torch.abs(tensor_1.cpu()))
    diff_dense_max_2 = torch.max(torch.abs(tensor_2.cpu()))
    return diff_dense

def compare_keyedtensor(keyed_tensor_1, keyed_tensor_2):
    try:
        safe_compare_intermidiate_results(keyed_tensor_1.values().cpu(), keyed_tensor_2.values().cpu())
        # first convert keyed tensor to dict for comparision
        max_diff_sparse = torch.tensor(0.0)
        keyed_tensor_1 = keyed_tensor_1.to_dict()
        keyed_tensor_2 = keyed_tensor_2.to_dict()
        for key in keyed_tensor_1:
            # print(torch.cat(all_sparse_r_dict_1[key]).size())
            diff_sparse = torch.max(torch.abs(keyed_tensor_1[key].cpu() - keyed_tensor_2[key].cpu()))
            diff_sparse_dim_1, indices_1 = torch.max(torch.abs(keyed_tensor_1[key].cpu() - keyed_tensor_2[key].cpu()), dim=1)
            diff_sparse_dim_0, indices_0 = torch.max(diff_sparse_dim_1, dim=0)
            
            diff_sparse_max_1 = torch.max(torch.abs(keyed_tensor_1[key].cpu()))
            diff_sparse_max_2 = torch.max(torch.abs(keyed_tensor_2[key].cpu()))
            if diff_sparse > max_diff_sparse:
                max_diff_sparse = diff_sparse
        return max_diff_sparse
    except:
        # if failed, compare value only
        try:
            diff_sparse = torch.max(torch.abs(keyed_tensor_1.values().cpu() - keyed_tensor_2.values().cpu()))
            diff_sparse_max_1 = torch.max(torch.abs(keyed_tensor_1.values().cpu()))
            diff_sparse_max_2 = torch.max(torch.abs(keyed_tensor_2.values().cpu()))
            return diff_sparse
        except:
            return torch.tensor(-100.0)
            pass

def safe_compare_intermidiate_results(ir_result_1, ir_result_2):
    size_1 = ir_result_1.size()
    size_2 = ir_result_2.size()
    # print(size_1, size_2)
    # print(ir_result_1.dtype, ir_result_2.dtype)
    # assert size_1[0] == size_2[0] and size_1[1] > size_2[1]
    # print(ir_result_1[0, :10])
    # print(ir_result_2[0, :10])
    # print(ir_result_1[0, -10:])
    # print(ir_result_2[0, -10:])
    index_1 = 0
    index_2 = 0
    skip_index_list = []
    while index_1 < size_1[1] and index_2 < size_2[1]:
        if torch.max(torch.abs(ir_result_1[:, index_1].cpu() - ir_result_2[:, index_2].cpu())) < 1e-5:
            index_1 += 1
            index_2 += 1
        else:
            skip_index_list.append(index_1)
            index_1 += 1
    # print("size_1: ", size_1)
    # print("size_2: ", size_2)
    # # print("skip_index_list: ", skip_index_list)
    # print("index_1: ", index_1)
    # print("index_2: ", index_2)
    # return skip_index_list


def compare_intermidiate_results(ir_result_1, ir_result_2):
    all_dense_r_1, all_sparse_r_dict_1, all_sparse_weighted_r_dict_1, all_over_r_1 = ir_result_1
    # print(ir_result_2)
    all_dense_r_2, all_sparse_r_dict_2, all_sparse_weighted_r_dict_2, all_over_r_2 = ir_result_2

    diff_dense = compare_tensor(all_dense_r_1, all_dense_r_2)

    # try:
    #     all_sparse_r_dict_1.to_dict()
    #     # print("all_sparse_r_dict_1 is a keyed tensor")
    # except:
    #     # print("all_sparse_r_dict_1 is not a keyed tensor")
    #     pass
    # try:
    #     all_sparse_r_dict_2.to_dict()
    #     # print("all_sparse_r_dict_2 is a keyed tensor")
    # except:
    #     # print("all_sparse_r_dict_2 is not a keyed tensor")
    #     pass

    max_diff_sparse = compare_keyedtensor(all_sparse_r_dict_1, all_sparse_r_dict_2)

    # try:
    #     all_sparse_weighted_r_dict_1.to_dict()
    #     print("all_sparse_weighted_r_dict_1 is a keyed tensor")
    # except:
    #     print("all_sparse_weighted_r_dict_1 is not a keyed tensor")
    # try:
    #     all_sparse_weighted_r_dict_2.to_dict()
    #     print("all_sparse_weighted_r_dict_2 is a keyed tensor")
    # except:
    #     print("all_sparse_weighted_r_dict_2 is not a keyed tensor")

    max_diff_sparse_weighted = compare_keyedtensor(all_sparse_weighted_r_dict_1, all_sparse_weighted_r_dict_2)

    diff_over = compare_tensor(all_over_r_1, all_over_r_2)
    return [diff_dense.item(), max_diff_sparse.item(), max_diff_sparse_weighted.item(), diff_over.item()]
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection
from torchrec.sparse.jagged_tensor import KeyedTensor


class TestModel(nn.Module):
    def __init__(
        self,
        tables=None,
        weighted_tables=None,
        dense_device=None,
        sparse_device=None,
        num_float_features=None,
        seed=None,
    ):
        super().__init__()
        seed = self.fix_seed(seed)

        self.layer_gen = LayerGenerator()

        if tables is None:
            self.tables = self.layer_gen.gen_EmbeddingBagConfigs()
        else:
            self.tables = tables

        if weighted_tables is None:
            self.weighted_tables = self.layer_gen.gen_EmbeddingBagConfigs(
                name_prefix="weighted_"
            )
        else:
            self.weighted_tables = weighted_tables

        if num_float_features is None:
            self.in_features = random.randint(1, 1000)
        else:
            self.in_features = num_float_features

        self.dense_out_features = random.randint(1, 1000)
        self.over_out_features = random.randint(1, 1000)

        self.dense_device = dense_device
        self.sparse_device = sparse_device

        self.dense = self.layer_gen.gen_Linear(
            in_features=self.in_features,
            out_features=self.dense_out_features,
            device=self.dense_device,
        )
        self.sparse = self.layer_gen.gen_EmbeddingBagCollection(
            tables=self.tables, device=self.sparse_device
        )
        self.sparse_weighted = self.layer_gen.gen_EmbeddingBagCollection(
            tables=self.weighted_tables, 
            is_weighted=True, 
            device=self.sparse_device
        )

        in_features_concat = (
            self.dense_out_features
            + sum(
                [
                    table.embedding_dim * len(table.feature_names)
                    for table in self.tables
                ]
            )
            + sum(
                [
                    table.embedding_dim * len(table.feature_names)
                    for table in self.weighted_tables
                ]
            )
        )

        self.over = self.layer_gen.gen_Linear(
            in_features=in_features_concat, out_features=self.over_out_features, device=self.dense_device
        )

    def forward(
        self,
        input: ModelInput,
        print_intermediate_layer = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        sparse_r = self.sparse(input.idlist_features)
        sparse_weighted_r = self.sparse_weighted(input.idscore_features)
        result = KeyedTensor(
            keys=sparse_r.keys() + sparse_weighted_r.keys(),
            length_per_key=sparse_r.length_per_key()
            + sparse_weighted_r.length_per_key(),
            values=torch.cat([sparse_r.values(), sparse_weighted_r.values()], dim=1),
        )

        _features = [
            feature for table in self.tables for feature in table.feature_names
        ]
        _weighted_features = [
            feature for table in self.weighted_tables for feature in table.feature_names
        ]

        ret_list = []
        ret_list.append(dense_r)
        for feature_name in _features:
            ret_list.append(result[feature_name])
        for feature_name in _weighted_features:
            ret_list.append(result[feature_name])
        ret_concat = torch.cat(ret_list, dim=1)

        over_r = self.over(ret_concat)
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training and not print_intermediate_layer:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        elif self.training and print_intermediate_layer:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
                (dense_r, sparse_r, sparse_weighted_r, over_r)
            )
        elif not self.training and print_intermediate_layer:
            return (
                pred,
                (dense_r, sparse_r, sparse_weighted_r, over_r)
            )
        else:
            return pred

    def fix_seed(self, seed=None):
        if seed is None:
            return
        print(f"Model Gen Using random seed: {seed}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return seed


class LayerGenerator(object):
    def __init__(self):
        super().__init__()
        # pass

    def gen_EmbeddingBagCollection(self, tables=None, is_weighted=False, device=None):
        if tables is None:
            tables = self.gen_EmbeddingTableConfig()
        if device is None:
            device = torch.device("cpu")
        layer: EmbeddingBagCollection = EmbeddingBagCollection(
            tables=tables,
            is_weighted=is_weighted,
            device=device,
        )
        return layer

    def gen_EmbeddingBagConfigs(self, num_features=None, name_prefix=None):
        if num_features is None:
            num_features = random.randint(1, 5)

        name_str = "table_"
        if name_prefix is not None:
            name_str = name_prefix + name_str
        feature_names_str = "feature_"
        if name_prefix is not None:
            feature_names_str = name_prefix + feature_names_str

        tables = [
            EmbeddingBagConfig(
                num_embeddings=random.randint(1, 1000),
                embedding_dim=random.randint(1, 250) * 4,
                name=name_str + str(i),
                feature_names=[feature_names_str + str(i)],
            )
            for i in range(num_features)
        ]
        return tables

    def gen_Linear(self, in_features=None, out_features=None, device=None):
        if in_features is None:
            in_features = random.randint(1, 1000)
        if out_features is None:
            out_features = random.randint(1, 1000)
        if device is None:
            device = torch.device("cpu")
        layer: nn.modules.Linear = nn.Linear(
            in_features=in_features, out_features=out_features, device=device
        )
        return layer

    def gen_TowerInteraction(self, table=None, device=None):
        if table is None:
            table = self.gen_EmbeddingTableConfig()
        if device is None:
            device = torch.device("cpu")

        class TowerInteraction(nn.Module):
            def __init__(
                self,
                tables: List[EmbeddingBagConfig],
                device: Optional[torch.device] = None,
            ) -> None:
                super().__init__()
                if device is None:
                    device = torch.device("cpu")
                self._features: List[str] = [
                    feature for table in tables for feature in table.feature_names
                ]
                in_features = sum(
                    [table.embedding_dim * len(table.feature_names) for table in tables]
                )
                self.linear: nn.modules.Linear = nn.Linear(
                    in_features=in_features,
                    out_features=in_features,
                    device=device,
                )

            def forward(
                self,
                sparse: KeyedTensor,
            ) -> torch.Tensor:
                ret_list = []
                for feature_name in self._features:
                    ret_list.append(sparse[feature_name])
                return self.linear(torch.cat(ret_list, dim=1))

        layer: TowerInteraction = TowerInteraction(
            tables=table,
            device=device,
        )
        return layer

    def gen_EmbeddingTower(
        self, embedding_module=None, interaction_module=None, device=None
    ):
        if embedding_module is None:
            embedding_module = self.gen_EmbeddingBagCollection()
        if interaction_module is None:
            interaction_module = self.gen_TowerInteraction()
        if device is None:
            device = torch.device("cpu")

        layer: EmbeddingTower = EmbeddingTower(
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            device=device,
        )
        return layer

    def gen_EmbeddingTowerCollection(self, towers=None, device=None):
        if towers is None:
            n = random.randint(1, 10)
            towers = [self.gen_EmbeddingTower() for _ in range(n)]
        if device is None:
            device = torch.device("cpu")

        layer: EmbeddingTowerCollection = EmbeddingTowerCollection(
            towers=towers, device=device
        )
        return layer


class TestTowerCollectionModel(nn.Module):
    def __init__(
        self,
        tables=None,
        weighted_tables=None,
        dense_device=None,
        sparse_device=None,
        num_float_features=None,
    ):
        super().__init__()
        self.layer_gen = LayerGenerator()

        if tables is None:
            self.tables = self.layer_gen.gen_EmbeddingBagConfigs()
        else:
            self.tables = tables

        self.tables_by_group = []
        index_s = 0
        while index_s < len(self.tables):
            index_e = index_s + random.randint(1, 4)
            if index_e > len(self.tables):
                index_e = len(self.tables)
            self.tables_by_group.append(self.tables[index_s:index_e])
            index_s = index_e

        if weighted_tables is None:
            self.weighted_tables = self.layer_gen.gen_EmbeddingBagConfigs(
                name_prefix="weighted_"
            )
        else:
            self.weighted_tables = weighted_tables

        self.weighted_tables_by_group = []
        index_s = 0
        while index_s < len(self.weighted_tables):
            index_e = index_s + random.randint(1, 4)
            if index_e > len(self.weighted_tables):
                index_e = len(self.weighted_tables)
            self.weighted_tables_by_group.append(self.weighted_tables[index_s:index_e])
            index_s = index_e

        if num_float_features is None:
            self.in_features = random.randint(0, 1000)
        else:
            self.in_features = num_float_features

        self.out_features = random.randint(0, 1000)

        self.dense_device = dense_device
        self.sparse_device = sparse_device

        self.dense = self.layer_gen.gen_Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            device=self.dense_device,
        )

        self.towers = []
        for tables in self.tables_by_group:
            self.towers.append(
                self.layer_gen.gen_EmbeddingTower(
                    embedding_module=self.layer_gen.gen_EmbeddingBagCollection(
                        tables=tables, device=self.sparse_device
                    ),
                    interaction_module=self.layer_gen.gen_TowerInteraction(
                        table=tables, device=self.sparse_device
                    ),
                    device=self.sparse_device,
                )
            )
        self.weighted_towers = []
        for tables in self.weighted_tables_by_group:
            self.weighted_towers.append(
                self.layer_gen.gen_EmbeddingTower(
                    embedding_module=self.layer_gen.gen_EmbeddingBagCollection(
                        tables=tables, is_weighted=True, device=self.sparse_device
                    ),
                    interaction_module=self.layer_gen.gen_TowerInteraction(
                        table=tables, device=self.sparse_device
                    ),
                    device=self.sparse_device,
                )
            )

        self.sparse = self.layer_gen.gen_EmbeddingTowerCollection(
            towers=self.towers + self.weighted_towers, device=self.sparse_device
        )

        in_features_concat = (
            self.out_features
            + sum([tower.interaction.linear.out_features for tower in self.towers])
            + sum(
                [
                    weighted_tower.interaction.linear.out_features
                    for weighted_tower in self.weighted_towers
                ]
            )
        )

        self.over = self.layer_gen.gen_Linear(
            in_features=in_features_concat, out_features=None, device=self.dense_device
        )

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        sparse_r = self.sparse(input.idlist_features, input.idscore_features)
        over_r = self.over(torch.cat([dense_r, sparse_r], dim=1))
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred


if __name__ == "__main__":
    model = TestModel()
    print(model)
    # input_gen = ModelInput()
    input = ModelInput.generate(
        2, 1, model.in_features, model.tables, model.weighted_tables
    )
    # print(input)
    # print(input[0].float_features)
    print(input[0].float_features.shape)
    idlist_features = input[0].idlist_features
    print(idlist_features.to_dict()['feature_0'])
    print(idlist_features._lengths)
    print(idlist_features._values)
    idscore_features = input[0].idscore_features
    print(idscore_features.to_dict()['weighted_feature_0'])

    result = model(input[0])
    print(result)

    # model = TestTowerCollectionModel()
    # input = ModelInput.generate(
    #     1, 1, model.in_features, model.tables, model.weighted_tables
    # )
    # print(input)
    # print(input[0].float_features)
    # print(input[0].float_features.shape)
    # result = model(input[0])
    # print(result)

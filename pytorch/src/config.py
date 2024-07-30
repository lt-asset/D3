from torchrec.distributed.test_utils.test_sharding import (
    SharderType,
)
from torchrec.distributed.types import (
    ShardingType,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel

class DistributedConfig:
    def __init__(self, sharder_type, sharding_type, kernel_type, world_size, backend, quantization):
        self.sharder_type = sharder_type
        self.sharding_type = sharding_type
        self.kernel_type = kernel_type
        self.world_size = world_size
        self.backend = backend
        self.quantization = quantization


distributed_settings = [
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "gloo", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "gloo", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 1, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 2, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 3, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 4, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 8, "gloo", False),

    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 1, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 2, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 3, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 4, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 8, "gloo", False),
    
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 1, "gloo", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 2, "gloo", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 3, "gloo", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 4, "gloo", True),

    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 1, "nccl", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 2, "nccl", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 3, "nccl", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 4, "nccl", True),
]


# according to https://pytorch.org/docs/stable/quantization.html, we can use the following quantization configuration
# dtype: torch.qint8, torch.float16
class QuantizationConfig:
    def __init__(self, dtype, output_dtype):
        self.quant_dtype = dtype
        self.quant_output_dtype = output_dtype

# all data types:
# 2 (non-distributed-non-quantized) + 55 (distributed-non-quantized) + 4 (non-distributed-quantized) + 16 (distributed-quantized) = 77
# quantization settings is all combination of dtype and output_dtype
quantization_settings = [
    QuantizationConfig("torch.qint8", "torch.qint8"),
    QuantizationConfig("torch.qint8", "torch.float16"),
    QuantizationConfig("torch.qint8", "torch.float32"),
]
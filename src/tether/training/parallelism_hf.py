from __future__ import annotations

import logging

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

logger = logging.getLogger(__name__)


def _divisors(n: int) -> list[int]:
    return [i for i in range(1, n + 1) if n % i == 0]


def validate_tp_size_hf(config, tp_size: int) -> None:
    if tp_size <= 1:
        return

    if config.num_attention_heads % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} does not evenly divide "
            f"num_attention_heads={config.num_attention_heads}. "
            f"Valid TP sizes: {_divisors(config.num_attention_heads)}"
        )

    if config.num_key_value_heads % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} does not evenly divide "
            f"num_key_value_heads={config.num_key_value_heads}. "
            f"Valid TP sizes: {_divisors(config.num_key_value_heads)}"
        )

    if config.intermediate_size % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} does not evenly divide "
            f"intermediate_size={config.intermediate_size}."
        )

    if config.vocab_size % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} does not evenly divide "
            f"vocab_size={config.vocab_size}."
        )


def apply_tp_hf(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    for block in model.model.layers:
        attn_plan = {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(output_layouts=Replicate()),
        }
        parallelize_module(module=block.self_attn, device_mesh=tp_mesh, parallelize_plan=attn_plan)

        mlp_plan = {
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(output_layouts=Replicate()),
        }
        parallelize_module(module=block.mlp, device_mesh=tp_mesh, parallelize_plan=mlp_plan)

    root_plan = {
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "lm_head": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
    }
    parallelize_module(module=model, device_mesh=tp_mesh, parallelize_plan=root_plan)

    logger.info("Applied tensor parallelism to HF model (tp_size=%d)", tp_mesh.size())


def apply_dp_hf(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
) -> None:
    fsdp_kwargs = {"mesh": dp_mesh, "mp_policy": mp_policy}

    n_layers = len(model.model.layers)
    for layer_idx, block in enumerate(model.model.layers):
        reshard = layer_idx < n_layers - 1
        fully_shard(block, **fsdp_kwargs, reshard_after_forward=reshard)

    fully_shard(model, **fsdp_kwargs)

    logger.info("Applied FSDP2 data parallelism to HF model (dp_size=%d)", dp_mesh.size())

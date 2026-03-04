import logging
import os

import torch
import torch.distributed as dist
import yaml
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

YAML_TO_ARG = {
    "hf_checkpoint": "model",
    "resume_checkpoint": "resume",
    "tp_size": "tp_size",
    "datamix_config": "datamix",
    "seq_len": "seq_len",
    "learning_rate": "lr",
    "lr_schedule": "lr_schedule",
    "min_lr_ratio": "min_lr_ratio",
    "weight_decay": "weight_decay",
    "beta1": "beta1",
    "beta2": "beta2",
    "max_grad_norm": "max_grad_norm",
    "warmup_steps": "warmup_steps",
    "total_steps": "total_steps",
    "batch_size_per_gpu": "batch_size",
    "gradient_accumulation_steps": "gradient_accumulation",
    "mixed_precision_param_dtype": "param_dtype",
    "mixed_precision_reduce_dtype": "reduce_dtype",
    "dtype": "param_dtype",
    "checkpoint_dir": "checkpoint_dir",
    "save_interval": "save_interval",
    "log_interval": "log_interval",
    "num_workers": "num_workers",
    "log_dir": "log_dir",
}


def apply_yaml_overrides(parser, args, extra_mappings=None):
    if not args.config:
        return
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    mapping = dict(YAML_TO_ARG)
    if extra_mappings:
        mapping.update(extra_mappings)
    defaults = parser.parse_args([])
    for yaml_key, arg_name in mapping.items():
        if yaml_key in yaml_cfg and getattr(args, arg_name) == getattr(defaults, arg_name):
            setattr(args, arg_name, yaml_cfg[yaml_key])


def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def init_mesh(world_size, tp_size):
    if world_size % tp_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by tp_size={tp_size}")
    dp_size = world_size // tp_size
    if tp_size > 1:
        mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        return mesh_2d["dp"], mesh_2d["tp"], dp_size
    return init_device_mesh("cuda", (world_size,)), None, dp_size


def load_model(model_name, device, tp_mesh, dp_mesh, param_dtype, reduce_dtype):
    from transformers import AutoModelForCausalLM
    from tether.training.parallelism_hf import apply_dp_hf, apply_tp_hf, validate_tp_size_hf

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="flash_attention_2",
    ).to(device)

    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    validate_tp_size_hf(model.config, tp_size)
    if tp_mesh is not None:
        apply_tp_hf(model, tp_mesh)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, param_dtype),
        reduce_dtype=getattr(torch, reduce_dtype),
    )
    apply_dp_hf(model, dp_mesh, mp_policy)
    return model


def build_optimizer(model, args):
    return torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, args):
    from tether.training.lr_schedule import constant_with_warmup, cosine_with_warmup, linear_with_warmup

    schedulers = {
        "cosine": lambda: cosine_with_warmup(optimizer, args.warmup_steps, args.total_steps, args.min_lr_ratio),
        "constant": lambda: constant_with_warmup(optimizer, args.warmup_steps),
        "linear": lambda: linear_with_warmup(optimizer, args.warmup_steps, args.total_steps, args.min_lr_ratio),
    }
    return schedulers[args.lr_schedule]()


def build_dataloader(args, dp_rank, dp_size, resume_step=0):
    from tether.training.datasets import PackedIterableDataset
    from tether.utils.config import DataMixConfig

    datamix = DataMixConfig.from_yaml(args.datamix)
    dataset = PackedIterableDataset(datamix, rank=dp_rank, world_size=dp_size)
    if resume_step > 0:
        dataset.samples_to_skip = resume_step * args.batch_size * args.gradient_accumulation
    return DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )


def export_hf(model, model_name, export_path, rank):
    from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
    from transformers import AutoModelForCausalLM

    full_sd, _ = get_state_dict(
        model, [], options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if rank == 0:
        export_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        export_model.load_state_dict(full_sd)
        export_model.save_pretrained(export_path)
        logger.info("Exported HF checkpoint to %s", export_path)
    dist.barrier()


def clip_grad_norm_2d(model, max_norm, device):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    local_norm_sq = torch.zeros(1, device=device)
    for g in grads:
        local_g = g.to_local() if hasattr(g, "to_local") else g
        local_norm_sq += local_g.detach().float().norm(2.0) ** 2
    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
    grad_norm = local_norm_sq.sqrt().item()
    clip_coef = max_norm / max(grad_norm, 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.detach().mul_(clip_coef)
    return grad_norm


def setup_tensorboard(log_dir, rank):
    if log_dir and rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    return None


def log_scalars(tb_writer, metrics, step):
    if tb_writer:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"train/{k}", v, step)


def add_common_args(parser):
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--datamix", type=str, default="test_mix.yaml")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "constant", "linear"])
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--param-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--reduce-dtype", type=str, default="float32")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/default")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--export-hf", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-dir", type=str, default=None)
    return parser

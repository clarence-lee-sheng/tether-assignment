import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distributed continued pretraining with HF model (DTensor TP + FSDP2 DP)"
    )

    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--resume", type=str, default=None, help="DCP checkpoint dir to resume from")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallelism degree; DP = world_size // tp_size")

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
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=4)

    parser.add_argument("--param-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--reduce-dtype", type=str, default="float32")

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/smollm2_135m_hf")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--export-hf", type=str, default=None, help="Export final model to HF format at this path")

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory")

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)

        yaml_to_arg = {
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
        defaults = parser.parse_args([])
        for yaml_key, arg_name in yaml_to_arg.items():
            if yaml_key in yaml_cfg and getattr(args, arg_name) == getattr(defaults, arg_name):
                setattr(args, arg_name, yaml_cfg[yaml_key])

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tp_size = args.tp_size
    if world_size % tp_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tp_size={tp_size}"
        )
    dp_size = world_size // tp_size

    if rank == 0:
        logger.info(
            "World size: %d | TP size: %d | DP size: %d", world_size, tp_size, dp_size
        )

    if tp_size > 1:
        mesh_2d = init_device_mesh(
            "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
    else:
        dp_mesh = init_device_mesh("cuda", (world_size,))
        tp_mesh = None

    if tp_size > 1:
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_rank = rank

    from transformers import AutoModelForCausalLM

    if rank == 0:
        logger.info("Loading HuggingFace model: %s", args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2",
    )
    param_count = sum(p.numel() for p in model.parameters())

    if rank == 0:
        logger.info("Model parameters: %s", f"{param_count:,}")

    model = model.to(device)

    from tether.training.parallelism_hf import apply_dp_hf, apply_tp_hf, validate_tp_size_hf

    validate_tp_size_hf(model.config, tp_size)

    if tp_mesh is not None:
        apply_tp_hf(model, tp_mesh)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, args.param_dtype),
        reduce_dtype=getattr(torch, args.reduce_dtype),
    )
    apply_dp_hf(model, dp_mesh, mp_policy)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    from tether.training.lr_schedule import constant_with_warmup, cosine_with_warmup, linear_with_warmup

    if args.lr_schedule == "cosine":
        scheduler = cosine_with_warmup(
            optimizer, args.warmup_steps, args.total_steps, args.min_lr_ratio
        )
    elif args.lr_schedule == "constant":
        scheduler = constant_with_warmup(optimizer, args.warmup_steps)
    else:
        scheduler = linear_with_warmup(
            optimizer, args.warmup_steps, args.total_steps, args.min_lr_ratio
        )

    from tether.training.checkpoint import TrainingState, load_checkpoint, save_checkpoint

    training_state = TrainingState(model, optimizer, scheduler)
    if args.resume:
        load_checkpoint(training_state, args.resume)
        if rank == 0:
            logger.info(
                "Resumed from step %d (tokens_seen=%d)",
                training_state.step,
                training_state.tokens_seen,
            )

    from tether.training.datasets import PackedIterableDataset
    from tether.utils.config import DataMixConfig

    datamix = DataMixConfig.from_yaml(args.datamix)
    dataset = PackedIterableDataset(datamix, rank=dp_rank, world_size=dp_size)

    if training_state.step > 0:
        dataset.samples_to_skip = (
            training_state.step * args.batch_size * args.gradient_accumulation
        )
        if rank == 0:
            logger.info("Resuming dataset: skipping %d samples per worker", dataset.samples_to_skip)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    tb_writer = None
    if args.log_dir and rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=args.log_dir)
        logger.info("TensorBoard logging to %s", args.log_dir)

    use_loss_parallel = tp_size > 1
    model.train()
    data_iter = iter(dataloader)
    step = training_state.step

    effective_batch = args.batch_size * args.gradient_accumulation * dp_size
    if rank == 0:
        logger.info(
            "Effective batch size: %d (per_gpu=%d x grad_accum=%d x dp=%d)",
            effective_batch,
            args.batch_size,
            args.gradient_accumulation,
            dp_size,
        )

    while step < args.total_steps:
        step += 1
        t0 = time.time()
        total_loss = 0.0
        total_tokens = 0

        for micro_step in range(args.gradient_accumulation):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            labels = input_ids

            is_last_micro = micro_step == args.gradient_accumulation - 1
            model.set_requires_gradient_sync(is_last_micro)

            if use_loss_parallel:
                from torch.distributed.tensor.parallel import loss_parallel

                output = model(input_ids)
                shift_logits = output.logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                with loss_parallel():
                    loss = F.cross_entropy(
                        shift_logits.flatten(0, 1),
                        shift_labels.flatten(0, 1),
                        ignore_index=-100,
                    )
                    scaled_loss = loss / args.gradient_accumulation
                    scaled_loss.backward()
            else:
                output = model(input_ids, labels=labels)
                loss = output.loss
                scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()

            total_loss += loss.detach().item()
            total_tokens += input_ids.numel()

        # Manual grad norm computation for 2D parallelism (params live on different meshes)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        local_norm_sq = torch.zeros(1, device=device)
        for g in grads:
            local_g = g.to_local() if hasattr(g, "to_local") else g
            local_norm_sq += local_g.detach().float().norm(2.0) ** 2
        dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
        grad_norm_val = local_norm_sq.sqrt().item()
        clip_coef = args.max_grad_norm / max(grad_norm_val, 1e-6)
        if clip_coef < 1.0:
            for g in grads:
                g.detach().mul_(clip_coef)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        training_state.step = step
        training_state.tokens_seen += total_tokens * dp_size

        if step % args.log_interval == 0 and rank == 0:
            avg_loss = total_loss / args.gradient_accumulation
            elapsed = time.time() - t0
            tokens_per_sec = (total_tokens * dp_size) / elapsed
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "step=%d/%d | loss=%.4f | lr=%.2e | grad_norm=%.4f | "
                "tok/s=%.0f | tokens_seen=%s",
                step,
                args.total_steps,
                avg_loss,
                lr,
                grad_norm_val,
                tokens_per_sec,
                f"{training_state.tokens_seen:,}",
            )
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", avg_loss, step)
                tb_writer.add_scalar("train/lr", lr, step)
                tb_writer.add_scalar("train/grad_norm", grad_norm_val, step)
                tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)
                tb_writer.add_scalar("train/tokens_seen", training_state.tokens_seen, step)

        if step % args.save_interval == 0:
            ckpt_dir = f"{args.checkpoint_dir}/step_{step}"
            save_checkpoint(training_state, ckpt_dir)

    ckpt_dir = f"{args.checkpoint_dir}/step_{step}"
    save_checkpoint(training_state, ckpt_dir)
    if rank == 0:
        logger.info(
            "Training complete at step %d | tokens_seen=%s",
            step,
            f"{training_state.tokens_seen:,}",
        )

    if args.export_hf:
        from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions

        full_sd, _ = get_state_dict(
            model, [],
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        if rank == 0:
            export_model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float32,
            )
            export_model.load_state_dict(full_sd)
            export_model.save_pretrained(args.export_hf)
            logger.info("Exported HF checkpoint to %s", args.export_hf)

        dist.barrier()

    if tb_writer is not None:
        tb_writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

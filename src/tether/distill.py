import argparse
import logging
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tether.training.distributed import (
    add_common_args,
    apply_yaml_overrides,
    build_dataloader,
    build_optimizer,
    build_scheduler,
    export_hf,
    init_distributed,
    init_mesh,
    load_model,
    log_scalars,
    setup_tensorboard,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DISTILL_YAML_OVERRIDES = {
    "teacher": "teacher",
    "teacher_dtype": "teacher_dtype",
    "distill_loss": "distill_loss",
    "temperature": "temperature",
    "alpha": "alpha",
}


def reverse_kl_loss(student_logits, teacher_logits, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
    return loss * (temperature**2)


DISTILL_LOSS_FNS = {
    "reverse_kl": reverse_kl_loss,
}


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.set_defaults(checkpoint_dir="checkpoints/distill_smollm2_135m")
    parser.add_argument("--teacher", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--teacher-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--distill-loss", type=str, default="reverse_kl", choices=list(DISTILL_LOSS_FNS.keys()))
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="L = alpha*CE + (1-alpha)*distill_loss")
    args = parser.parse_args()
    apply_yaml_overrides(parser, args, extra_mappings=DISTILL_YAML_OVERRIDES)

    rank, world_size, device = init_distributed()
    dp_mesh, tp_mesh, dp_size = init_mesh(world_size, args.tp_size)
    dp_rank = dp_mesh.get_local_rank() if tp_mesh else rank

    if rank == 0:
        logger.info("World size: %d | TP: %d | DP: %d", world_size, args.tp_size, dp_size)

    student = load_model(args.model, device, tp_mesh, dp_mesh, args.param_dtype, args.reduce_dtype)
    if rank == 0:
        logger.info("Student: %s (%s params)", args.model, f"{sum(p.numel() for p in student.parameters()):,}")

    teacher_dtype = getattr(torch, args.teacher_dtype)
    from transformers import AutoModelForCausalLM
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from tether.training.parallelism_hf import apply_dp_hf, apply_tp_hf, validate_tp_size_hf

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=teacher_dtype, attn_implementation="flash_attention_2",
    )
    if rank == 0:
        logger.info("Teacher: %s (%s params)", args.teacher, f"{sum(p.numel() for p in teacher.parameters()):,}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.to(device)

    validate_tp_size_hf(teacher.config, args.tp_size)
    if tp_mesh is not None:
        apply_tp_hf(teacher, tp_mesh)
    apply_dp_hf(teacher, dp_mesh, MixedPrecisionPolicy(param_dtype=teacher_dtype, reduce_dtype=teacher_dtype))

    optimizer = build_optimizer(student, args)
    scheduler = build_scheduler(optimizer, args)

    from tether.training.checkpoint import TrainingState, load_checkpoint, save_checkpoint

    state = TrainingState(student, optimizer, scheduler)
    if args.resume:
        load_checkpoint(state, args.resume)
        if rank == 0:
            logger.info("Resumed from step %d (tokens_seen=%d)", state.step, state.tokens_seen)

    dataloader = build_dataloader(args, dp_rank, dp_size, state.step)
    tb_writer = setup_tensorboard(args.log_dir, rank)

    use_loss_parallel = args.tp_size > 1
    distill_loss_fn = DISTILL_LOSS_FNS[args.distill_loss]
    alpha, temperature = args.alpha, args.temperature

    student.train()
    data_iter = iter(dataloader)
    step = state.step

    effective = args.batch_size * args.gradient_accumulation * dp_size
    if rank == 0:
        logger.info("Effective batch: %d (per_gpu=%d x accum=%d x dp=%d)", effective, args.batch_size, args.gradient_accumulation, dp_size)
        logger.info("Distillation: loss=%s | alpha=%.2f | temperature=%.1f", args.distill_loss, alpha, temperature)

    while step < args.total_steps:
        step += 1
        t0 = time.time()
        total_loss = total_ce = total_distill = total_tokens = 0

        for micro_step in range(args.gradient_accumulation):
            input_ids = next(data_iter)["input_ids"].to(device)
            student.set_requires_gradient_sync(micro_step == args.gradient_accumulation - 1)

            student_logits = student(input_ids).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits

            if use_loss_parallel:
                s_logits = student_logits.full_tensor()
                t_logits = teacher_logits.full_tensor()
            else:
                s_logits, t_logits = student_logits, teacher_logits

            shift_s = s_logits[..., :-1, :].contiguous()
            shift_t = t_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            d_loss = distill_loss_fn(shift_s.flatten(0, 1), shift_t.flatten(0, 1).detach(), temperature)

            if use_loss_parallel:
                from torch.distributed.tensor.parallel import loss_parallel
                with loss_parallel():
                    ce_loss = F.cross_entropy(shift_s.flatten(0, 1), shift_labels.flatten(0, 1), ignore_index=-100)
                    loss = alpha * ce_loss + (1.0 - alpha) * d_loss
                    (loss / args.gradient_accumulation).backward()
            else:
                ce_loss = F.cross_entropy(shift_s.flatten(0, 1), shift_labels.flatten(0, 1), ignore_index=-100)
                loss = alpha * ce_loss + (1.0 - alpha) * d_loss
                (loss / args.gradient_accumulation).backward()

            total_loss += loss.detach().item()
            total_ce += ce_loss.detach().item()
            total_distill += d_loss.detach().item()
            total_tokens += input_ids.numel()

        grad_norm = nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        grad_norm_val = grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else float(grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        state.step = step
        state.tokens_seen += total_tokens * dp_size

        if step % args.log_interval == 0 and rank == 0:
            ga = args.gradient_accumulation
            avg_loss, avg_ce, avg_distill = total_loss / ga, total_ce / ga, total_distill / ga
            elapsed = time.time() - t0
            tok_s = (total_tokens * dp_size) / elapsed
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "step=%d/%d | loss=%.4f (ce=%.4f, distill=%.4f) | lr=%.2e | grad_norm=%.4f | tok/s=%.0f | tokens_seen=%s",
                step, args.total_steps, avg_loss, avg_ce, avg_distill, lr, grad_norm_val, tok_s, f"{state.tokens_seen:,}",
            )
            log_scalars(tb_writer, {
                "loss": avg_loss, "ce_loss": avg_ce, "distill_loss": avg_distill,
                "lr": lr, "grad_norm": grad_norm_val, "tokens_per_sec": tok_s, "tokens_seen": state.tokens_seen,
            }, step)

        if step % args.save_interval == 0:
            save_checkpoint(state, f"{args.checkpoint_dir}/step_{step}")

    save_checkpoint(state, f"{args.checkpoint_dir}/step_{step}")
    if rank == 0:
        logger.info("Distillation complete at step %d | tokens_seen=%s", step, f"{state.tokens_seen:,}")

    if args.export_hf:
        export_hf(student, args.model, args.export_hf, rank)

    if tb_writer:
        tb_writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

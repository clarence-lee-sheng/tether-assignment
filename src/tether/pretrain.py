import argparse
import logging
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tether.training.distributed import (
    add_common_args,
    apply_yaml_overrides,
    build_dataloader,
    build_optimizer,
    build_scheduler,
    clip_grad_norm_2d,
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


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.set_defaults(checkpoint_dir="checkpoints/smollm2_135m_hf")
    args = parser.parse_args()
    apply_yaml_overrides(parser, args)

    rank, world_size, device = init_distributed()
    dp_mesh, tp_mesh, dp_size = init_mesh(world_size, args.tp_size)
    dp_rank = dp_mesh.get_local_rank() if tp_mesh else rank

    if rank == 0:
        logger.info("World size: %d | TP: %d | DP: %d", world_size, args.tp_size, dp_size)

    model = load_model(args.model, device, tp_mesh, dp_mesh, args.param_dtype, args.reduce_dtype)
    if rank == 0:
        logger.info("Model: %s (%s params)", args.model, f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    from tether.training.checkpoint import TrainingState, load_checkpoint, save_checkpoint

    state = TrainingState(model, optimizer, scheduler)
    if args.resume:
        load_checkpoint(state, args.resume)
        if rank == 0:
            logger.info("Resumed from step %d (tokens_seen=%d)", state.step, state.tokens_seen)

    dataloader = build_dataloader(args, dp_rank, dp_size, state.step)
    tb_writer = setup_tensorboard(args.log_dir, rank)

    use_loss_parallel = args.tp_size > 1
    model.train()
    data_iter = iter(dataloader)
    step = state.step

    if rank == 0:
        effective = args.batch_size * args.gradient_accumulation * dp_size
        logger.info("Effective batch: %d (per_gpu=%d x accum=%d x dp=%d)", effective, args.batch_size, args.gradient_accumulation, dp_size)

    while step < args.total_steps:
        step += 1
        t0 = time.time()
        total_loss = total_tokens = 0

        for micro_step in range(args.gradient_accumulation):
            input_ids = next(data_iter)["input_ids"].to(device)
            model.set_requires_gradient_sync(micro_step == args.gradient_accumulation - 1)

            if use_loss_parallel:
                from torch.distributed.tensor.parallel import loss_parallel
                output = model(input_ids)
                shift_logits = output.logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                with loss_parallel():
                    loss = F.cross_entropy(shift_logits.flatten(0, 1), shift_labels.flatten(0, 1), ignore_index=-100)
                    (loss / args.gradient_accumulation).backward()
            else:
                output = model(input_ids, labels=input_ids)
                loss = output.loss
                (loss / args.gradient_accumulation).backward()

            total_loss += loss.detach().item()
            total_tokens += input_ids.numel()

        grad_norm = clip_grad_norm_2d(model, args.max_grad_norm, device)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        state.step = step
        state.tokens_seen += total_tokens * dp_size

        if step % args.log_interval == 0 and rank == 0:
            avg_loss = total_loss / args.gradient_accumulation
            elapsed = time.time() - t0
            tok_s = (total_tokens * dp_size) / elapsed
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "step=%d/%d | loss=%.4f | lr=%.2e | grad_norm=%.4f | tok/s=%.0f | tokens_seen=%s",
                step, args.total_steps, avg_loss, lr, grad_norm, tok_s, f"{state.tokens_seen:,}",
            )
            log_scalars(tb_writer, {"loss": avg_loss, "lr": lr, "grad_norm": grad_norm, "tokens_per_sec": tok_s, "tokens_seen": state.tokens_seen}, step)

        if step % args.save_interval == 0:
            save_checkpoint(state, f"{args.checkpoint_dir}/step_{step}")

    save_checkpoint(state, f"{args.checkpoint_dir}/step_{step}")
    if rank == 0:
        logger.info("Training complete at step %d | tokens_seen=%s", step, f"{state.tokens_seen:,}")

    if args.export_hf:
        export_hf(model, args.model, args.export_hf, rank)

    if tb_writer:
        tb_writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

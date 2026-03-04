import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _find_checkpoint_dirs(base_dir: str, last: int | None = None) -> list[Path]:
    base = Path(base_dir)
    if not base.is_dir():
        logger.error("Checkpoint directory not found: %s", base)
        sys.exit(1)

    step_dirs = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"step_(\d+)$", d.name) or re.match(r"checkpoint-(\d+)$", d.name)
        if m:
            step_dirs.append((int(m.group(1)), d))

    step_dirs.sort(key=lambda x: x[0])
    dirs = [d for _, d in step_dirs]

    if last is not None:
        dirs = dirs[-last:]

    return dirs


def build_model_args(cfg: dict, model: str) -> str:
    parts = [f"model_name={model}"]

    for key in ("dtype", "max_model_length", "gpu_memory_utilization", "tensor_parallel_size"):
        if key in cfg:
            parts.append(f"{key}={cfg[key]}")

    if "generation_parameters" in cfg:
        gen = cfg["generation_parameters"]
        inner = ",".join(f"{k}:{v}" for k, v in gen.items())
        parts.append(f"generation_parameters={{{inner}}}")

    return ",".join(str(p) for p in parts)


def build_cmd(cfg: dict, model: str, output_dir: str) -> list[str]:
    tasks = cfg["tasks"]
    tasks_str = ",".join(tasks) if isinstance(tasks, list) else str(tasks)

    cmd = ["lighteval", "vllm", build_model_args(cfg, model), tasks_str]
    cmd += ["--output-dir", output_dir]

    if "custom_tasks" in cfg:
        cmd += ["--custom-tasks", str(cfg["custom_tasks"])]

    if cfg.get("save_details", False):
        cmd.append("--save-details")

    if "max_samples" in cfg:
        cmd += ["--max-samples", str(cfg["max_samples"])]

    return cmd


def run_eval(cmd: list[str], dry_run: bool) -> int:
    logger.info("Command: %s", " ".join(cmd))
    if dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return subprocess.run(cmd, env=env).returncode


def already_evaluated(output_dir: str) -> bool:
    out = Path(output_dir)
    return any(out.glob("results_*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lighteval vLLM evaluation from a YAML config")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument(
        "--model", type=str, default=None, help="Override model (single-model mode only)."
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit examples per task (for debugging)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate even if results already exist."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples

    for field in ("tasks", "output_dir"):
        if field not in cfg:
            logger.error("Config missing required field: %s", field)
            sys.exit(1)

    if args.model:
        checkpoints = [Path(args.model)]
    elif "checkpoints" in cfg:
        checkpoints = [Path(c) for c in cfg["checkpoints"]]
    elif "checkpoint_dir" in cfg:
        checkpoints = _find_checkpoint_dirs(cfg["checkpoint_dir"], last=cfg.get("last"))
        if not checkpoints:
            logger.error("No step_N or checkpoint-N directories found in %s", cfg["checkpoint_dir"])
            sys.exit(1)
    elif "model" in cfg:
        checkpoints = [Path(cfg["model"])]
    else:
        logger.error("Config must have one of: model, checkpoints, checkpoint_dir")
        sys.exit(1)

    logger.info("Evaluating %d checkpoint(s)", len(checkpoints))

    failed = []
    for i, ckpt in enumerate(checkpoints):
        if len(checkpoints) == 1 and "checkpoint_dir" not in cfg and "checkpoints" not in cfg:
            out_dir = cfg["output_dir"]
        else:
            out_dir = str(Path(cfg["output_dir"]) / ckpt.name)

        logger.info("[%d/%d] %s -> %s", i + 1, len(checkpoints), ckpt, out_dir)

        if not args.force and already_evaluated(out_dir):
            logger.info("  Skipping — results already exist (use --force to re-run)")
            continue

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(cfg, str(ckpt), out_dir)

        rc = run_eval(cmd, args.dry_run)
        if rc != 0:
            logger.error("  FAILED with exit code %d", rc)
            failed.append(str(ckpt))

    if len(checkpoints) > 1:
        done = len(checkpoints) - len(failed)
        logger.info("Done: %d/%d checkpoints evaluated", done, len(checkpoints))

    if failed:
        logger.error("Failed checkpoints:")
        for f in failed:
            logger.error("  %s", f)
        sys.exit(1)


if __name__ == "__main__":
    main()

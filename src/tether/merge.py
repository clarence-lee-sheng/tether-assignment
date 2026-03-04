import argparse
import logging
import re
import shutil
import subprocess
import sys
import tempfile
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
        logger.error("Directory not found: %s", base)
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


def _build_mergekit_config(
    model_dirs: list[Path],
    method: str,
    weights: list[float],
    dtype: str,
    base_model: Path | None,
    density: float | None,
) -> dict:
    config: dict = {"merge_method": method, "dtype": dtype}

    if method == "slerp":
        if len(model_dirs) != 2:
            logger.error("SLERP requires exactly 2 checkpoints, got %d", len(model_dirs))
            sys.exit(1)
        config["base_model"] = str(model_dirs[0])
        config["parameters"] = {"t": weights[1] / sum(weights)}
        config["models"] = [{"model": str(d)} for d in model_dirs]
        return config

    if method in ("ties", "dare_ties", "dare_linear"):
        config["base_model"] = str(base_model or model_dirs[0])

    total_w = sum(weights)
    models = []
    for d, w in zip(model_dirs, weights):
        entry: dict = {"model": str(d), "parameters": {"weight": round(w / total_w, 6)}}
        if density is not None and method in ("ties", "dare_ties", "dare_linear"):
            entry["parameters"]["density"] = density
        models.append(entry)

    config["models"] = models
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge HF checkpoints using mergekit")
    parser.add_argument("--config", type=str, default=None, help="YAML config file. CLI args override config values.")
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument("--checkpoints", nargs="+", type=str, help="Explicit list of HF checkpoint directories.")
    ckpt_group.add_argument("--checkpoint-dir", type=str, help="Parent dir containing step_N or checkpoint-N subdirs.")
    parser.add_argument("--last", type=int, default=None, help="Only use the last N checkpoints from --checkpoint-dir.")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Per-checkpoint weights (unnormalized).")
    parser.add_argument("--method", choices=["linear", "slerp", "ties", "dare_ties", "dare_linear"], default=None, help="Merge method.")
    parser.add_argument("--density", type=float, default=None, help="Density for TIES/DARE methods (0-1).")
    parser.add_argument("--base-model", type=str, default=None, help="Base model for TIES/DARE.")
    parser.add_argument("--output", type=str, default=None, help="Output directory for the merged model.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None, help="Output dtype.")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for yaml_key, arg_name in {
            "checkpoints": "checkpoints", "checkpoint_dir": "checkpoint_dir",
            "last": "last", "weights": "weights", "method": "method",
            "density": "density", "base_model": "base_model",
            "output": "output", "dtype": "dtype",
        }.items():
            if yaml_key in cfg and getattr(args, arg_name) is None:
                setattr(args, arg_name, cfg[yaml_key])

    if args.method is None:
        args.method = "linear"
    if args.dtype is None:
        args.dtype = "bfloat16"

    if args.output is None:
        parser.error("--output is required (or set 'output' in config)")
    if args.checkpoints is None and args.checkpoint_dir is None:
        parser.error("--checkpoints or --checkpoint-dir is required (or set in config)")

    if args.checkpoints:
        ckpt_dirs = [Path(c) for c in args.checkpoints]
        for d in ckpt_dirs:
            if not d.is_dir():
                logger.error("Checkpoint not found: %s", d)
                sys.exit(1)
    else:
        ckpt_dirs = _find_checkpoint_dirs(args.checkpoint_dir, last=args.last)
        if not ckpt_dirs:
            logger.error("No step_N or checkpoint-N directories found in %s", args.checkpoint_dir)
            sys.exit(1)

    n = len(ckpt_dirs)
    if n < 2:
        logger.error("Need at least 2 checkpoints to merge, got %d", n)
        sys.exit(1)

    merge_weights = args.weights if args.weights is not None else [1.0] * n
    if len(merge_weights) != n:
        logger.error("--weights has %d values but there are %d checkpoints", len(merge_weights), n)
        sys.exit(1)

    logger.info("Merging %d checkpoints with method=%s:", n, args.method)
    for d, w in zip(ckpt_dirs, merge_weights):
        logger.info("  %s  (weight=%.3f)", d, w)

    base_model = Path(args.base_model) if args.base_model else None

    config = _build_mergekit_config(
        ckpt_dirs, args.method, merge_weights, args.dtype,
        base_model=base_model, density=args.density,
    )

    tmp_dir = tempfile.mkdtemp(prefix="tether_merge_")
    config_path = Path(tmp_dir) / "merge_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Mergekit config:\n%s", yaml.dump(config, default_flow_style=False, sort_keys=False))

    output = Path(args.output)
    cmd = ["mergekit-yaml", str(config_path), str(output)]
    logger.info("Running: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp_dir)

    if result.returncode != 0:
        logger.error("mergekit failed:\n%s", result.stderr)
        sys.exit(1)
    if result.stdout:
        logger.info("mergekit output:\n%s", result.stdout)

    logger.info("Done: %d checkpoints merged with %s -> %s", n, args.method, output)


if __name__ == "__main__":
    main()

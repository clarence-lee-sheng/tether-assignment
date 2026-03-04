import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if "models" not in raw or not raw["models"]:
        raise ValueError("Config must contain a non-empty 'models' list")
    output_dir = raw.get("output_dir", "models")
    for entry in raw["models"]:
        if "model" not in entry:
            raise ValueError(f"Model entry missing 'model' field: {entry}")
        if "output" not in entry:
            entry["output"] = str(Path(output_dir) / entry["model"].split("/")[-1])
    return raw


def download_one(model: str, output: str) -> None:
    from huggingface_hub import snapshot_download
    Path(output).mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", model, output)
    snapshot_download(model, local_dir=output)
    logger.info("Done: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models from HuggingFace Hub")
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only download models whose name matches (last part of repo ID).")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    entries = cfg["models"]

    if args.only:
        only_set = set(args.only)
        entries = [e for e in entries if e["model"].split("/")[-1] in only_set]
        if not entries:
            logger.error("No models matched --only %s", args.only)
            return

    logger.info("Download plan: %d model(s)", len(entries))
    for e in entries:
        logger.info("  %-40s → %s", e["model"], e["output"])

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    for e in entries:
        download_one(e["model"], e["output"])

    logger.info("All models downloaded.")


if __name__ == "__main__":
    main()

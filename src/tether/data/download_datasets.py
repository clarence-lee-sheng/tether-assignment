import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_name(dataset: str, subset: str | None) -> str:
    slug = dataset.split("/")[-1]
    if subset:
        slug += f"-{subset}"
    return slug.lower().replace(" ", "-")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")
    if "datasets" not in raw or not raw["datasets"]:
        raise ValueError("Config must contain a non-empty 'datasets' list")

    for i, ds in enumerate(raw["datasets"]):
        if "dataset" not in ds:
            raise ValueError(f"Dataset entry {i} missing required 'dataset' field")
        if "name" not in ds or ds["name"] is None:
            ds["name"] = _make_name(ds["dataset"], ds.get("subset"))
        ds.setdefault("split", "train")
        ds.setdefault("text_column", "text")
        ds.setdefault("streaming", raw.get("streaming", True))
        if "max_samples" not in ds:
            ds["max_samples"] = None

    return raw


def download_one(
    dataset: str,
    subset: str | None,
    name: str,
    split: str,
    output_dir: str,
    streaming: bool,
    max_samples: int | None,
    num_proc: int | None,
) -> str:
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _logger = _logging.getLogger(__name__)

    _logger.info("Downloading %s (subset=%s, name=%s)", dataset, subset, name)

    from tether.data.downloader import DatasetDownloader
    from tether.utils.config import DownloadConfig

    config = DownloadConfig(
        dataset_name=dataset,
        dataset_config=subset,
        split=split,
        output_dir=output_dir,
        streaming=streaming,
        max_samples=max_samples,
        num_proc=num_proc,
    )
    downloader = DatasetDownloader(config)
    return str(downloader.download())


def tokenize_one(
    raw_dir: str,
    output_prefix: str,
    tokenizer_name: str,
    text_column: str,
    batch_size: int = 1024,
) -> str:
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _logger = _logging.getLogger(__name__)

    meta_file = Path(f"{output_prefix}.meta.json")
    if meta_file.exists():
        _logger.info("Already tokenized: %s — skipping", output_prefix)
        return output_prefix

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    _logger.info("Tokenizing %s → %s", raw_dir, output_prefix)

    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from tether.data.memmap_writer import MMapIndexedDatasetWriter, dtype_for_vocab_size

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    vocab_size = tokenizer.vocab_size
    dtype = dtype_for_vocab_size(vocab_size)
    eos_id = tokenizer.eos_token_id

    raw_path = Path(raw_dir)
    parquet_files = sorted(raw_path.glob("*.parquet"))
    if not parquet_files:
        _logger.warning("No .parquet files in %s — skipping", raw_dir)
        return output_prefix

    writer = MMapIndexedDatasetWriter(output_prefix, dtype=dtype)
    total_docs = 0
    empty_docs = 0

    for fpath in parquet_files:
        table = pq.read_table(fpath, columns=[text_column])
        col = table.column(text_column)

        for start in range(0, len(col), batch_size):
            texts = [str(col[i].as_py()) for i in range(start, min(start + batch_size, len(col)))]

            encoded = tokenizer(
                texts,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            for ids in encoded["input_ids"]:
                tokens = list(ids)
                if eos_id is not None:
                    tokens.append(eos_id)
                if not tokens:
                    empty_docs += 1
                    continue
                writer.add_item(np.array(tokens, dtype=np.int64))
                writer.end_document()
                total_docs += 1

    if empty_docs > 0:
        _logger.warning("Skipped %d empty documents", empty_docs)

    writer.finalize(
        metadata={
            "tokenizer": tokenizer_name,
            "vocab_size": vocab_size,
            "source_dataset": raw_dir,
            "sequence_length": None,
            "add_bos": True,
            "add_eos": True,
        }
    )

    _logger.info("Tokenization complete: %d documents, %d tokens", total_docs, writer.total_tokens)
    return output_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven dataset download and tokenization")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file defining datasets to download.",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        help="Only download/tokenize these dataset names (from the config).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max_samples for ALL datasets (for testing).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Override tokenizer from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir from config.",
    )
    parser.add_argument(
        "--tokenized-dir",
        type=str,
        default=None,
        help="Override tokenized_dir from config.",
    )
    parser.add_argument(
        "--tokenize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tokenize after downloading (overrides config).",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override streaming mode for all datasets.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Parallel download workers (non-streaming only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show download plan without executing.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = args.output_dir or cfg.get("output_dir", "data/raw")
    tokenized_dir = args.tokenized_dir or cfg.get("tokenized_dir", "data/tokenized")
    tokenizer = args.tokenizer or cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M")
    do_tokenize = args.tokenize if args.tokenize is not None else cfg.get("tokenize", True)

    entries = cfg["datasets"]
    if args.only:
        names_set = set(args.only)
        entries = [e for e in entries if e["name"] in names_set]
        found = {e["name"] for e in entries}
        missing = names_set - found
        if missing:
            logger.error("Unknown dataset names: %s", ", ".join(sorted(missing)))
            logger.error("Available: %s", ", ".join(e["name"] for e in cfg["datasets"]))
            sys.exit(1)

    if not entries:
        logger.info("No datasets to download.")
        return

    for entry in entries:
        if args.max_samples is not None:
            entry["max_samples"] = args.max_samples
        if args.streaming is not None:
            entry["streaming"] = args.streaming

    logger.info("Download plan: %d dataset(s)", len(entries))
    for entry in entries:
        samples_str = f"max={entry['max_samples']}" if entry["max_samples"] else "all"
        stream_str = "streaming" if entry["streaming"] else "full"
        logger.info(
            "  %-30s  %s/%s  (%s, %s)",
            entry["name"],
            entry["dataset"],
            entry.get("subset", "—"),
            samples_str,
            stream_str,
        )

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    import ray

    runtime_env = {"env_vars": {"PYTHONPATH": ":".join(sys.path)}}
    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    download_remote = ray.remote(download_one)

    download_futures = {}
    for entry in entries:
        ref = download_remote.remote(
            dataset=entry["dataset"],
            subset=entry.get("subset"),
            name=entry["name"],
            split=entry["split"],
            output_dir=output_dir,
            streaming=entry["streaming"],
            max_samples=entry["max_samples"],
            num_proc=args.num_proc,
        )
        download_futures[entry["name"]] = (ref, entry)

    logger.info("Launched %d parallel downloads", len(download_futures))

    raw_dirs: dict[str, str] = {}
    for name, (ref, _) in download_futures.items():
        raw_dirs[name] = ray.get(ref)
        logger.info("Download complete: %s → %s", name, raw_dirs[name])

    if do_tokenize:
        tokenize_remote = ray.remote(tokenize_one)

        tok_futures = {}
        for name, (_, entry) in download_futures.items():
            prefix = str(Path(tokenized_dir) / name / name)
            ref = tokenize_remote.remote(
                raw_dir=raw_dirs[name],
                output_prefix=prefix,
                tokenizer_name=tokenizer,
                text_column=entry["text_column"],
            )
            tok_futures[name] = ref

        logger.info("Launched %d parallel tokenizations", len(tok_futures))

        for name, ref in tok_futures.items():
            ray.get(ref)
            logger.info("Tokenization complete: %s", name)

    ray.shutdown()
    logger.info("Done!")


if __name__ == "__main__":
    main()

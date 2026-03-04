import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if "datasets" not in raw or not raw["datasets"]:
        raise ValueError("Config must contain a non-empty 'datasets' list")
    for i, ds in enumerate(raw["datasets"]):
        if "name" not in ds:
            raise ValueError(f"Dataset entry {i} missing required 'name' field")
        ds.setdefault("text_column", raw.get("text_column", "text"))
        ds.setdefault("batch_size", raw.get("batch_size", 1024))
        ds.setdefault("add_bos", raw.get("add_bos", False))
        ds.setdefault("add_eos", raw.get("add_eos", True))
        ds.setdefault("tokenizer", raw.get("tokenizer"))
        ds.setdefault("split", "train")
        ds.setdefault("preprocess", None)
    return raw


def _raw_dir_slug(dataset: str, subset: str | None, split: str = "train") -> str:
    slug = dataset.replace("/", "__")
    if subset:
        slug += f"__{subset}"
    return f"{slug}__{split}"


def _flatten_messages(row: object) -> str:
    if row is None:
        return ""
    msgs = row if isinstance(row, list) else list(row)
    parts = []
    for msg in msgs:
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if content:
            parts.append(str(content).strip())
    return "\n\n".join(parts)


def tokenize_one(
    raw_dir: str,
    output_prefix: str,
    tokenizer_name: str,
    text_column: str,
    batch_size: int = 1024,
    add_bos: bool = False,
    add_eos: bool = True,
    preprocess: str | None = None,
) -> str:
    import logging as _logging

    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from tether.data.memmap_writer import MMapIndexedDatasetWriter, dtype_for_vocab_size

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _logger = _logging.getLogger(__name__)

    meta_file = Path(f"{output_prefix}.meta.json")
    if meta_file.exists():
        _logger.info("Already tokenized: %s — skipping", output_prefix)
        return output_prefix

    parquet_files = sorted(Path(raw_dir).glob("*.parquet"))
    if not parquet_files:
        _logger.warning("No .parquet files in %s — skipping", raw_dir)
        return output_prefix

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    _logger.info("Tokenizing %s → %s", raw_dir, output_prefix)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    dtype = dtype_for_vocab_size(tokenizer.vocab_size)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    writer = MMapIndexedDatasetWriter(output_prefix, dtype=dtype)
    total_docs = 0
    empty_docs = 0

    for fpath in parquet_files:
        col = pq.read_table(fpath, columns=[text_column]).column(text_column)
        for start in range(0, len(col), batch_size):
            rows = [col[i].as_py() for i in range(start, min(start + batch_size, len(col)))]
            if preprocess == "flatten_messages":
                texts = [_flatten_messages(r) for r in rows]
            else:
                texts = [str(r) for r in rows]

            encoded = tokenizer(texts, truncation=False, add_special_tokens=False,
                                return_attention_mask=False, return_token_type_ids=False)

            for ids in encoded["input_ids"]:
                tokens = []
                if add_bos and bos_id is not None:
                    tokens.append(bos_id)
                tokens.extend(ids)
                if add_eos and eos_id is not None:
                    tokens.append(eos_id)
                if not tokens:
                    empty_docs += 1
                    continue
                writer.add_item(np.array(tokens, dtype=np.int64))
                writer.end_document()
                total_docs += 1

    if empty_docs:
        _logger.warning("Skipped %d empty documents", empty_docs)

    writer.finalize(metadata={
        "tokenizer": tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "source_dataset": raw_dir,
        "add_bos": add_bos,
        "add_eos": add_eos,
    })
    _logger.info("Done: %d documents, %d tokens", total_docs, writer.total_tokens)
    return output_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven dataset tokenization")
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--only", nargs="+", default=None, help="Only tokenize these dataset names.")
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer from config.")
    parser.add_argument("--input-dir", default=None, help="Override raw data base directory.")
    parser.add_argument("--output-dir", default=None, help="Override tokenized output base directory.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = Path(args.input_dir or cfg.get("input_dir", "data/raw"))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "data/tokenized"))
    tokenizer_override = args.tokenizer or cfg.get("tokenizer")

    entries = cfg["datasets"]
    if args.only:
        names_set = set(args.only)
        entries = [e for e in entries if e["name"] in names_set]
        missing = names_set - {e["name"] for e in entries}
        if missing:
            logger.error("Unknown names: %s. Available: %s",
                         ", ".join(sorted(missing)),
                         ", ".join(e["name"] for e in cfg["datasets"]))
            sys.exit(1)

    tasks = []
    for entry in entries:
        name = entry["name"]
        tok = tokenizer_override or entry.get("tokenizer")
        if not tok:
            logger.error("No tokenizer for %s", name)
            sys.exit(1)

        raw_path = Path(entry["raw_dir"]) if "raw_dir" in entry else (
            input_dir / _raw_dir_slug(entry.get("dataset", ""), entry.get("subset"), entry.get("split", "train"))
        )

        if not (raw_path / ".download_complete").exists():
            logger.warning("Not downloaded (no sentinel): %s — skipping", raw_path)
            continue

        output_prefix = str(output_dir / name / name)
        if Path(f"{output_prefix}.meta.json").exists():
            logger.info("Already tokenized: %s — skipping", name)
            continue

        tasks.append({
            "name": name,
            "raw_dir": str(raw_path),
            "output_prefix": output_prefix,
            "tokenizer": tok,
            "text_column": entry["text_column"],
            "batch_size": entry["batch_size"],
            "add_bos": entry["add_bos"],
            "add_eos": entry["add_eos"],
            "preprocess": entry.get("preprocess"),
        })

    if not tasks:
        logger.info("Nothing to tokenize.")
        return

    logger.info("Tokenization plan: %d dataset(s)", len(tasks))
    for t in tasks:
        logger.info("  %-30s  %s → %s", t["name"], t["raw_dir"], t["output_prefix"])

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    import ray

    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": ":".join(sys.path)}})
    tokenize_remote = ray.remote(tokenize_one)

    futures = {
        t["name"]: tokenize_remote.remote(
            raw_dir=t["raw_dir"],
            output_prefix=t["output_prefix"],
            tokenizer_name=t["tokenizer"],
            text_column=t["text_column"],
            batch_size=t["batch_size"],
            add_bos=t["add_bos"],
            add_eos=t["add_eos"],
            preprocess=t["preprocess"],
        )
        for t in tasks
    }

    for name, ref in futures.items():
        ray.get(ref)
        logger.info("Complete: %s", name)

    ray.shutdown()
    logger.info("All done!")


if __name__ == "__main__":
    main()

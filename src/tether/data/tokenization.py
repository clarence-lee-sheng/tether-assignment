import logging
from pathlib import Path
from typing import Optional

import numpy as np

from tether.data.memmap_writer import MMapIndexedDatasetWriter, dtype_for_vocab_size
from tether.utils.config import TokenizationConfig

logger = logging.getLogger(__name__)


class TokenizerActor:
    def __init__(
        self,
        tokenizer_name_or_path: str,
        text_column: str,
        max_seq_length: Optional[int],
        add_bos: bool,
        add_eos: bool,
    ):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, use_fast=True
        )
        self.text_column = text_column
        self.max_seq_length = max_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
        texts = [str(t) for t in batch[self.text_column].tolist()]

        encoded = self.tokenizer(
            texts,
            truncation=self.max_seq_length is not None,
            max_length=self.max_seq_length,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        results = []
        num_tokens = []
        for ids in encoded["input_ids"]:
            tokens = []
            if self.add_bos and bos_id is not None:
                tokens.append(bos_id)
            tokens.extend(ids)
            if self.add_eos and eos_id is not None:
                tokens.append(eos_id)
            arr = np.array(tokens, dtype=np.int64)
            results.append(arr)
            num_tokens.append(len(arr))

        return {
            "token_ids": results,
            "num_tokens": np.array(num_tokens, dtype=np.int64),
        }


def run_tokenization(config: TokenizationConfig) -> Path:
    import ray
    import ray.data

    input_dir = Path(config.input_dir)
    parquet_files = sorted(input_dir.glob("*.parquet"))
    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    if parquet_files:
        input_files = [str(f) for f in parquet_files]
        file_format = "parquet"
    elif jsonl_files:
        input_files = [str(f) for f in jsonl_files]
        file_format = "jsonl"
    else:
        raise FileNotFoundError(f"No .parquet or .jsonl files found in {input_dir}")

    logger.info("Found %d %s files in %s", len(input_files), file_format, input_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, use_fast=True)
    vocab_size = tokenizer.vocab_size
    dtype = dtype_for_vocab_size(vocab_size)
    logger.info(
        "Tokenizer %s: vocab_size=%d, output dtype=%s",
        config.tokenizer_name_or_path,
        vocab_size,
        dtype,
    )
    del tokenizer

    import sys

    runtime_env = {"env_vars": {"PYTHONPATH": ":".join(sys.path)}}
    ray_ctx = ray.init(
        address=config.ray_address,
        ignore_reinit_error=True,
        runtime_env=runtime_env,
    )
    logger.info("Ray initialized: %s", ray_ctx.dashboard_url or "local mode")

    try:
        if file_format == "parquet":
            ds = ray.data.read_parquet(input_files)
        else:
            ds = ray.data.read_json(input_files)

        tokenized_ds = ds.map_batches(
            TokenizerActor,
            fn_constructor_kwargs={
                "tokenizer_name_or_path": config.tokenizer_name_or_path,
                "text_column": config.text_column,
                "max_seq_length": config.max_seq_length,
                "add_bos": config.add_bos,
                "add_eos": config.add_eos,
            },
            batch_size=config.batch_size,
            compute=ray.data.ActorPoolStrategy(size=config.num_workers),
        )

        output_prefix = config.output_prefix
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        writer = MMapIndexedDatasetWriter(output_prefix, dtype=dtype)
        total_docs = 0
        empty_docs = 0

        for batch in tokenized_ds.iter_batches(batch_size=config.batch_size):
            for token_ids in batch["token_ids"]:
                token_ids = np.asarray(token_ids)
                if len(token_ids) == 0:
                    empty_docs += 1
                    continue
                writer.add_item(token_ids)
                writer.end_document()
                total_docs += 1

        if empty_docs > 0:
            logger.warning("Skipped %d empty documents", empty_docs)

        result_path = writer.finalize(
            metadata={
                "tokenizer": config.tokenizer_name_or_path,
                "vocab_size": vocab_size,
                "source_dataset": str(config.input_dir),
                "sequence_length": config.max_seq_length,
                "add_bos": config.add_bos,
                "add_eos": config.add_eos,
            }
        )

        logger.info(
            "Tokenization complete: %d documents, %d tokens -> %s",
            total_docs,
            writer.total_tokens,
            result_path,
        )
        return result_path

    finally:
        ray.shutdown()

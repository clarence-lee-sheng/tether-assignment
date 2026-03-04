import json
import logging
from pathlib import Path

from tqdm import tqdm

from tether.utils.config import DownloadConfig

logger = logging.getLogger(__name__)

_SENTINEL = ".download_complete"
_SHARD_SIZE_BYTES = 256 * 1024 * 1024  # 256 MB target per shard in streaming mode


class DatasetDownloader:
    def __init__(self, config: DownloadConfig):
        self.config = config

    def download(self) -> Path:
        output_dir = Path(self.config.output_dir) / self._dataset_slug()
        sentinel = output_dir / _SENTINEL

        if sentinel.exists():
            logger.info("Dataset already downloaded at %s (sentinel found), skipping", output_dir)
            return output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.streaming:
            self._download_streaming(output_dir)
        else:
            self._download_full(output_dir)

        sentinel.write_text(
            json.dumps(
                {
                    "dataset": self.config.dataset_name,
                    "config": self.config.dataset_config,
                    "split": self.config.split,
                }
            )
        )
        logger.info("Download complete: %s", output_dir)
        return output_dir

    def _download_full(self, output_dir: Path) -> None:
        import datasets

        existing_shards = sorted(output_dir.glob("shard_*.parquet"))
        if existing_shards:
            logger.info(
                "Found %d existing shards in %s, skipping full download. "
                "Delete the directory to re-download.",
                len(existing_shards),
                output_dir,
            )
            return

        logger.info(
            "Downloading %s (config=%s, split=%s)",
            self.config.dataset_name,
            self.config.dataset_config,
            self.config.split,
        )
        try:
            ds = datasets.load_dataset(
                self.config.dataset_name,
                name=self.config.dataset_config,
                split=self.config.split,
                num_proc=self.config.num_proc,
                token=self.config.hf_token,
            )
        except ValueError as e:
            if "Config name is missing" in str(e) or "Please pick one among" in str(e):
                raise ValueError(
                    f"Dataset '{self.config.dataset_name}' requires a config name. "
                    f"Use --config to specify one. Original error: {e}"
                ) from e
            raise

        if self.config.max_samples is not None:
            ds = ds.select(range(min(self.config.max_samples, len(ds))))

        num_shards = max(1, ds.data.nbytes // _SHARD_SIZE_BYTES)
        for i in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=i)
            path = output_dir / f"shard_{i:05d}.parquet"
            shard.to_parquet(str(path))
        logger.info("Saved %d parquet shards to %s", num_shards, output_dir)

    def _download_streaming(self, output_dir: Path) -> None:
        import datasets

        logger.info(
            "Streaming %s (config=%s, split=%s)",
            self.config.dataset_name,
            self.config.dataset_config,
            self.config.split,
        )
        try:
            ds = datasets.load_dataset(
                self.config.dataset_name,
                name=self.config.dataset_config,
                split=self.config.split,
                streaming=True,
                token=self.config.hf_token,
            )
        except ValueError as e:
            if "Config name is missing" in str(e) or "Please pick one among" in str(e):
                raise ValueError(
                    f"Dataset '{self.config.dataset_name}' requires a config name. "
                    f"Use --config to specify one. Original error: {e}"
                ) from e
            raise

        existing_shards = sorted(output_dir.glob("shard_*.parquet"))
        skip_examples = 0
        shard_idx = 0

        if existing_shards:
            import pyarrow.parquet as pq

            for shard_path in existing_shards:
                skip_examples += pq.read_metadata(shard_path).num_rows
            shard_idx = len(existing_shards)
            logger.info(
                "Resuming: found %d existing shards (%d examples), "
                "continuing from shard %d",
                len(existing_shards),
                skip_examples,
                shard_idx,
            )

        buffer = []
        buffer_size = 0
        total_written = 0

        for i, example in enumerate(tqdm(ds, desc="Streaming", initial=skip_examples)):
            if self.config.max_samples is not None and i >= self.config.max_samples:
                break

            if i < skip_examples:
                continue

            buffer.append(example)
            buffer_size += sum(
                len(str(v).encode("utf-8")) for v in example.values() if isinstance(v, str)
            )

            if buffer_size >= _SHARD_SIZE_BYTES:
                self._flush_shard(buffer, output_dir, shard_idx)
                total_written += len(buffer)
                shard_idx += 1
                buffer = []
                buffer_size = 0

        if buffer:
            self._flush_shard(buffer, output_dir, shard_idx)
            total_written += len(buffer)
            shard_idx += 1

        total_examples = skip_examples + total_written
        logger.info(
            "Streamed %d new examples (%d total) into %d shards",
            total_written,
            total_examples,
            shard_idx,
        )

    def _flush_shard(self, records: list[dict], output_dir: Path, shard_idx: int) -> None:
        import datasets

        ds = datasets.Dataset.from_list(records)
        path = output_dir / f"shard_{shard_idx:05d}.parquet"
        ds.to_parquet(str(path))
        logger.debug("Wrote shard %s (%d records)", path, len(records))

    def _dataset_slug(self) -> str:
        slug = self.config.dataset_name.replace("/", "__")
        if self.config.dataset_config:
            slug += f"__{self.config.dataset_config}"
        slug += f"__{self.config.split}"
        return slug

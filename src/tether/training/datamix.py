from __future__ import annotations

from pathlib import Path

from tether.utils.config import DataMixConfig


def validate_mix_paths(config: DataMixConfig) -> None:
    for ds in config.datasets:
        bin_path = Path(f"{ds.path}.bin")
        idx_path = Path(f"{ds.path}.idx")
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing binary file: {bin_path}")
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing index file: {idx_path}")


def load_datamix(
    config_path: str | Path,
    mode: str = "iterable",
    rank: int = 0,
    world_size: int = 1,
):
    config = DataMixConfig.from_yaml(config_path)
    validate_mix_paths(config)

    if mode == "iterable":
        from tether.training.datasets import PackedIterableDataset

        return PackedIterableDataset(config, rank=rank, world_size=world_size)
    elif mode == "map":
        from tether.training.datasets import PackedMapDataset

        return PackedMapDataset(config)
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected 'iterable' or 'map'")

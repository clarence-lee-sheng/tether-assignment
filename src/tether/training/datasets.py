from __future__ import annotations

import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset

from tether.data.memmap_writer import MMapIndexedDatasetReader
from tether.training.packing import SequencePacker
from tether.utils.config import DataMixConfig


class PackedMapDataset(Dataset):
    def __init__(self, mix_config: DataMixConfig) -> None:
        super().__init__()
        self._seq_len = mix_config.seq_len

        self._packers: list[SequencePacker] = []
        self._block_offsets: list[int] = []
        total_blocks = 0

        for ds_cfg in mix_config.datasets:
            reader = MMapIndexedDatasetReader(ds_cfg.path)
            packer = SequencePacker(reader, seq_len=mix_config.seq_len + 1)
            self._packers.append(packer)
            self._block_offsets.append(total_blocks)
            total_blocks += packer.num_blocks

        self._total_blocks = total_blocks

    def __len__(self) -> int:
        return self._total_blocks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._total_blocks:
            raise IndexError(f"Index {idx} out of range [0, {self._total_blocks})")

        packer_idx = len(self._block_offsets) - 1
        for i in range(len(self._block_offsets) - 1):
            if idx < self._block_offsets[i + 1]:
                packer_idx = i
                break

        local_idx = idx - self._block_offsets[packer_idx]
        block = self._packers[packer_idx].get_block(local_idx)

        return {
            "input_ids": torch.from_numpy(block[:-1].astype(np.int64)),
            "labels": torch.from_numpy(block[1:].astype(np.int64)),
        }


class PackedIterableDataset(IterableDataset):
    def __init__(
        self,
        mix_config: DataMixConfig,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self._config = mix_config
        self._rank = rank
        self._world_size = world_size
        self.samples_to_skip: int = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        global_worker_id = self._rank * num_workers + worker_id
        total_workers = self._world_size * num_workers

        streams = []
        for ds_cfg in self._config.datasets:
            reader = MMapIndexedDatasetReader(ds_cfg.path)
            packer = SequencePacker(reader, seq_len=self._config.seq_len + 1)
            stream = packer.iter_blocks_strided(
                stride=total_workers,
                offset=global_worker_id,
                seed=self._config.seed,
            )
            streams.append(stream)

        weights = np.array(self._config.normalized_weights, dtype=np.float64)
        rng = np.random.RandomState(self._config.seed + global_worker_id)

        skipped = 0
        while skipped < self.samples_to_skip:
            ds_idx = rng.choice(len(streams), p=weights)
            next(streams[ds_idx])  # consume and discard
            skipped += 1

        while True:
            ds_idx = rng.choice(len(streams), p=weights)
            block = next(streams[ds_idx])

            yield {
                "input_ids": torch.from_numpy(block[:-1].astype(np.int64)),
                "labels": torch.from_numpy(block[1:].astype(np.int64)),
            }

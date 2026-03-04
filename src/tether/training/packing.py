from __future__ import annotations

from typing import Iterator

import numpy as np

from tether.data.memmap_writer import MMapIndexedDatasetReader


class SequencePacker:
    def __init__(self, reader: MMapIndexedDatasetReader, seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        self._reader = reader
        self._seq_len = seq_len
        self._sizes = reader.sizes
        self._cumulative_sizes = np.cumsum(self._sizes, dtype=np.int64)

    @property
    def total_tokens(self) -> int:
        if len(self._cumulative_sizes) == 0:
            return 0
        return int(self._cumulative_sizes[-1])

    @property
    def num_blocks(self) -> int:
        return self.total_tokens // self._seq_len

    def get_block(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.num_blocks:
            raise IndexError(f"Block index {idx} out of range [0, {self.num_blocks})")

        start_token = idx * self._seq_len
        end_token = start_token + self._seq_len

        return self._gather_tokens(start_token, end_token)

    def _gather_tokens(self, start_token: int, end_token: int) -> np.ndarray:
        num_tokens = end_token - start_token
        result = np.empty(num_tokens, dtype=self._reader.dtype)
        filled = 0

        seq_idx = int(np.searchsorted(self._cumulative_sizes, start_token, side="right"))
        if seq_idx > 0:
            offset_in_seq = start_token - int(self._cumulative_sizes[seq_idx - 1])
        else:
            offset_in_seq = start_token

        while filled < num_tokens:
            seq_data = self._reader[seq_idx]
            available = seq_data[offset_in_seq:]
            take = min(len(available), num_tokens - filled)
            result[filled : filled + take] = available[:take]
            filled += take
            seq_idx += 1
            offset_in_seq = 0

        return result

    def iter_blocks(self, start_seq_idx: int = 0) -> Iterator[np.ndarray]:
        num_seqs = len(self._reader)
        if num_seqs == 0 or self.total_tokens == 0:
            return

        buf = np.empty(0, dtype=self._reader.dtype)
        seq_idx = start_seq_idx % num_seqs if num_seqs > 0 else 0

        while True:
            while len(buf) < self._seq_len:
                seq_data = self._reader[seq_idx]
                buf = np.concatenate([buf, seq_data])
                seq_idx = (seq_idx + 1) % num_seqs

            yield buf[: self._seq_len]
            buf = buf[self._seq_len :]

    def iter_blocks_strided(
        self,
        stride: int,
        offset: int,
        seed: int = 0,
        start_epoch: int = 0,
        start_block: int = 0,
    ) -> Iterator[np.ndarray]:
        num_seqs = len(self._reader)
        if num_seqs == 0 or self.total_tokens == 0:
            return

        effective_stride = min(stride, num_seqs)
        effective_offset = offset % effective_stride

        buf = np.empty(0, dtype=self._reader.dtype)
        epoch = start_epoch
        blocks_to_skip = start_block

        while True:
            rng = np.random.RandomState(seed + epoch)
            perm = rng.permutation(num_seqs)
            my_indices = perm[effective_offset::effective_stride]

            for seq_idx in my_indices:
                seq_data = self._reader[int(seq_idx)]
                buf = np.concatenate([buf, seq_data])

                while len(buf) >= self._seq_len:
                    if blocks_to_skip > 0:
                        blocks_to_skip -= 1
                        buf = buf[self._seq_len :]
                        continue
                    yield buf[: self._seq_len]
                    buf = buf[self._seq_len :]

            epoch += 1

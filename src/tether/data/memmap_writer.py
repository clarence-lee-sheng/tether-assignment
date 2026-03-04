import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HDR_MAGIC = b"MMIDIDX\x00\x00"
_HDR_VERSION = 1

DTYPE_MAP = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}
DTYPE_TO_CODE = {np.dtype(v): k for k, v in DTYPE_MAP.items()}


def dtype_for_vocab_size(vocab_size: int) -> np.dtype:
    if vocab_size < 2**16:
        return np.dtype(np.uint16)
    return np.dtype(np.int32)


class MMapIndexedDatasetWriter:
    def __init__(self, output_prefix: str, dtype: np.dtype = np.int32):
        self._output_prefix = output_prefix
        self._bin_path = Path(f"{output_prefix}.bin")
        self._idx_path = Path(f"{output_prefix}.idx")
        self._meta_path = Path(f"{output_prefix}.meta.json")
        self._dtype = np.dtype(dtype)

        if self._dtype not in DTYPE_TO_CODE:
            raise ValueError(
                f"Unsupported dtype {self._dtype}. "
                f"Supported: {[str(np.dtype(d)) for d in DTYPE_TO_CODE]}"
            )

        self._bin_path.parent.mkdir(parents=True, exist_ok=True)
        self._bin_file = open(self._bin_path, "wb")
        self._sizes: list[int] = []
        self._doc_indices: list[int] = []
        self._total_tokens: int = 0
        self._finalized = False

    def add_item(self, token_ids: np.ndarray) -> None:
        if self._finalized:
            raise RuntimeError("Writer has already been finalized")
        token_ids = np.asarray(token_ids, dtype=self._dtype)
        if len(token_ids) == 0:
            logger.warning("Skipping empty token sequence")
            return
        self._bin_file.write(token_ids.tobytes(order="C"))
        self._sizes.append(len(token_ids))
        self._total_tokens += len(token_ids)

    def end_document(self) -> None:
        if self._finalized:
            raise RuntimeError("Writer has already been finalized")
        self._doc_indices.append(len(self._sizes) - 1)

    def finalize(self, metadata: Optional[dict] = None) -> Path:
        if self._finalized:
            raise RuntimeError("Writer has already been finalized")
        self._bin_file.close()
        self._write_index()
        if metadata is not None:
            self._write_metadata(metadata)
        self._finalized = True
        logger.info(
            "Wrote %d tokens in %d sequences (%d documents) to %s",
            self._total_tokens,
            len(self._sizes),
            len(self._doc_indices),
            self._bin_path,
        )
        return self._bin_path

    def _write_index(self) -> None:
        with open(self._idx_path, "wb") as f:
            f.write(_HDR_MAGIC)
            f.write(np.uint64(_HDR_VERSION).tobytes())
            f.write(np.uint8(DTYPE_TO_CODE[self._dtype]).tobytes())
            f.write(np.uint64(len(self._sizes)).tobytes())
            f.write(np.uint64(len(self._doc_indices)).tobytes())

            sizes = np.array(self._sizes, dtype=np.int32)
            f.write(sizes.tobytes())

            pointers = np.zeros(len(self._sizes), dtype=np.int64)
            if len(self._sizes) > 1:
                pointers[1:] = np.cumsum(sizes[:-1].astype(np.int64)) * self._dtype.itemsize
            f.write(pointers.tobytes())

            doc_idx = np.array(self._doc_indices, dtype=np.int64)
            f.write(doc_idx.tobytes())

    def _write_metadata(self, metadata: dict) -> None:
        from tether import __version__

        meta = {
            "version": "tether-1.0",
            "tether_version": __version__,
            "dtype": str(self._dtype),
            "dtype_code": DTYPE_TO_CODE[self._dtype],
            "num_sequences": len(self._sizes),
            "num_documents": len(self._doc_indices),
            "num_tokens": self._total_tokens,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def num_sequences(self) -> int:
        return len(self._sizes)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._finalized:
            self._bin_file.close()
        return False


class MMapIndexedDatasetReader:
    def __init__(self, prefix: str):
        self._idx_path = Path(f"{prefix}.idx")
        self._bin_path = Path(f"{prefix}.bin")
        self._read_index()
        self._bin_mmap = np.memmap(self._bin_path, dtype=self._dtype, mode="r")

    def _read_index(self) -> None:
        with open(self._idx_path, "rb") as f:
            magic = f.read(9)
            if magic != _HDR_MAGIC:
                raise ValueError(f"Invalid magic: {magic!r}")
            version = np.frombuffer(f.read(8), dtype=np.uint64)[0]
            if version != _HDR_VERSION:
                raise ValueError(f"Unsupported version: {version}")
            dtype_code = np.frombuffer(f.read(1), dtype=np.uint8)[0]
            self._dtype = np.dtype(DTYPE_MAP[int(dtype_code)])
            self._num_sequences = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
            self._num_documents = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
            self._sizes = np.frombuffer(f.read(self._num_sequences * 4), dtype=np.int32).copy()
            self._pointers = np.frombuffer(f.read(self._num_sequences * 8), dtype=np.int64).copy()
            self._doc_indices = np.frombuffer(
                f.read(self._num_documents * 8), dtype=np.int64
            ).copy()

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._num_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self._num_sequences})")
        start = self._pointers[idx] // self._dtype.itemsize
        length = self._sizes[idx]
        return np.array(self._bin_mmap[start : start + length])

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def num_sequences(self) -> int:
        return self._num_sequences

    @property
    def num_documents(self) -> int:
        return self._num_documents

    @property
    def sizes(self) -> np.ndarray:
        return self._sizes

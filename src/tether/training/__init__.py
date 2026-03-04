from tether.training.packing import SequencePacker

__all__ = [
    "SequencePacker",
    "PackedMapDataset",
    "PackedIterableDataset",
    "load_datamix",
]


def __getattr__(name):
    # Lazy imports for torch-dependent classes
    if name == "PackedMapDataset":
        from tether.training.datasets import PackedMapDataset

        return PackedMapDataset
    if name == "PackedIterableDataset":
        from tether.training.datasets import PackedIterableDataset

        return PackedIterableDataset
    if name == "load_datamix":
        from tether.training.datamix import load_datamix

        return load_datamix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

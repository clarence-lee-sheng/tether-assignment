from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DownloadConfig:
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = "train"
    output_dir: Path = Path("./data/raw")
    streaming: bool = False
    num_proc: Optional[int] = None
    max_samples: Optional[int] = None
    hf_token: Optional[str] = None


@dataclass
class TokenizationConfig:
    input_dir: Path
    output_prefix: str
    tokenizer_name_or_path: str
    text_column: str = "text"
    max_seq_length: Optional[int] = None
    batch_size: int = 1024
    num_workers: int = 4
    ray_address: Optional[str] = None
    add_bos: bool = False
    add_eos: bool = True


@dataclass
class DatasetSourceConfig:
    path: str
    weight: float = 1.0
    name: Optional[str] = None


@dataclass
class DataMixConfig:
    datasets: List[DatasetSourceConfig]
    seq_len: int = 2048
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataMixConfig":
        import yaml

        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")

        datasets_raw = raw.get("datasets")
        if not datasets_raw:
            raise ValueError("Config must contain a non-empty 'datasets' list")

        sources = []
        for i, ds in enumerate(datasets_raw):
            if "path" not in ds:
                raise ValueError(f"Dataset entry {i} missing required 'path' field")
            weight = ds.get("weight", 1.0)
            if weight <= 0:
                raise ValueError(f"Dataset entry {i} has non-positive weight: {weight}")
            sources.append(
                DatasetSourceConfig(
                    path=ds["path"],
                    weight=weight,
                    name=ds.get("name"),
                )
            )

        return cls(
            datasets=sources,
            seq_len=raw.get("seq_len", 2048),
            seed=raw.get("seed", 42),
        )

    @property
    def normalized_weights(self) -> List[float]:
        total = sum(ds.weight for ds in self.datasets)
        return [ds.weight / total for ds in self.datasets]


@dataclass
class SlurmConfig:
    job_name: str = "tether"
    partition: str = "cpu"
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 32
    mem: str = "64G"
    time: str = "12:00:00"
    account: Optional[str] = None
    output_log: str = "logs/%x_%j.out"
    error_log: str = "logs/%x_%j.err"
    extra_sbatch_flags: Dict[str, str] = field(default_factory=dict)

    # Environment
    conda_env: Optional[str] = None
    venv_path: Optional[str] = None
    module_loads: List[str] = field(default_factory=list)

    # Ray-specific
    ray_num_workers: int = 4
    ray_head_port: int = 6379
    ray_dashboard_port: int = 8265
    ray_object_store_memory: Optional[int] = None

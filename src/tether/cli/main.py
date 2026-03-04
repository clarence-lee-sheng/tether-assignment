import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(name="tether", help="Continued pretraining toolkit.", no_args_is_help=True)


def _run(cmd: list[str]) -> None:
    raise typer.Exit(subprocess.run(cmd).returncode)


def _flag(cmd: list[str], name: str, val) -> None:
    if val is None or val is False:
        return
    if val is True:
        cmd.append(name)
    elif isinstance(val, list):
        cmd += [name] + [str(v) for v in val]
    else:
        cmd += [name, str(val)]


@app.callback()
def main(log_level: str = typer.Option("INFO", help="DEBUG/INFO/WARNING/ERROR")):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def download(
    config: Path = typer.Option(...),
    only: Optional[List[str]] = typer.Option(None),
    max_samples: Optional[int] = typer.Option(None),
    tokenizer: Optional[str] = typer.Option(None),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    tokenized_dir: Optional[str] = typer.Option(None, "--tokenized-dir"),
    no_tokenize: bool = typer.Option(False, "--no-tokenize"),
    streaming: Optional[bool] = typer.Option(None),
    num_proc: Optional[int] = typer.Option(None, "--num-proc"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Download HF datasets from a YAML config."""
    cmd = [sys.executable, "-m", "tether.data.download_datasets", "--config", str(config)]
    _flag(cmd, "--only", only)
    _flag(cmd, "--max-samples", max_samples)
    _flag(cmd, "--tokenizer", tokenizer)
    _flag(cmd, "--output-dir", output_dir)
    _flag(cmd, "--tokenized-dir", tokenized_dir)
    _flag(cmd, "--no-tokenize", no_tokenize)
    _flag(cmd, "--num-proc", num_proc)
    _flag(cmd, "--dry-run", dry_run)
    if streaming is True:
        cmd.append("--streaming")
    elif streaming is False:
        cmd.append("--no-streaming")
    _run(cmd)


@app.command("download-models")
def download_models(
    config: Path = typer.Option(...),
    only: Optional[List[str]] = typer.Option(None),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Download models from HuggingFace Hub."""
    cmd = [sys.executable, "-m", "tether.data.download_models", "--config", str(config)]
    _flag(cmd, "--only", only)
    _flag(cmd, "--dry-run", dry_run)
    _run(cmd)


@app.command()
def tokenize(
    input_dir: Path = typer.Option(..., exists=True),
    output_prefix: str = typer.Option(...),
    tokenizer: str = typer.Option(...),
    text_column: str = typer.Option("text"),
    max_seq_length: Optional[int] = typer.Option(None),
    batch_size: int = typer.Option(1024),
    num_workers: int = typer.Option(4),
    add_bos: bool = typer.Option(False),
    add_eos: bool = typer.Option(True),
    ray_address: Optional[str] = typer.Option(None),
):
    """Tokenize a dataset into memmap format (.bin/.idx)."""
    from tether.data.tokenization import run_tokenization
    from tether.utils.config import TokenizationConfig

    out = run_tokenization(
        TokenizationConfig(
            input_dir=input_dir,
            output_prefix=output_prefix,
            tokenizer_name_or_path=tokenizer,
            text_column=text_column,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            num_workers=num_workers,
            ray_address=ray_address,
            add_bos=add_bos,
            add_eos=add_eos,
        )
    )
    typer.echo(f"Done: {out}")


@app.command("tokenize-datasets")
def tokenize_datasets(
    config: Path = typer.Option(...),
    only: Optional[List[str]] = typer.Option(None),
    tokenizer: Optional[str] = typer.Option(None),
    input_dir: Optional[str] = typer.Option(None, "--input-dir"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Batch-tokenize downloaded datasets from a YAML config."""
    cmd = [sys.executable, "-m", "tether.data.tokenize_datasets", "--config", str(config)]
    _flag(cmd, "--only", only)
    _flag(cmd, "--tokenizer", tokenizer)
    _flag(cmd, "--input-dir", input_dir)
    _flag(cmd, "--output-dir", output_dir)
    _flag(cmd, "--dry-run", dry_run)
    _run(cmd)


@app.command("validate-mix")
def validate_mix(
    config_path: Path = typer.Argument(..., exists=True),
):
    """Print stats for a data mix config."""
    from tether.data.memmap_writer import MMapIndexedDatasetReader
    from tether.training.datamix import validate_mix_paths
    from tether.training.packing import SequencePacker
    from tether.utils.config import DataMixConfig

    mix = DataMixConfig.from_yaml(config_path)
    validate_mix_paths(mix)
    w = mix.normalized_weights

    typer.echo(
        f"Seq length: {mix.seq_len}  |  Seed: {mix.seed}  |  Datasets: {len(mix.datasets)}\n"
    )

    tot_tok = tot_seq = tot_blk = 0
    for i, ds in enumerate(mix.datasets):
        reader = MMapIndexedDatasetReader(ds.path)
        packer = SequencePacker(reader, seq_len=mix.seq_len + 1)
        n_tok, n_seq, n_blk = packer.total_tokens, reader.num_sequences, packer.num_blocks
        tot_tok += n_tok
        tot_seq += n_seq
        tot_blk += n_blk
        typer.echo(
            f"  [{i}] {ds.name or ds.path}\n"
            f"      {n_seq:,} seqs, {n_tok:,} tokens, {n_blk:,} blocks  "
            f"(weight {ds.weight} -> {w[i]:.1%})"
        )

    typer.echo(f"\nTotal: {tot_seq:,} seqs, {tot_tok:,} tokens, {tot_blk:,} blocks")


def _torchrun(module: str, nproc: int, config, extra_args, **kwargs):
    cmd = ["torchrun", f"--nproc_per_node={nproc}", "-m", module]
    _flag(cmd, "--config", config)
    for k, v in kwargs.items():
        _flag(cmd, f"--{k.replace('_', '-')}", v)
    if extra_args:
        cmd += extra_args.split()
    _run(cmd)


@app.command()
def pretrain(
    nproc_per_node: int = typer.Option(1, "--nproc-per-node"),
    config: Optional[Path] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    tp_size: Optional[int] = typer.Option(None, "--tp-size"),
    extra_args: Optional[str] = typer.Option(None, "--extra-args"),
):
    """Continued pretraining with DTensor TP + FSDP2."""
    _torchrun("tether.pretrain", nproc_per_node, config, extra_args, model=model, tp_size=tp_size)


@app.command()
def distill(
    nproc_per_node: int = typer.Option(1, "--nproc-per-node"),
    config: Optional[Path] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    teacher: Optional[str] = typer.Option(None),
    tp_size: Optional[int] = typer.Option(None, "--tp-size"),
    extra_args: Optional[str] = typer.Option(None, "--extra-args"),
):
    """On-policy knowledge distillation with DTensor TP + FSDP2."""
    _torchrun(
        "tether.distill",
        nproc_per_node,
        config,
        extra_args,
        model=model,
        teacher=teacher,
        tp_size=tp_size,
    )


@app.command("eval")
def eval_cmd(
    config: Path = typer.Option(...),
    model: Optional[str] = typer.Option(None),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
):
    """Run lighteval evaluation."""
    cmd = [sys.executable, "-m", "tether.eval.eval_lighteval", "--config", str(config)]
    _flag(cmd, "--model", model)
    _flag(cmd, "--output-dir", output_dir)
    _flag(cmd, "--max-samples", max_samples)
    _flag(cmd, "--dry-run", dry_run)
    _flag(cmd, "--force", force)
    _run(cmd)


@app.command()
def merge(
    config: Optional[Path] = typer.Option(None),
    checkpoints: Optional[List[str]] = typer.Option(None),
    checkpoint_dir: Optional[str] = typer.Option(None, "--checkpoint-dir"),
    last: Optional[int] = typer.Option(None),
    weights: Optional[List[float]] = typer.Option(None),
    method: Optional[str] = typer.Option(None),
    density: Optional[float] = typer.Option(None),
    base_model: Optional[str] = typer.Option(None, "--base-model"),
    output: Optional[str] = typer.Option(None),
    dtype: Optional[str] = typer.Option(None),
):
    """Merge HF checkpoints (linear/slerp/ties/dare)."""
    cmd = [sys.executable, "-m", "tether.merge"]
    _flag(cmd, "--config", config)
    _flag(cmd, "--checkpoints", checkpoints)
    _flag(cmd, "--checkpoint-dir", checkpoint_dir)
    _flag(cmd, "--last", last)
    _flag(cmd, "--weights", weights)
    _flag(cmd, "--method", method)
    _flag(cmd, "--density", density)
    _flag(cmd, "--base-model", base_model)
    _flag(cmd, "--output", output)
    _flag(cmd, "--dtype", dtype)
    _run(cmd)


def cli():
    app()

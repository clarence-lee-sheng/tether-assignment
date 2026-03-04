# Tether

A continued pretraining toolkit for to perform distillation and model merging. Handles the full workflow: download datasets, tokenize into efficient memmap format, pack sequences, train with 2D parallelism (DTensor TP + FSDP2), and evaluate with lighteval.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Download datasets

Datasets are defined in `configs/utils/download_datasets.yaml`. Each entry specifies a HuggingFace dataset, an optional subset, max samples, and whether to tokenize.

```bash
tether download --config configs/utils/download_datasets.yaml
```

Downloads run in parallel with Ray. Tokenized data is saved as memmap files (`.bin` / `.idx`) for fast random-access loading during training.

## Download models

Models are defined in `configs/utils/download_models.yaml`.

```bash
tether download-models --config configs/utils/download_models.yaml

# Download a specific model:
tether download-models --config configs/utils/download_models.yaml --only SmolLM2-135M
```

## Tokenize datasets

Tokenization is configured in `configs/utils/tokenize_datasets.yaml`.

```bash
tether tokenize-datasets --config configs/utils/tokenize_datasets.yaml

# Tokenize specific datasets:
tether tokenize-datasets --config configs/utils/tokenize_datasets.yaml --only cosmopedia-v2

# Dry run:
tether tokenize-datasets --config configs/utils/tokenize_datasets.yaml --dry-run
```

Output is Megatron-format memmap files (`.bin` / `.idx`) written to `data/tokenized/<name>/`. Datasets are skipped if already tokenized.

## Train

Distributed pretraining using HuggingFace `LlamaForCausalLM` with DTensor TP + FSDP2. Configs are in `configs/experiments/`.

```bash
# Single GPU:
tether pretrain --nproc-per-node 1 --config configs/experiments/cpt_smollm2_135m_nemotron.yaml

# Multi-GPU (TP=3):
tether pretrain --nproc-per-node 3 --config configs/experiments/cpt_smollm2_135m_nemotron_tp3.yaml
```

Use `--export-hf exports/my_model` to save an HF-format checkpoint at the end of training.

## Distill

On-policy knowledge distillation with reverse KL loss. Runs teacher and student on the same data; loss = α · CE + (1-α) · KL(student ∥ teacher). Uses the same TP + FSDP2 parallelism as pretraining.

```bash
# Single GPU:
tether distill --nproc-per-node 1 \
    --config configs/experiments/distill_smollm2_135m_teacher_smollm_1p7b_nemotron.yaml

# Multi-GPU:
tether distill --nproc-per-node 4 \
    --config configs/experiments/distill_smollm2_135m_teacher_smollm_1p7b_nemotron.yaml
```

## Evaluate

Evaluation uses [lighteval](https://github.com/huggingface/lighteval) with vLLM. Tasks are defined in `src/tether/eval/tasks.py`. Configs are in `configs/eval/`.

```bash
# Evaluate a model:
tether eval --config configs/eval/eval.yaml

# Sweep over all checkpoints in a directory:
tether eval --config configs/eval/eval_sweep.yaml

# Dry run:
tether eval --config configs/eval/eval.yaml --dry-run
```

Already-evaluated checkpoints are skipped (use `--force` to re-run). Set `tensor_parallel_size` in the config to match the number of GPUs.

## Merge checkpoints

Merges HF-format checkpoints (exported via `--export-hf` during training) using [mergekit](https://github.com/arcee-ai/mergekit). Config is in `configs/merge/`.

```bash
tether merge --config configs/merge/merge_average.yaml
```

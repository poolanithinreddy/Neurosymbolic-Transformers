# Neuro-Symbolic Transformers (NST)

NST blends sequence models with interpretable predicate heads (R-CBM) trained under differentiable logic (DFOL). It supports constrained decoding and optional logic-aware reranking.

Highlights:
- LoRA-fine-tuned T5 models
- Relational Concept Bottleneck (R-CBM) heads
- Differentiable FOL (product t-norm, Reichenbach implication)
- Constrained decoding via hard label masks; optional logic rerank

## Install

We recommend a virtual environment.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

## Quick start (Mac smoke)

```bash
make setup
make smoke  # trains a tiny FEVER model (t5-small) and evaluates
```

## Colab

See `colab/colab_commands.md` for copy-paste cells. A quick path:

```bash
!git clone https://github.com/<your-username>/Neurosymbolic-Transformers.git nst
%cd nst
!pip install -U pip wheel
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -e ".[dev]"
!python -m spacy download en_core_web_sm
!python training/train.py --config configs/colab_lora.yaml --task fever
!python eval/fever.py --ckpt outputs/ckpt --data data/fever.tsv --split dev --report outputs/fever_dev.json --device cuda
```

## Datasets

- FEVER: small TSV sample included at `data/fever.tsv` (columns: split, claim, label, evidence)
- TruthfulQA: `python scripts/get_truthfulqa.py --out data/truthfulqa.tsv`
- COGS (placeholder TSVs): `python scripts/get_cogs.py --outdir data/cogs`

See `data/README.md` for schema details and examples.

## Training recipes

```bash
# FEVER (Mac quick)
python training/train.py --config configs/mac_quick.yaml --task fever --outdir outputs_quick
python eval/fever.py --ckpt outputs_quick/ckpt --data data/fever.tsv --split dev --report outputs_quick/fever_dev.json --device cpu

# FEVER (Colab, T5-base LoRA)
python training/train.py --config configs/colab_lora.yaml --task fever

# TruthfulQA (after download)
python eval/truthfulqa.py --ckpt outputs/ckpt --data data/truthfulqa.tsv --report outputs/tqa_dev.json --device cuda

# COGS (placeholder)
python eval/cogs.py --ckpt outputs/ckpt --data data/cogs --split test --report outputs/cogs_test.json --device cuda
```

Configs of interest: `configs/mac_quick.yaml`, `configs/colab_lora.yaml`, `configs/gpu_full.yaml`.

## Results (placeholders)

| Task      | Config              | Metric    | Value |
|-----------|---------------------|-----------|-------|
| FEVER     | mac_quick           | Dev Acc   | 0.25  |
| TruthfulQA| colab_lora (sample) | Acc       | TBD   |
| COGS      | placeholder         | EM / F1   | TBD   |

## Troubleshooting

- If HF Hub is temporarily unavailable, NST falls back to small models or offline stubs for smoke tests.
- On macOS, set `device: mps` in config if available; otherwise device auto-selection prefers CUDA > MPS > CPU.
- For large models (flan-t5-large), consider LoRA, gradient checkpointing, and 8-bit optimizers.

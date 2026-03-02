# Neuro-Symbolic Transformers (NST)

**Neural CEGIS: Counterexample-Guided Training for Provably Constraint-Satisfying Neural Networks**

NST brings counterexample-guided inductive synthesis (CEGIS) — the gold standard in formal verification — into the training loop of neural networks with symbolic constraints. A symbolic verifier finds inputs where the model violates domain constraints, feeds them back as targeted training data, and retrains until violations reach zero.

## Key Contributions

1. **Neural CEGIS**: A training paradigm where a symbolic verifier identifies constraint-violating inputs, generates counterexamples, and retrains the model — converging when violations reach zero.
2. **Augmented Lagrangian** with learned dual variable λ\* — the "price of logic" — eliminating manual constraint weight tuning.
3. **Three benchmarks**: multi-digit addition with carry propagation, kinship reasoning with distractors, and CLUTRR natural-language reasoning.
4. **Controlled baselines**: random replay, hard-example mining, and same-budget training to isolate the effect of constraint-targeted counterexamples.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Neural CEGIS Outer Loop                                │
│  ┌───────────────┐    ┌──────────────┐                  │
│  │ LEARNER       │◄───│ CE Buffer    │                  │
│  │ (Lagrangian   │    │ (targeted    │                  │
│  │  inner loop)  │    │  failures)   │                  │
│  └───────┬───────┘    └──────▲───────┘                  │
│          │                   │                          │
│          ▼                   │                          │
│  ┌───────────────┐    ┌──────┴───────┐                  │
│  │ Trained Model │───►│ VERIFIER     │                  │
│  │               │    │ (symbolic    │                  │
│  │               │    │  constraint  │                  │
│  │               │    │  checker)    │                  │
│  └───────────────┘    └──────────────┘                  │
│  Converges when CE count → 0                            │
└─────────────────────────────────────────────────────────┘
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
```

### macOS / Apple Silicon Notes

MPS (Metal Performance Shaders) is supported via `device: "auto"` in configs.

```bash
# If you hit MPS out-of-memory during training:
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python main.py train-fever-nst \
    --config configs/fever_mac_neural.yaml
```

**Troubleshooting:**

| Issue | Fix |
|-------|-----|
| MPS OOM | Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, reduce `batch_size` to 2, reduce `max_length` to 128 |
| `fp16` not supported on MPS | Set `fp16: false` in config (MPS autocast only supports bfloat16/float16 partially) |
| DeBERTa-v3 tokenizer crash | Ensure `sentencepiece>=0.1.99` and `protobuf>=4.0` are installed (see `requirements.txt`) |
| Slow tokenizer warnings | Benign "overflowing tokens" messages from `DebertaV2Tokenizer`; safe to ignore |
| `tiktoken` import error | Run `pip install tiktoken` (needed by some transformers tokenizer backends) |

**Pinned dependencies** (macOS-tested): see [requirements.txt](requirements.txt) for exact versions.

## Quick Start

```bash
# Verify everything works (193 tests)
python -m pytest tests/ -v

# Dataset statistics
python main.py multi-digit-stats
python main.py kinship-stats

# Train Neural CEGIS (the core contribution)
python main.py train-cegis --config configs/multi_digit_cegis.yaml

# Train kinship with Neural CEGIS
python main.py train-kinship-cegis --config configs/kinship_cegis.yaml
```

## Full Experiment Suite

```bash
# Run everything for the paper (~3 hrs on Colab T4)
chmod +x run_all.sh && ./run_all.sh

# Or use Makefile targets:
make experiments      # all multi-digit + kinship methods (3 seeds)
make baselines        # random-replay, hard-mining, same-budget
make latency          # inference latency benchmark
make plots            # alignment phase + CEGIS convergence figures
make tables           # LaTeX + Markdown results tables
make all-paper        # all of the above
```

## CLI Reference (18 commands)

| Command | Description |
|---------|-------------|
| `train` | Train single-digit addition model |
| `eval` | Evaluate digit-addition checkpoint |
| `train-multi-digit` | Train multi-digit addition (neural/soft/lagrangian) |
| `train-cegis` | Train multi-digit with Neural CEGIS |
| `train-kinship` | Train kinship relational reasoning |
| `train-kinship-cegis` | Train kinship with Neural CEGIS |
| `multi-seed` | Multi-seed run for error bars (mean ± std) |
| `baseline` | Run controlled baseline (random-replay/hard-mining/same-budget) |
| `ablation` | Run all single-digit ablation experiments |
| `results` | Generate results tables from output directories |
| `export-tables` | Export multi-seed tables (LaTeX/Markdown) |
| `latency` | Benchmark inference latency |
| `plot` | Generate alignment/convergence plots |
| `data-stats` | Print dataset statistics |
| `multi-digit-stats` | Multi-digit dataset statistics |
| `kinship-stats` | Kinship dataset statistics |
| `train-fever` | Train FEVER (original pipeline) |
| `train-fever-nst` | Train FEVER NLI with DeBERTa + NST constraints |
| `build-fever-wiki-cache` | Build SQLite wiki cache (avoids OOM) |
| `fever-stats` | Print FEVER dataset statistics + split hashes |
| `eval-fever` | Evaluate FEVER NLI checkpoint |
| `export-fever-tables` | Export FEVER results as Markdown tables |

## Configs

| Config | Method | Task |
|--------|--------|------|
| `multi_digit_neural.yaml` | Pure neural baseline | Multi-digit |
| `multi_digit_soft.yaml` | Soft constraints (fixed λ) | Multi-digit |
| `multi_digit_lagrangian.yaml` | Augmented Lagrangian | Multi-digit |
| `multi_digit_cegis.yaml` | **Neural CEGIS** | Multi-digit |
| `multi_digit_random_replay.yaml` | Random replay baseline | Multi-digit |
| `multi_digit_hard_mining.yaml` | Hard mining baseline | Multi-digit |
| `multi_digit_same_budget.yaml` | Same budget baseline | Multi-digit |
| `kinship_neural.yaml` | Pure neural baseline | Kinship |
| `kinship_lagrangian.yaml` | Augmented Lagrangian | Kinship |
| `kinship_cegis.yaml` | **Neural CEGIS** | Kinship |
| `fever_gold_neural.yaml` | Pure neural baseline | FEVER (Gold Evidence) |
| `fever_gold_nst_soft.yaml` | Soft constraints (fixed λ) | FEVER (Gold Evidence) |
| `fever_gold_nst_cegis.yaml` | **Neural CEGIS** | FEVER (Gold Evidence) |
| `fever_gold_nst_gated.yaml` | **ECCG** (novel) | FEVER (Gold Evidence) |
| `fever_gold_smoke.yaml` | Smoke test (200 samples) | FEVER (Gold Evidence) |
| `fever_pipeline_neural.yaml` | Pure neural baseline | FEVER (Full Pipeline) |
| `fever_pipeline_nst_cegis.yaml` | **Neural CEGIS** | FEVER (Full Pipeline) |

## Colab

See `colab/nst_playbook.py` for copy-paste cells. Runs the full experiment suite in ~40 minutes on a T4 GPU.

For FEVER specifically, see `colab/fever_playbook.py` (12 cells, ~45 min on T4).

```bash
!git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git nst
%cd nst
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -e ".[dev]" -q
!python main.py build-fever-wiki-cache   # one-time, avoids OOM
!python main.py train-cegis --config configs/multi_digit_cegis.yaml
```

## FEVER Fact Verification

NST includes a full FEVER fact verification pipeline with two evaluation settings:

- **Setting A (Gold Evidence)**: Oracle evidence → Label Accuracy. Uses `FeverGoldDataset` with canonical FEVER labels (SUPPORTS / REFUTES / NOT ENOUGH INFO).
- **Setting B (Full Pipeline)**: BM25 retrieval + NLI → End-to-End FEVER Score. Uses `FeverPipelineDataset` with leakage guards.

**Architecture**: DeBERTa-v3-base for 3-class NLI + 5 differentiable constraints (date contradiction → ¬SUPPORTS, number contradiction → ¬SUPPORTS, negation mismatch → ¬SUPPORTS, low entity overlap → NEI, empty evidence → NEI).

```bash
# Build wiki cache (one-time, avoids OOM on Colab)
python main.py build-fever-wiki-cache

# Dataset stats + split hashes
python main.py fever-stats

# Train neural baseline (gold evidence)
python main.py train-fever-nst --config configs/fever_gold_neural.yaml

# Train NST-CEGIS (gold evidence)
python main.py train-fever-nst --config configs/fever_gold_nst_cegis.yaml

# Multi-seed for error bars
python main.py multi-seed --task train-fever-nst \
    --config configs/fever_gold_nst_cegis.yaml --seeds 42,43,44

# Evaluate a checkpoint
python main.py eval-fever --ckpt outputs_fever_gold_neural/ckpt/best_model.pt
```

## Reproducibility

| Item | Detail |
|------|--------|
| Hardware | Colab T4 (16 GB VRAM) or macOS with MPS |
| Seeds | {42, 43, 44} — mean ± std over 3 seeds |
| Framework | PyTorch 2.9.0, Python ≥ 3.10 |
| Tests | 193 tests, `pytest tests/ -v` |
| License | MIT |

## Data & Storage

### Where data lives

| Artifact | Path | Size | Committed? |
|----------|------|------|------------|
| Fixture TSVs (toy data) | `data/*.tsv` | ~16 KB | **Yes** — tiny test fixtures |
| Dataset loaders (source) | `data/*.py` | ~80 KB | **Yes** — source code |
| FEVER wiki cache (SQLite) | `data/fever_wiki.db` | ~25 MB | **No** — generated at runtime |
| Training outputs | `outputs_*/` | Varies | **No** — gitignored |
| HuggingFace cache | `~/.cache/huggingface/` | 1–20 GB | **No** — system-wide cache |

### Building caches (do NOT commit them)

```bash
# Build full FEVER wiki cache (~200s, ~25 MB SQLite)
python main.py build-fever-wiki-cache

# Check dataset stats + split hashes
python main.py fever-stats
```

### Cleaning up downloaded artifacts

```bash
# Remove repo-local outputs and caches
bash scripts/cleanup_local_data.sh

# Also clear HuggingFace system cache (~1–20 GB)
bash scripts/cleanup_local_data.sh --hf

# Preview what would be deleted (dry run)
bash scripts/cleanup_local_data.sh --dry
```

> **⚠️ Warning:** Do not commit datasets, caches, or model checkpoints to git.
> The `.gitignore` blocks `*.db`, `*.safetensors`, `*.pt`, `*.bin`, and `outputs_*/`.
> If you accidentally stage a large file, run: `git rm --cached <file>`

## Project Structure

```
nst/
├── main.py                 # 18-command CLI
├── PAPER.md                # Submission draft
├── RUNBOOK.md              # Full FEVER experiment playbook
├── RESULTS.md              # Experiment results tracker
├── run_all.sh              # Full experiment script
├── configs/                # 25 YAML experiment configs
├── models/                 # CNN perception, Transformer, neuro-symbolic models
├── training/               # Training loops, CEGIS, baselines, multi-seed
├── symbolic/               # Lagrangian, constraint solver, rule engine
├── data/                   # Datasets (digit addition, kinship, CLUTRR, FEVER)
├── eval/                   # Evaluation, calibration, FEVER Score, rule checking
├── scripts/                # Latency benchmark, plots, table export, cleanup
├── results/                # Table generation (LaTeX/Markdown)
├── tests/                  # 193 unit tests
└── colab/                  # One-click Colab playbook
```

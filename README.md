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

## Quick Start

```bash
# Verify everything works (119 tests)
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
| `fever_pipeline_neural.yaml` | Pure neural baseline | FEVER (Full Pipeline) |
| `fever_pipeline_nst_cegis.yaml` | **Neural CEGIS** | FEVER (Full Pipeline) |

## Colab

See `colab/nst_playbook.py` for copy-paste cells. Runs the full experiment suite in ~40 minutes on a T4 GPU.

For FEVER specifically, see `colab/fever_playbook.py` (10 cells, ~30 min on T4).

```bash
!git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git nst
%cd nst
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -e ".[dev]" -q
!python main.py train-cegis --config configs/multi_digit_cegis.yaml
```

## FEVER Fact Verification

NST includes a full FEVER fact verification pipeline with two evaluation settings:

- **Setting A (Gold Evidence)**: Oracle evidence → Label Accuracy. Uses `FeverGoldDataset` with canonical FEVER labels (SUPPORTS / REFUTES / NOT ENOUGH INFO).
- **Setting B (Full Pipeline)**: BM25 retrieval + NLI → End-to-End FEVER Score. Uses `FeverPipelineDataset` with leakage guards.

**Architecture**: DeBERTa-v3-base for 3-class NLI + 5 differentiable constraints (date contradiction → ¬SUPPORTS, number contradiction → ¬SUPPORTS, negation mismatch → ¬SUPPORTS, low entity overlap → NEI, empty evidence → NEI).

```bash
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
| Hardware | Colab T4 (16 GB VRAM) |
| Seeds | {42, 43, 44} — mean ± std over 3 seeds |
| Framework | PyTorch 2.9.0, Python ≥ 3.10 |
| Tests | 119 tests, `pytest tests/ -v` |
| License | MIT |

## Project Structure

```
nst/
├── main.py                 # 18-command CLI
├── PAPER.md                # Submission draft
├── run_all.sh              # Full experiment script
├── configs/                # 21 YAML experiment configs
├── models/                 # CNN perception, Transformer, neuro-symbolic models
├── training/               # Training loops, CEGIS, baselines, multi-seed
├── symbolic/               # Lagrangian, constraint solver, rule engine
├── data/                   # Datasets (digit addition, kinship, CLUTRR)
├── eval/                   # Evaluation, calibration, rule checking
├── scripts/                # Latency benchmark, plots, table export
├── results/                # Table generation (LaTeX/Markdown)
├── tests/                  # 119 unit tests
└── colab/                  # One-click Colab playbook
```

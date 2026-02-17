# Neuro-Symbolic Transformers (NST)

**Neural CEGIS: Counterexample-Guided Training for Provably Constraint-Satisfying Neural Networks**

NST blends neural perception with differentiable symbolic reasoning. The core novelty is **Neural CEGIS** — Counterexample-Guided Inductive Synthesis adapted for neural network training — which provably drives constraint violations to zero through a verification loop.

## Key Contributions

1. **Neural CEGIS**: A training paradigm where a symbolic verifier finds specific inputs where the model violates constraints, feeds them back as counterexamples, and retrains — converging when violations reach zero.
2. **Augmented Lagrangian** dual-variable framework with learned price-of-logic λ*.
3. **Multi-digit addition benchmark**: 2-digit + 2-digit with carry propagation — compositional split forces genuine generalisation (train WITHOUT carries, test WITH carries).
4. **Kinship relational reasoning**: with distractors, label corruption, balanced labels, extended compositional depth (1–6).

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
pip install -U pip
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run all 102 tests
python -m pytest tests/ -v

# Dataset statistics
python main.py multi-digit-stats
python main.py kinship-stats

# Train multi-digit baselines
python main.py train-multi-digit --config configs/multi_digit_neural.yaml
python main.py train-multi-digit --config configs/multi_digit_soft.yaml
python main.py train-multi-digit --config configs/multi_digit_lagrangian.yaml

# Train with Neural CEGIS (the core contribution)
python main.py train-cegis --config configs/multi_digit_cegis.yaml

# Train kinship with distractors + corruption
python main.py train-kinship --config configs/kinship_cegis.yaml
```

## Colab

See `colab/nst_playbook.py` for 9 copy-paste cells. Runs the full experiment suite in ~40 minutes on a T4 GPU.

```bash
!git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git nst
%cd nst
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -e ".[dev]" -q
!python main.py train-cegis --config configs/multi_digit_cegis.yaml
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

## Digit Addition (Neuro-Symbolic Experiment)

NST now includes a **digit-addition** neuro-symbolic experiment that demonstrates:
- CNN-based digit perception (neural)
- Differentiable arithmetic constraints via product t-norm (symbolic)
- Optional Z3 SMT solver verification at inference (hard constraints)
- Compositional generalisation evaluation (IID vs unseen digit pairs)

### Quick run (single mode)

```bash
# Soft constraints (recommended)
python main.py train --config configs/digit_add_soft.yaml

# Evaluate (optionally with --hard for Z3 repair)
python main.py eval --ckpt outputs_digit_add_soft/ckpt/best_model.pt
python main.py eval --ckpt outputs_digit_add_soft/ckpt/best_model.pt --hard
```

### Run all ablations

```bash
python main.py ablation
```

This trains five variants: Pure Neural, NST-Soft, NST-Hard, NST-CAGrad, **NST-Lagrangian**,
and prints a comparison table.

### Ablation configs

| Config | Mode | Description |
|--------|------|-------------|
| `configs/digit_add_neural.yaml` | neural | Pure CNN + MLP sum head (no symbolic signal) |
| `configs/digit_add_soft.yaml` | soft | CNN + differentiable arithmetic constraint loss |
| `configs/digit_add_hard.yaml` | hard | Soft-trained + Z3 repair at inference |
| `configs/digit_add_cagrad.yaml` | soft | Soft + CAGrad multi-task gradient balancing |
| `configs/digit_add_lagrangian.yaml` | lagrangian | Augmented Lagrangian with learned λ* |

### Dataset statistics

```bash
python main.py data-stats --threshold 9
```

### Expected results

| Model | Digit Acc (IID) | Sum Acc (IID) | CSR (IID) | Sum Acc (Comp) | CSR (Comp) |
|-------|-----------------|---------------|-----------|----------------|------------|
| Pure Neural | — | — | — | — | — |
| NST-Soft | — | — | — | — | — |
| NST-Hard | — | — | — | — | — |
| NST-CAGrad | — | — | — | — | — |
| NST-Lagrangian | — | — | — | — | — |

Fill in after running `python main.py ablation`.

## Kinship Relational Reasoning (Benchmark 2)

A CLUTRR-inspired synthetic benchmark testing **systematic compositional generalisation** on kinship relations.
Models train on short chains (depth 1–3) and must generalise to longer chains (depth 4–5).

### Quick run

```bash
# Dataset statistics
python main.py kinship-stats

# Neural baseline
python main.py train-kinship --config configs/kinship_neural.yaml

# Soft constraints (fixed λ)
python main.py train-kinship --config configs/kinship_soft.yaml

# Lagrangian (learned λ*)
python main.py train-kinship --config configs/kinship_lagrangian.yaml
```

### Kinship configs

| Config | Mode | Description |
|--------|------|-------------|
| `configs/kinship_neural.yaml` | neural | Transformer encoder only |
| `configs/kinship_soft.yaml` | soft | Transformer + fixed constraint weight |
| `configs/kinship_lagrangian.yaml` | lagrangian | Transformer + learned dual variable λ* |

## Augmented Lagrangian Optimisation

The core technical contribution. Instead of tuning a fixed hyperparameter λ for the logic loss:

$$\min_\theta \max_{\lambda \geq 0} \; \mathcal{L}_{\text{task}} + \lambda \cdot (\mathcal{L}_{\text{logic}} - \varepsilon) + \frac{\rho}{2} [\max(0, \mathcal{L}_{\text{logic}} - \varepsilon)]^2$$

The dual variable λ is updated each epoch:

$$\lambda \leftarrow \max(0,\; \lambda + \alpha \cdot (\mathcal{L}_{\text{logic}} - \varepsilon))$$

Key hyperparameters (in YAML config):
- `epsilon` (ε): constraint tolerance (default 0.05)
- `alpha` (α): dual learning rate (default 0.01)
- `rho` (ρ): quadratic penalty weight (default 1.0)
- `lam_max`: upper bound on λ (default 50.0)

The converged λ* is the **price of logic** — how much the model must pay (in task loss) per unit of constraint satisfaction.

## Calibration Metrics

After training, evaluate model calibration:
- **ECE** (Expected Calibration Error, 15 bins)
- **Brier score** (multi-class MSE)
- **Noise robustness** (accuracy under Gaussian noise σ ∈ {0, 0.1, 0.2, 0.3, 0.5})

See `eval/calibration_metrics.py` for API details.

## Results Tables

Generate publication-ready tables from output directories:

```bash
# Markdown
python main.py results --format markdown

# LaTeX
python main.py results --format latex --save results_table.tex
```

## Colab Playbook

See `colab/nst_playbook.py` for a **9-cell Colab playbook** covering:
1. Environment setup
2. Dataset statistics
3. Smoke test (2 epochs)
4. Full digit-addition ablation
5. Evaluation + comparison table
6. Kinship training
7. λ* trajectory plot
8. LaTeX / Markdown results tables
9. Test suite execution

## Results (FEVER — existing)

| Task      | Config              | Metric    | Value |
|-----------|---------------------|-----------|-------|
| FEVER     | mac_quick           | Dev Acc   | 0.25  |
| TruthfulQA| colab_lora (sample) | Acc       | TBD   |
| COGS      | placeholder         | EM / F1   | TBD   |

## Project Structure

```
nst/
├── main.py                        # Unified CLI entry point
├── PAPER.md                       # Paper draft (10-researcher war room + blueprint)
├── PAPER_v1.md                    # Archived v1 paper draft
├── configs/
│   ├── digit_add_neural.yaml      # Pure neural baseline
│   ├── digit_add_soft.yaml        # Soft constraint
│   ├── digit_add_hard.yaml        # Hard constraint (Z3 at inference)
│   ├── digit_add_cagrad.yaml      # CAGrad multi-task
│   ├── digit_add_lagrangian.yaml  # Augmented Lagrangian (core novelty)
│   ├── kinship_neural.yaml        # Kinship neural baseline
│   ├── kinship_soft.yaml          # Kinship soft constraint
│   ├── kinship_lagrangian.yaml    # Kinship Lagrangian
│   ├── mac_quick.yaml             # FEVER quick config
│   └── ...
├── data/
│   ├── digit_addition.py          # Synthetic digit-addition dataset
│   ├── kinship.py                 # Synthetic kinship relational dataset
│   ├── fever.tsv                  # FEVER sample data
│   └── ...
├── models/
│   ├── perception.py              # CNN digit encoder
│   ├── nst_model.py               # Digit-addition neuro-symbolic model
│   ├── nst_kinship.py             # Kinship Transformer model
│   ├── rcbm.py                    # R-CBM predicate heads
│   ├── heads.py                   # Unary/Binary heads
│   └── ...
├── symbolic/
│   ├── lagrangian.py              # Augmented Lagrangian dual-variable optimiser
│   ├── constraint_solver.py       # Soft constraints + Z3 hard verification
│   └── rule_engine.py             # YAML rule loader + evaluation
├── logic/
│   ├── logic.py                   # T-norm fuzzy logic primitives
│   ├── rules.yaml                 # Domain rules (Horn clauses)
│   └── predicates.yaml            # Predicate definitions
├── training/
│   ├── train_nst.py               # Digit-addition training loop
│   ├── train_kinship.py           # Kinship training loop
│   ├── train.py                   # FEVER training loop
│   ├── cagrad.py                  # Conflict-averse gradient descent
│   └── losses.py                  # Loss utilities
├── eval/
│   ├── eval_nst.py                # Digit-addition evaluation
│   ├── calibration_metrics.py     # ECE, Brier, noise robustness
│   ├── fever.py                   # FEVER evaluation
│   └── ...
├── results/
│   └── __init__.py                # LaTeX / Markdown table generators
├── colab/
│   ├── nst_playbook.py            # 9-cell Colab playbook
│   ├── colab_commands.md          # FEVER Colab commands
│   └── colab_setup.py             # Colab setup script
├── tests/
│   ├── test_symbolic.py           # Symbolic module unit tests
│   ├── test_lagrangian.py         # Lagrangian optimiser tests
│   ├── test_kinship.py            # Kinship dataset/model tests
│   ├── test_calibration.py        # Calibration metrics tests
│   ├── test_logic.py              # Logic primitive tests
│   ├── test_rules.py              # Rule evaluation tests
│   └── ...
└── decoding/
    ├── hard_masks.py              # Constrained decoding
    └── rerank.py                  # Logic-aware reranking
```

## Troubleshooting

- If HF Hub is temporarily unavailable, NST falls back to small models or offline stubs for smoke tests.
- On macOS, set `device: mps` in config if available; otherwise device auto-selection prefers CUDA > MPS > CPU.
- For large models (flan-t5-large), consider LoRA, gradient checkpointing, and 8-bit optimizers.

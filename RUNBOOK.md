# FEVER RUNBOOK — Neurosymbolic Transformers

## Overview

This runbook describes how to reproduce all FEVER fact verification experiments,
from environment setup through final results. Every command is deterministic
given the same seed, data, and library versions.

## Prerequisites

| Requirement | Minimum | Tested |
|---|---|---|
| Python | 3.10+ | 3.13.11 |
| PyTorch | 2.9.0 | 2.9.0 |
| Transformers | 4.46.3 | 4.46.3 |
| datasets | 2.21.0 | 2.21.0 |
| GPU (recommended) | T4 16 GB | MPS / T4 / A100 |

## 1. Environment Setup

```bash
# Clone and setup
git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git
cd Neurosymbolic-Transformers/nst
git checkout fever-sota

# Create venv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Or: pip install -r requirements.txt
```

## 2. Build Wiki Cache

The wiki cache resolves FEVER evidence annotations to actual Wikipedia text.
**This step is mandatory for meaningful results.**

```bash
# Full build (~4 min, downloads 1.7 GB wiki dump, produces 24 MB SQLite)
python main.py build-fever-wiki-cache

# Verify
python main.py build-fever-wiki-cache --stats_only
# Expected: ~14,363 pages, 98.8% coverage
```

## 3. Smoke Tests

### Micro smoke (tiny BERT, <30s)
```bash
python main.py train-fever-nst --config configs/fever_micro_smoke.yaml
# Expected: dev_acc ≈ 0.28, runtime <30s
```

### DeBERTa smoke (200 examples, <30s on GPU)
```bash
python main.py train-fever-nst --config configs/fever_gold_smoke.yaml
# Expected: dev_acc ≈ 0.32, runtime <30s
```

## 4. Full Experiments (Setting A: Gold Evidence)

All experiments use **DeBERTa-v3-base** with gold evidence (oracle Setting A).
Each config uses `dev_test_ratio: 0.1` to hold out 10% of labelled_dev for
final evaluation, preventing overfitting to the dev set.

### 4a. Neural Baseline
```bash
python main.py train-fever-nst --config configs/fever_gold_neural.yaml
# Expected: ~88-92% label accuracy, ~3h on T4
```

### 4b. NST Soft Constraints (fixed λ=0.1)
```bash
python main.py train-fever-nst --config configs/fever_gold_nst_soft.yaml
```

### 4c. NST Lagrangian (adaptive λ)
```bash
python main.py train-fever-nst --config configs/fever_gold_lagrangian.yaml
```

### 4d. NST CEGIS (counterexample-guided)
```bash
python main.py train-fever-nst --config configs/fever_gold_nst_cegis.yaml
```

### 4e. NST Gated / ECCG (novel: Evidence-Conditioned Constraint Gating)
```bash
python main.py train-fever-nst --config configs/fever_gold_nst_gated.yaml
```

### 4f. Multi-seed runs (for mean ± std)
```bash
python main.py multi-seed --config configs/fever_gold_neural.yaml --seeds 42,1337,2024
python main.py multi-seed --config configs/fever_gold_nst_gated.yaml --seeds 42,1337,2024
```

## 5. Full Experiments (Setting B: Retrieved Evidence)

Pipeline mode uses BM25 retriever — gold evidence is NEVER accessed.

```bash
python main.py train-fever-nst --config configs/fever_pipeline_neural.yaml
python main.py train-fever-nst --config configs/fever_pipeline_nst_cegis.yaml
```

## 6. Evaluation

### Export results tables
```bash
python main.py export-fever-tables --results_dir outputs_fever_gold_neural outputs_fever_gold_nst_soft outputs_fever_gold_lagrangian outputs_fever_gold_nst_cegis outputs_fever_gold_nst_gated
```

### Run integrity checks
```bash
python scripts/verify_no_leakage.py --config configs/fever_gold_neural.yaml
```

## 7. Key Metrics

| Metric | Setting | What it measures |
|---|---|---|
| **Label Accuracy** | A (gold) | NLI classification accuracy |
| **FEVER Score** | B (pipeline) | Label correct AND evidence sufficient |
| **ECE ↓** | Both | Expected Calibration Error |
| **Brier ↓** | Both | Brier score (lower = better calibrated) |
| **Recall@5** | B only | Retrieval coverage |

## 8. Anti-Leakage Discipline

1. Gold evidence is NEVER used in pipeline mode (enforced by `FeverPipelineDataset`)
2. CEGIS counterexamples are mined from TRAINING set only
3. Test set is NEVER touched — all tuning on dev or dev_test
4. No hyperparameter tuning on test set
5. `dev_test_ratio: 0.1` ensures final numbers come from held-out data
6. Split hashes logged for every run

## 9. Reproducibility Checklist

- [ ] Same `seed` in config (default: 42)
- [ ] Same `requirements.txt` / `pyproject.toml` versions
- [ ] Wiki cache built with same HF dataset version
- [ ] `torch.backends.cudnn.deterministic = True`
- [ ] DataLoader uses seeded `torch.Generator`
- [ ] Split hashes match between runs

## 10. Directory Structure

```
outputs_fever_gold_neural/
├── report.json          # Full metrics + config
├── train_log.json       # Per-epoch/step training log
├── ckpt/                # Best model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── training_state.pt
└── lambda_trajectory.json  # (Lagrangian/CEGIS/gated modes only)
```

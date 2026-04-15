# NST Runbook

Step-by-step instructions for reproducing all experiments. Every command is deterministic given the same seed, data, and library versions.

---

## Prerequisites

| Requirement | Minimum | Tested |
|-------------|---------|--------|
| Python | 3.10+ | 3.13 |
| PyTorch | 2.2.0+ | 2.9.0 |
| Transformers | 4.40.0+ | 4.46.3 |
| GPU (FEVER runs) | T4 16 GB | T4 / A100 / MPS |

---

## 1. Setup

```bash
git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git
cd Neurosymbolic-Transformers/nst

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 2. Smoke Test (no GPU, <60s)

Verifies the full stack — imports, dataset generators, training loop, latency benchmark — without needing a GPU or downloading any models.

```bash
bash scripts/smoke_test.sh
# Expected: 15 passed, 0 failed
```

For a quicker check (skips unit test suite):

```bash
bash scripts/smoke_test.sh --fast
```

---

## 3. Unit Tests

```bash
TRANSFORMERS_OFFLINE=1 python -m pytest tests/ -v
# Expected: 232 passed, 1 skipped
```

---

## 4. Multi-Digit Addition Experiments

### 4a. Individual training runs

```bash
# Neural baseline
python main.py train-multi-digit --config configs/multi_digit_neural.yaml

# Augmented Lagrangian (adaptive λ)
python main.py train-multi-digit --config configs/multi_digit_lagrangian.yaml

# Neural CEGIS
python main.py train-cegis --config configs/multi_digit_cegis.yaml
```

### 4b. Multi-seed (for mean ± std)

```bash
python main.py multi-seed --task train-multi-digit --config configs/multi_digit_neural.yaml --seeds 42,43,44
python main.py multi-seed --task train-multi-digit --config configs/multi_digit_lagrangian.yaml --seeds 42,43,44
python main.py multi-seed --task train-cegis --config configs/multi_digit_cegis.yaml --seeds 42,43,44
```

### 4c. Controlled baselines

```bash
python main.py baseline --method random-replay --config configs/multi_digit_random_replay.yaml --seeds 42,43,44
python main.py baseline --method hard-mining --config configs/multi_digit_hard_mining.yaml --seeds 42,43,44
python main.py baseline --method same-budget --config configs/multi_digit_same_budget.yaml --seeds 42,43,44
```

### 4d. Dataset statistics

```bash
python main.py multi-digit-stats
```

---

## 5. Kinship Reasoning Experiments

```bash
python main.py multi-seed --task train-kinship --config configs/kinship_neural.yaml --seeds 42,43,44
python main.py multi-seed --task train-kinship --config configs/kinship_lagrangian.yaml --seeds 42,43,44
python main.py multi-seed --task train-kinship-cegis --config configs/kinship_cegis.yaml --seeds 42,43,44

python main.py kinship-stats
```

---

## 6. FEVER Fact Verification

### 6a. Build wiki cache (one-time)

Required for any FEVER run beyond the micro smoke test.

```bash
# Full build (~4 min, downloads ~1.7 GB wiki dump, produces ~25 MB SQLite)
python main.py build-fever-wiki-cache

# Verify
python main.py build-fever-wiki-cache --stats_only
# Expected: ~14,363 pages, ~98.8% coverage
```

### 6b. Smoke tests (CPU-friendly)

```bash
# Tiny BERT, 50 train / 25 dev, <30s CPU
python main.py train-fever-nst --config configs/fever_micro_smoke.yaml
# Expected: dev_acc ≈ 0.28, no errors

# DeBERTa-v3, 200 samples, ~30s GPU
python main.py train-fever-nst --config configs/fever_gold_smoke.yaml
```

### 6c. Full experiments — Setting A (Gold Evidence)

Each config runs ~3h on a T4 GPU.

```bash
python main.py train-fever-nst --config configs/fever_gold_neural.yaml
python main.py train-fever-nst --config configs/fever_gold_nst_soft.yaml
python main.py train-fever-nst --config configs/fever_gold_lagrangian.yaml
python main.py train-fever-nst --config configs/fever_gold_nst_cegis.yaml
python main.py train-fever-nst --config configs/fever_gold_nst_gated.yaml
```

### 6d. Multi-seed (for mean ± std)

```bash
python main.py multi-seed --task train-fever-nst \
    --config configs/fever_gold_nst_cegis.yaml --seeds 42,43,44
```

### 6e. Setting B (Full Pipeline, Retrieved Evidence)

```bash
python main.py train-fever-nst --config configs/fever_pipeline_neural.yaml
python main.py train-fever-nst --config configs/fever_pipeline_nst_cegis.yaml
```

### 6f. Evaluate a checkpoint

```bash
python main.py eval-fever --ckpt outputs_fever_gold_neural/ckpt/best_model.pt
```

### 6g. Dataset statistics

```bash
python main.py fever-stats
```

---

## 7. Tables and Figures

```bash
# Results tables (requires completed training runs)
python main.py export-tables --task multi_digit --format markdown --outdir results/
python main.py export-tables --task kinship --format markdown --outdir results/
python main.py export-fever-tables --results_dir outputs_fever_gold_neural outputs_fever_gold_nst_cegis

# Figures
mkdir -p figures
python scripts/plot_alignment.py --logdir outputs_multi_digit_lagrangian --outdir figures/
python scripts/plot_alignment.py --logdir outputs_multi_digit_cegis --cegis --outdir figures/

# Inference latency
python scripts/benchmark_latency.py --n_samples 500 --device cpu --json results/latency_cpu.json
```

---

## 8. Data Integrity Checks

```bash
python main.py leakage-check
python scripts/verify_no_leakage.py --config configs/fever_gold_neural.yaml
```

---

## 9. Makefile Targets

```bash
make setup        # install dependencies
make test         # run unit tests
make smoke        # run smoke test
make experiments  # full multi-digit + kinship suite, 3 seeds
make baselines    # controlled baseline suite
make latency      # latency benchmark
make plots        # alignment + convergence figures
make tables       # LaTeX + Markdown results tables
make all-paper    # everything above
make fmt          # format code (ruff + black)
make lint         # lint (ruff + mypy)
```

---

## 10. Output Directory Structure

```
outputs_<config_name>/
├── report.json           # Final metrics, config snapshot
├── train_log.json        # Per-step/epoch training log
├── ckpt/                 # Best model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── training_state.pt
└── lambda_trajectory.json  # (Lagrangian / CEGIS modes only)
```

---

## 11. Reproducibility Checklist

- [ ] Same `seed` in config (default: 42)
- [ ] Same library versions (`requirements.txt`)
- [ ] Wiki cache built with same HF dataset version
- [ ] `torch.backends.cudnn.deterministic = True` (set in all training scripts)
- [ ] DataLoader uses seeded `torch.Generator`
- [ ] Split hashes match between runs (logged in `report.json`)

# Neural CEGIS — Colab Reproduction Guide

**Estimated time:** ~3 hours on T4 GPU (full suite) · ~40 min (quick mode)

## Prerequisites

- Google Colab with GPU runtime (T4 recommended)
- No local setup required — everything is installed in the notebook

## Quick Start

Open a new Colab notebook and paste the cells below in order.

---

### Cell 1: Setup (2–3 min)

```python
!git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git nst
%cd nst

!pip install -U pip wheel -q
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -e ".[dev]" -q
!pip install z3-solver matplotlib -q

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

!python -m pytest tests/ -q --no-cov
```

### Cell 2: Dataset Statistics

```python
!python main.py multi-digit-stats
print()
!python main.py kinship-stats
```

### Cell 3: Multi-Digit — All Methods (3 seeds, ~30 min)

```python
SEEDS = "42,43,44"

# Pure Neural (should FAIL on carry splits)
!python main.py multi-seed --task train-multi-digit \
    --config configs/multi_digit_neural.yaml --seeds {SEEDS}

# NST-Lagrangian
!python main.py multi-seed --task train-multi-digit \
    --config configs/multi_digit_lagrangian.yaml --seeds {SEEDS}

# Neural CEGIS (the core contribution)
!python main.py multi-seed --task train-cegis \
    --config configs/multi_digit_cegis.yaml --seeds {SEEDS}
```

### Cell 4: Controlled Baselines (3 seeds, ~30 min)

```python
SEEDS = "42,43,44"

# Random Replay
!python main.py baseline --method random-replay \
    --config configs/multi_digit_random_replay.yaml --seeds {SEEDS}

# Hard Example Mining
!python main.py baseline --method hard-mining \
    --config configs/multi_digit_hard_mining.yaml --seeds {SEEDS}

# Same Budget
!python main.py baseline --method same-budget \
    --config configs/multi_digit_same_budget.yaml --seeds {SEEDS}
```

### Cell 5: Kinship — All Methods (3 seeds, ~45 min)

```python
SEEDS = "42,43,44"

!python main.py multi-seed --task train-kinship \
    --config configs/kinship_neural.yaml --seeds {SEEDS}

!python main.py multi-seed --task train-kinship \
    --config configs/kinship_lagrangian.yaml --seeds {SEEDS}

!python main.py multi-seed --task train-kinship-cegis \
    --config configs/kinship_cegis.yaml --seeds {SEEDS}
```

### Cell 6: Latency Benchmark

```python
!mkdir -p results
!python scripts/benchmark_latency.py --n_samples 500 --device cuda \
    --json results/latency_gpu.json

import json
with open("results/latency_gpu.json") as f:
    lat = json.load(f)
for entry in lat:
    print(f"{entry['mode']:<15} {entry['per_sample_ms_mean']:>8.3f} ms/sample  "
          f"({entry['throughput_samples_per_s']:>8.1f} samples/s)")
```

### Cell 7: Generate Plots + Tables

```python
!mkdir -p figures results

# Alignment phase plot
!python scripts/plot_alignment.py \
    --logdir outputs_train-multi-digit_multiseed/seed_42 \
    --outdir figures/ 2>/dev/null || echo "Skipped (no Lagrangian logs)"

# CEGIS convergence plot
!python scripts/plot_alignment.py \
    --logdir outputs_train-cegis_multiseed/seed_42 \
    --cegis --outdir figures/ 2>/dev/null || echo "Skipped (no CEGIS logs)"

# Tables (LaTeX + Markdown)
!python scripts/export_tables.py --task all --format latex --save results/
!python scripts/export_tables.py --task all --format markdown --save results/
```

### Cell 8: Display Results

```python
import json, os, glob

print("=" * 70)
print("  NEURAL CEGIS — FULL RESULTS SUMMARY")
print("=" * 70)

report_files = sorted(glob.glob("outputs_*/report.json")) + \
               sorted(glob.glob("outputs_*/seed_*/report.json"))
for rf in report_files:
    name = rf.replace("/report.json", "").replace("outputs_", "")
    try:
        with open(rf) as f:
            report = json.load(f)
    except Exception:
        continue
    print(f"\n{'─'*50}")
    print(f"  {name}")
    for key in ["iid_test", "comp_test", "hard_test"]:
        if key in report:
            m = report[key]
            acc = m.get("sum_acc", m.get("accuracy", "N/A"))
            csr = m.get("csr", "N/A")
            print(f"  {key}: acc={acc:.4f}, CSR={csr:.4f}" if isinstance(acc, float) else f"  {key}: {acc}")

# Display tables
for f in glob.glob("results/*.md"):
    print(f"\n{'='*60}\n  {f}\n{'='*60}")
    print(open(f).read())
```

---

## CPU/MPS Fallback

If you do **not** have a CUDA GPU (e.g., running on Mac with MPS):

```python
# Replace --device cuda with --device cpu or --device mps
!python scripts/benchmark_latency.py --n_samples 100 --device cpu \
    --json results/latency_cpu.json
```

All training commands auto-detect the device. On MPS (Apple Silicon), training
works but may be slower. On CPU, reduce `n_samples` in configs for faster runs.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `batch_size` in config YAML to 32 |
| `z3 not found` | `!pip install z3-solver` |
| `matplotlib not found` | `!pip install matplotlib` |
| MPS errors | Add `--device cpu` to training commands |

## Outputs

After a full run, you'll find:

```
outputs_*/seed_*/report.json   — Per-seed training reports
results/latency_*.json         — Latency benchmark data
results/*.tex, results/*.md    — Publication-ready tables
figures/*.png, figures/*.pdf   — Alignment & convergence plots
```

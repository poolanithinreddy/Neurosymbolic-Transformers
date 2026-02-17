"""NST Colab Playbook v4 — Neural CEGIS Full Experiment Suite.

Copy-paste cells for end-to-end execution on Google Colab (T4 GPU).
Runs all experiments for the Neural CEGIS paper:
  - Multi-digit addition (neural, soft, lagrangian, CEGIS)
  - Controlled baselines (random replay, hard mining, same budget)
  - Kinship relational reasoning (neural, lagrangian, CEGIS)
  - Multi-seed runs for error bars
  - Latency benchmark
  - Publication-ready plots and tables

Total estimated runtime: ~3 hours on a T4 GPU (full suite).
Quick mode: ~40 minutes (single seed).
"""

COLAB_CELLS = [

# ============================================================
# Cell 1: Setup
# ============================================================
"""
# Cell 1: Setup (run once)
# ========================
# Estimated time: 2-3 minutes

!git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git nst
%cd nst

!pip install -U pip wheel -q
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -e ".[dev]" -q
!pip install z3-solver -q  # Optional: for hard constraint verification

# Verify
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Run tests
!python -m pytest tests/ -v --tb=short
""",

# ============================================================
# Cell 2: Dataset Statistics
# ============================================================
"""
# Cell 2: Dataset Statistics
# ===========================

# Multi-digit addition (the HARD benchmark)
!python main.py multi-digit-stats

print()

# Kinship
!python main.py kinship-stats
""",

# ============================================================
# Cell 3: Multi-Digit — All Methods (3 seeds)
# ============================================================
"""
# Cell 3: Multi-Digit Addition — All Methods (3 seeds)
# ======================================================
# ~30 min total.  Neural baseline should FAIL on carry splits.
# CEGIS should close the compositional gap.

SEEDS = "42,43,44"

# Pure Neural
!python main.py multi-seed --task train-multi-digit \\
    --config configs/multi_digit_neural.yaml --seeds {SEEDS}

# NST-Lagrangian
!python main.py multi-seed --task train-multi-digit \\
    --config configs/multi_digit_lagrangian.yaml --seeds {SEEDS}

# Neural CEGIS (the core contribution)
!python main.py multi-seed --task train-cegis \\
    --config configs/multi_digit_cegis.yaml --seeds {SEEDS}
""",

# ============================================================
# Cell 4: Controlled Baselines
# ============================================================
"""
# Cell 4: Controlled Baselines (3 seeds)
# ========================================
# These isolate what CEGIS actually contributes.
# If CEGIS beats all three, the improvement is from
# constraint-targeted counterexamples, not extra data.

SEEDS = "42,43,44"

# Random Replay — same data budget, random samples
!python main.py baseline --method random-replay \\
    --config configs/multi_digit_random_replay.yaml --seeds {SEEDS}

# Hard Example Mining — highest-loss samples, not constraint violations
!python main.py baseline --method hard-mining \\
    --config configs/multi_digit_hard_mining.yaml --seeds {SEEDS}

# Same Budget — train K×E epochs with Lagrangian only
!python main.py baseline --method same-budget \\
    --config configs/multi_digit_same_budget.yaml --seeds {SEEDS}
""",

# ============================================================
# Cell 5: Kinship — All Methods (3 seeds)
# ============================================================
"""
# Cell 5: Kinship Relational Reasoning — All Methods (3 seeds)
# ==============================================================
# Train depth 1-3, test depth 4-6. With distractors + balanced labels.

SEEDS = "42,43,44"

# Pure Neural
!python main.py multi-seed --task train-kinship \\
    --config configs/kinship_neural.yaml --seeds {SEEDS}

# NST-Lagrangian
!python main.py multi-seed --task train-kinship \\
    --config configs/kinship_lagrangian.yaml --seeds {SEEDS}

# Neural CEGIS
!python main.py multi-seed --task train-kinship-cegis \\
    --config configs/kinship_cegis.yaml --seeds {SEEDS}
""",

# ============================================================
# Cell 6: CEGIS Convergence Analysis
# ============================================================
"""
# Cell 6: CEGIS Convergence Analysis
# ====================================
# Print the counterexample trajectory — the signature CEGIS result.

import json, os, glob

cegis_dirs = sorted(glob.glob("outputs_train-cegis_multiseed/seed_*/"))
for d in cegis_dirs:
    log_path = os.path.join(d, "cegis_log.json")
    if not os.path.exists(log_path):
        continue
    with open(log_path) as f:
        log = json.load(f)

    seed = d.rstrip("/").split("_")[-1]
    print(f"\\nSeed {seed}: Converged={log['converged']}, "
          f"Rounds={len(log['rounds'])}")
    for r in log["rounds"]:
        print(f"  Round {r['round']}: CE={r['n_counterexamples']:>4d}, "
              f"λ={r['lambda']:.4f}, CSR={r['csr']:.4f}")
""",

# ============================================================
# Cell 7: Latency Benchmark
# ============================================================
"""
# Cell 7: Inference Latency Benchmark
# =====================================

!mkdir -p results
!python scripts/benchmark_latency.py --n_samples 500 --device cuda \\
    --json results/latency_gpu.json

import json
with open("results/latency_gpu.json") as f:
    lat = json.load(f)

print(f"\\n{'Mode':<15} {'ms/sample':>10} {'throughput':>12}")
print("-" * 40)
for entry in lat:
    print(f\"{entry['mode']:<15} {entry['per_sample_ms_mean']:>10.2f} \"
          f\"{entry['throughput_samples_per_s']:>12.1f}\")
""",

# ============================================================
# Cell 8: Generate Plots + Tables
# ============================================================
"""
# Cell 8: Plots + Tables
# ========================

!mkdir -p figures results

# Alignment phase plot (Lagrangian)
!python scripts/plot_alignment.py \\
    --logdir outputs_train-multi-digit_multiseed/seed_42 \\
    --outdir figures/ 2>/dev/null || echo "Skipped (no logs)"

# CEGIS convergence plot
!python scripts/plot_alignment.py \\
    --logdir outputs_train-cegis_multiseed/seed_42 \\
    --cegis --outdir figures/ 2>/dev/null || echo "Skipped (no logs)"

# Export tables
!python scripts/export_tables.py --task multi_digit --format markdown \\
    --outdir results/ 2>/dev/null || echo "Skipped (no data)"
!python scripts/export_tables.py --task kinship --format markdown \\
    --outdir results/ 2>/dev/null || echo "Skipped (no data)"

# Display table if generated
import os
for f in ["results/multi_digit_results.md", "results/kinship_results.md"]:
    if os.path.exists(f):
        print(f"\\n{'='*60}")
        print(f"  {f}")
        print(f"{'='*60}")
        print(open(f).read())
""",

# ============================================================
# Cell 9: Full Results Summary
# ============================================================
"""
# Cell 9: Results Summary
# =========================

import json, os, glob

print("=" * 70)
print("  NEURAL CEGIS — FULL RESULTS SUMMARY")
print("=" * 70)

# Collect all reports
report_files = sorted(glob.glob("outputs_*/report.json")) + \\
               sorted(glob.glob("outputs_*/seed_*/report.json"))
for rf in report_files:
    name = rf.replace("/report.json", "").replace("outputs_", "")
    try:
        with open(rf) as f:
            report = json.load(f)
    except Exception:
        continue

    print(f"\\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")

    for key in ["iid_test", "comp_test", "hard_test"]:
        if key in report:
            m = report[key]
            acc = m.get("sum_acc", m.get("accuracy", "N/A"))
            csr = m.get("csr", "N/A")
            if isinstance(acc, float):
                acc = f"{acc:.4f}"
            if isinstance(csr, float):
                csr = f"{csr:.4f}"
            print(f"  {key}: acc={acc}, CSR={csr}")

    if "cegis" in report:
        c = report["cegis"]
        print(f"  CEGIS converged: {c.get('converged', 'N/A')}, "
              f"rounds: {c.get('total_rounds', 'N/A')}")

print(f"\\n{'='*70}")
print("  Expected: Neural CEGIS >> Lagrangian >> Soft >> Neural")
print("  on comp_test and hard_test (carry generalisation)")
print(f"{'='*70}")
""",

]  # end COLAB_CELLS


def print_playbook():
    """Print all cells for copy-paste into Colab."""
    for i, cell in enumerate(COLAB_CELLS, 1):
        print(f"\n{'#' * 60}")
        print(f"# CELL {i}")
        print(f"{'#' * 60}")
        print(cell.strip())
        print()


if __name__ == "__main__":
    print_playbook()

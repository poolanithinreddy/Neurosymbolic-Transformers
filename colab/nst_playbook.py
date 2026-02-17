"""NST Colab Playbook v3 — Neural CEGIS Full Experiment Suite.

Copy-paste cells for end-to-end execution on Google Colab (T4 GPU).
Runs all ablations: single-digit, multi-digit (neural/soft/lagrangian/CEGIS),
and kinship relational reasoning.

Total estimated runtime: ~40 minutes on a T4 GPU.
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

# Single-digit addition
!python main.py data-stats --threshold 9

print()

# Multi-digit addition (the HARD benchmark)
!python main.py multi-digit-stats

print()

# Kinship
!python main.py kinship-stats
""",

# ============================================================
# Cell 3: Single-Digit Ablation (baseline sanity check)
# ============================================================
"""
# Cell 3: Single-Digit Addition Ablation
# ========================================
# ~5 min total — proves ALL methods reach 100% on easy benchmark

import yaml, os

modes = ["neural", "soft", "lagrangian"]
for mode in modes:
    cfg = {
        "model": {"mode": mode, "constraint_mode": mode},
        "data": {"n_train": 3000, "n_test": 500, "threshold": 9, "seed": 42},
        "training": {
            "epochs": 15,
            "lr": 0.001,
            "batch_size": 64,
            "device": "auto",
            "seed": 42,
            "lambda_constraint": 0.5,
            "cagrad": False,
            "outdir": f"outputs_digit_add_{mode}",
        },
    }
    if mode == "lagrangian":
        cfg["training"]["lagrangian_epsilon"] = 0.05
        cfg["training"]["lagrangian_alpha"] = 0.01

    cfg_path = f"configs/_colab_digit_{mode}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"\\n{'='*50}\\n  Training: {mode}\\n{'='*50}")
    !python main.py train --config {cfg_path}
""",

# ============================================================
# Cell 4: Multi-Digit Neural Baseline
# ============================================================
"""
# Cell 4: Multi-Digit Addition — Neural Baseline
# =================================================
# The neural baseline should FAIL on carry-propagation (comp/hard splits).

!python main.py train-multi-digit --config configs/multi_digit_neural.yaml
""",

# ============================================================
# Cell 5: Multi-Digit Soft Constraint
# ============================================================
"""
# Cell 5: Multi-Digit Addition — Soft Constraint
# ==================================================

!python main.py train-multi-digit --config configs/multi_digit_soft.yaml
""",

# ============================================================
# Cell 6: Multi-Digit Lagrangian
# ============================================================
"""
# Cell 6: Multi-Digit Addition — Lagrangian (Adaptive λ)
# ========================================================

!python main.py train-multi-digit --config configs/multi_digit_lagrangian.yaml
""",

# ============================================================
# Cell 7: Multi-Digit CEGIS (The Core Contribution)
# ============================================================
"""
# Cell 7: Multi-Digit Addition — Neural CEGIS 🎯
# ==================================================
# THIS IS THE MAIN EXPERIMENT. CEGIS should outperform all baselines
# on the compositional (carry) split.

!python main.py train-cegis --config configs/multi_digit_cegis.yaml

# Print convergence trajectory
import json
with open("outputs_multi_digit_cegis/cegis_log.json") as f:
    log = json.load(f)

print(f"\\nConverged: {log['converged']}")
print(f"Rounds: {len(log['rounds'])}")
print(f"Total counterexamples: {log['total_counterexamples']}")
print("\\nCE trajectory:")
for r in log["rounds"]:
    print(f"  Round {r['round']}: CE={r['n_counterexamples']}, "
          f"λ={r['lambda']:.4f}, CSR={r['csr']:.4f}")
""",

# ============================================================
# Cell 8: Kinship Relational Reasoning
# ============================================================
"""
# Cell 8: Kinship with Distractors + Corruption
# =================================================
# Tests compositional generalisation: train depth 1-3, test depth 4-6
# With 2 distractors + 10% label corruption + balanced labels.

!python main.py train-kinship --config configs/kinship_cegis.yaml
""",

# ============================================================
# Cell 9: Results Summary + Visualisation
# ============================================================
"""
# Cell 9: Results Summary
# =========================

import json, os, glob

print("=" * 70)
print("  NEURAL CEGIS — FULL RESULTS SUMMARY")
print("=" * 70)

# Collect all reports
report_files = glob.glob("outputs_*/report.json")
for rf in sorted(report_files):
    name = os.path.dirname(rf).replace("outputs_", "")
    with open(rf) as f:
        report = json.load(f)

    print(f"\\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")

    for key in ["iid_test", "comp_test", "hard_test"]:
        if key in report:
            m = report[key]
            print(f"  {key}: sum_acc={m.get('sum_acc', 'N/A'):.4f}, "
                  f"CSR={m.get('csr', 'N/A'):.4f}")

    if "cegis" in report:
        c = report["cegis"]
        print(f"  CEGIS converged: {c.get('converged', 'N/A')}, "
              f"rounds: {c.get('total_rounds', 'N/A')}, "
              f"final_λ: {c.get('final_lambda', 'N/A'):.4f}")

print(f"\\n{'='*70}")
print("  Expected result: Neural CEGIS >> Lagrangian >> Soft >> Neural")
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

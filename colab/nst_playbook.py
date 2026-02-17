"""NST Colab Playbook — Copy-paste cells for end-to-end execution.

This file generates the Colab cells as printed output.
Alternatively, copy the blocks below directly into Colab cells.

Total estimated runtime: ~25 minutes on a T4 GPU.
"""

COLAB_CELLS = [

# ============================================================
# Cell 1: Setup
# ============================================================
"""
# Cell 1: Setup (run once)
# ========================
# Estimated time: 2-3 minutes

!git clone https://github.com/<YOUR-USERNAME>/Neurosymbolic-Transformers.git nst
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
""",

# ============================================================
# Cell 2: Dataset Statistics
# ============================================================
"""
# Cell 2: Dataset statistics
# ===========================
# Verify data generation works

!python main.py data-stats --threshold 9

print("\\n--- Kinship dataset ---")
from data.kinship import generate_stats
generate_stats()
""",

# ============================================================
# Cell 3: Smoke Test — Neural Baseline (Digit Addition)
# ============================================================
"""
# Cell 3: Smoke test — Neural baseline (2 epochs)
# =================================================
# Estimated time: ~1 minute

import yaml, os

# Create a quick smoke config
smoke_cfg = {
    'seed': 42, 'device': 'auto', 'mode': 'neural',
    'data': {'n_train': 500, 'n_test': 200, 'comp_threshold': 9},
    'train': {'epochs': 2, 'batch_size': 64, 'lr': 0.001, 'warmup_steps': 50},
    'logic': {'lambda': 0.0},
    'io': {'out_dir': 'outputs_smoke_neural'},
}
os.makedirs('configs', exist_ok=True)
with open('configs/smoke_neural.yaml', 'w') as f:
    yaml.dump(smoke_cfg, f)

!python main.py train --config configs/smoke_neural.yaml
print("✅ Smoke test passed!")
""",

# ============================================================
# Cell 4: Full Training — All Digit-Addition Ablations
# ============================================================
"""
# Cell 4: Full digit-addition ablation
# ======================================
# Trains: Neural, Soft (fixed-λ), Lagrangian, Hard (Lagrangian+Z3)
# Estimated time: ~10 minutes total on T4

import time
start = time.time()

# Neural baseline
!python main.py train --config configs/digit_add_neural.yaml
print(f"\\n⏱ Neural done ({time.time()-start:.0f}s)")

# Fixed-λ soft constraints
!python main.py train --config configs/digit_add_soft.yaml
print(f"⏱ Soft done ({time.time()-start:.0f}s)")

# Lagrangian adaptive constraints (CORE contribution)
!python main.py train --config configs/digit_add_lagrangian.yaml
print(f"⏱ Lagrangian done ({time.time()-start:.0f}s)")

# CAGrad gradient balancing
!python main.py train --config configs/digit_add_cagrad.yaml
print(f"⏱ CAGrad done ({time.time()-start:.0f}s)")

print(f"\\n✅ All digit-add ablations complete ({time.time()-start:.0f}s)")
""",

# ============================================================
# Cell 5: Evaluate Digit-Addition Models
# ============================================================
"""
# Cell 5: Evaluate digit-addition models
# ========================================

import os

models = {
    'Neural': 'outputs_digit_add_neural/ckpt/best_model.pt',
    'Soft': 'outputs_digit_add_soft/ckpt/best_model.pt',
    'Lagrangian': 'outputs_digit_add_lagrangian/ckpt/best_model.pt',
    'CAGrad': 'outputs_digit_add_cagrad/ckpt/best_model.pt',
}

for name, ckpt in models.items():
    if os.path.exists(ckpt):
        print(f"\\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        !python main.py eval --ckpt {ckpt} --n_test 2000

# Lagrangian with Z3 hard inference
lagr_ckpt = 'outputs_digit_add_lagrangian/ckpt/best_model.pt'
if os.path.exists(lagr_ckpt):
    print(f"\\n{'='*50}")
    print(f"  Lagrangian + Z3 Hard")
    print(f"{'='*50}")
    !python main.py eval --ckpt {lagr_ckpt} --hard --n_test 2000
""",

# ============================================================
# Cell 6: Kinship Training
# ============================================================
"""
# Cell 6: Kinship relational reasoning
# ======================================
# Estimated time: ~8 minutes

import time
start = time.time()

!python main.py train-kinship --config configs/kinship_neural.yaml
print(f"⏱ Kinship Neural done ({time.time()-start:.0f}s)")

!python main.py train-kinship --config configs/kinship_soft.yaml
print(f"⏱ Kinship Soft done ({time.time()-start:.0f}s)")

!python main.py train-kinship --config configs/kinship_lagrangian.yaml
print(f"⏱ Kinship Lagrangian done ({time.time()-start:.0f}s)")

print(f"\\n✅ All kinship experiments complete ({time.time()-start:.0f}s)")
""",

# ============================================================
# Cell 7: λ* Trajectory Plot
# ============================================================
"""
# Cell 7: Plot λ* trajectory (Lagrangian dual variable)
# =======================================================

import json
import matplotlib.pyplot as plt

for task, path in [
    ('Digit-Add', 'outputs_digit_add_lagrangian/lambda_trajectory.json'),
    ('Kinship', 'outputs_kinship_lagrangian/lambda_trajectory.json'),
]:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)

    steps = [t['step'] for t in data['trajectory']]
    lambdas = [t['lambda'] for t in data['trajectory']]
    logic_losses = [t['loss_logic'] for t in data['trajectory']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, lambdas, 'b-o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('λ (dual variable)')
    ax1.set_title(f'{task}: λ* Trajectory')
    ax1.axhline(y=data['epsilon'], color='r', linestyle='--', label=f'ε={data["epsilon"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, logic_losses, 'g-o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L_logic')
    ax2.set_title(f'{task}: Constraint Loss')
    ax2.axhline(y=data['epsilon'], color='r', linestyle='--', label=f'ε={data["epsilon"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{task.lower().replace("-","_")}_lambda_trajectory.png', dpi=150)
    plt.show()
    print(f"Final λ* = {data['final_lambda']:.4f} (Price of Logic)")
""",

# ============================================================
# Cell 8: Generate Results Tables
# ============================================================
"""
# Cell 8: Generate LaTeX + Markdown results tables
# ===================================================

from results import render_results, results_to_latex
import json, os, glob

# Collect all reports
reports = {}
for d in sorted(glob.glob('outputs_*/final_report.json')):
    with open(d) as f:
        data = json.load(f)
    name = os.path.basename(os.path.dirname(d))
    reports[name] = data

# Print markdown
from results import results_to_markdown
print("## Digit-Addition Results")
digit_reports = {k: v for k, v in reports.items() if 'digit' in k}
print(results_to_markdown(digit_reports, task='digit_add'))

print("\\n## Kinship Results")
kinship_reports = {k: v for k, v in reports.items() if 'kinship' in k}
if kinship_reports:
    print(results_to_markdown(kinship_reports, task='kinship'))

# Save LaTeX
if digit_reports:
    latex = results_to_latex(digit_reports, task='digit_add',
                             caption='Digit addition results', label='tab:digit')
    with open('results_digit_add.tex', 'w') as f:
        f.write(latex)
    print("\\n✅ LaTeX table saved to results_digit_add.tex")
""",

# ============================================================
# Cell 9: Run Tests
# ============================================================
"""
# Cell 9: Run test suite
# ========================
!python -m pytest tests/ -v --tb=short
""",

]


def print_playbook():
    """Print the Colab playbook as numbered cells."""
    for i, cell in enumerate(COLAB_CELLS, 1):
        print(f"\n{'#' * 60}")
        print(f"# CELL {i}")
        print(f"{'#' * 60}")
        print(cell.strip())
        print()


if __name__ == "__main__":
    print_playbook()

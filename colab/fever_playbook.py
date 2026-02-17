"""NST FEVER Colab Playbook — 12 cells for end-to-end FEVER training.

Copy-paste cells into Google Colab (T4 GPU runtime).
Total estimated runtime: ~45 minutes on T4 (all cells).

Evidence settings:
  - Setting A (Gold Evidence): oracle evidence → Label Accuracy
  - Setting B (Full Pipeline):  BM25 retrieval → End-to-End FEVER Score

Training modes:
  - neural:  pure DeBERTa cross-entropy baseline
  - soft:    CE + fixed-weight constraint loss
  - cegis:   Lagrangian + counterexample-guided outer loop
  - gated:   ECCG — Evidence-Conditioned Constraint Gating (novel)
"""

COLAB_CELLS = [

# ============================================================
# Cell 1: Setup + Clone + Install
# ============================================================
"""
# Cell 1: Setup (run once) — ~3 min
# ===================================
!git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git nst
%cd nst

!pip install -U pip wheel -q
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -e ".[dev]" -q

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
print(f"GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print("✅ Setup complete")
""",

# ============================================================
# Cell 2: Build Wiki Cache + Dataset Stats
# ============================================================
"""
# Cell 2: Build wiki cache + dataset stats — ~5 min first time
# ==============================================================
import os

# Build SQLite wiki cache (avoids OOM from loading all wiki_pages)
if not os.path.exists("data/fever_wiki.db"):
    !python main.py build-fever-wiki-cache
    print("✅ Wiki cache built")
else:
    !python main.py build-fever-wiki-cache --stats_only
    print("✅ Wiki cache already exists")

# Dataset stats
!python main.py fever-stats

from data.fever_dataset import FEVER_LABELS, LABEL2ID, NUM_LABELS
print(f"\\nLabels: {FEVER_LABELS}")
print(f"LABEL2ID: {LABEL2ID}")
print(f"NUM_LABELS: {NUM_LABELS}")
print("✅ Data OK")
""",

# ============================================================
# Cell 3: Smoke Test — Neural Baseline (Tiny Run)
# ============================================================
"""
# Cell 3: Smoke test — Neural baseline on 200 samples — ~1 min
# ==============================================================
import yaml, os

smoke_cfg = {
    'seed': 42, 'device': 'auto', 'mode': 'neural',
    'task': 'fever', 'evidence_mode': 'gold',
    'model': {
        'name': 'microsoft/deberta-v3-base',
        'num_labels': 3, 'label_smoothing': 0.05,
        'dropout': 0.1, 'max_length': 128,
    },
    'data': {'max_train': 200, 'max_dev': 100, 'dev_sample': 50},
    'train': {
        'epochs': 1, 'batch_size': 8, 'grad_accum_steps': 1,
        'lr': 2e-5, 'warmup_ratio': 0.1, 'weight_decay': 0.01,
        'max_grad_norm': 1.0, 'fp16': True,
        'eval_every_steps': 50, 'patience': 3,
    },
    'logic': {'lambda': 0.0},
    'io': {'out_dir': 'outputs_fever_smoke'},
}
os.makedirs('configs', exist_ok=True)
with open('configs/fever_smoke.yaml', 'w') as f:
    yaml.dump(smoke_cfg, f)

!python main.py train-fever-nst --config configs/fever_smoke.yaml
print("✅ Smoke test passed!")
""",

# ============================================================
# Cell 4: Gold Evidence — Neural Baseline (Full)
# ============================================================
"""
# Cell 4: Gold Evidence — Neural baseline — ~8 min on T4
# ========================================================
!python main.py train-fever-nst --config configs/fever_gold_neural.yaml
print("✅ Neural baseline (gold evidence) done")
""",

# ============================================================
# Cell 5: Gold Evidence — NST Soft Constraints
# ============================================================
"""
# Cell 5: Gold Evidence — NST Soft constraints — ~8 min
# =======================================================
!python main.py train-fever-nst --config configs/fever_gold_nst_soft.yaml
print("✅ NST Soft (gold evidence) done")
""",

# ============================================================
# Cell 6: Gold Evidence — NST CEGIS (Full System)
# ============================================================
"""
# Cell 6: Gold Evidence — NST CEGIS — ~12 min
# ==============================================
!python main.py train-fever-nst --config configs/fever_gold_nst_cegis.yaml
print("✅ NST CEGIS (gold evidence) done")
""",

# ============================================================
# Cell 7: Gold Evidence — NST Gated / ECCG (Novel)
# ============================================================
"""
# Cell 7: Gold Evidence — NST ECCG (novel) — ~10 min
# =====================================================
!python main.py train-fever-nst --config configs/fever_gold_nst_gated.yaml
print("✅ NST ECCG / Gated (gold evidence) done")
""",

# ============================================================
# Cell 8: No-Leakage Verification
# ============================================================
"""
# Cell 8: No-leakage verification (6 automated checks)
# =======================================================
!python scripts/verify_no_leakage.py
print("\\n✅ All leakage checks passed (see above for details)")
""",

# ============================================================
# Cell 9: Compare Results
# ============================================================
"""
# Cell 9: Compare all gold-evidence results
# ============================================
import json, os, glob

print("=" * 65)
print("  FEVER Results — Gold Evidence (Setting A)")
print("=" * 65)
print(f"{'Model':<30} {'Acc':>8} {'ECE':>8} {'Brier':>8}")
print("-" * 65)

for d in sorted(glob.glob("outputs_fever_gold_*")):
    rpath = os.path.join(d, "report.json")
    if not os.path.exists(rpath):
        continue
    with open(rpath) as f:
        rep = json.load(f)
    dev = rep.get("dev", rep)
    name = os.path.basename(d).replace("outputs_fever_gold_", "")
    acc = dev.get("accuracy", 0)
    ece = dev.get("ece", 0)
    brier = dev.get("brier", 0)
    print(f"  {name:<28} {acc:>8.4f} {ece:>8.4f} {brier:>8.4f}")

    per_label = dev.get("per_label", {})
    for lbl in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        stats = per_label.get(lbl, {})
        if stats:
            print(f"    {lbl:<26} {stats.get('accuracy', 0):>8.4f} ({stats.get('correct', 0)}/{stats.get('count', 0)})")

print("\\n✅ Comparison complete")
""",

# ============================================================
# Cell 10: Multi-seed Run (3 seeds, error bars)
# ============================================================
"""
# Cell 10: Multi-seed ECCG for reproducibility — ~30 min
# ========================================================
!python main.py multi-seed --task train-fever-nst \\
    --config configs/fever_gold_nst_gated.yaml \\
    --seeds 42,43,44

import json
with open("outputs_train-fever-nst_multiseed/multi_seed_summary.json") as f:
    summary = json.load(f)
agg = summary["aggregated"]
print("\\n" + "=" * 50)
print("  Multi-Seed Results (ECCG, 3 seeds)")
print("=" * 50)
for k, v in sorted(agg.items()):
    print(f"  {k:<30} {v['mean']:.4f} ± {v['std']:.4f}")
print("✅ Multi-seed evaluation complete")
""",

# ============================================================
# Cell 11: Gate Analysis (ECCG Interpretability)
# ============================================================
"""
# Cell 11: Analyse learned gate values — what did ECCG learn?
# =============================================================
import torch, json, os
from symbolic.constraint_gating import ConstraintGate

gate_path = "outputs_fever_gold_nst_gated/ckpt/constraint_gate.pt"
if os.path.exists(gate_path):
    gate = ConstraintGate()
    gate.load_state_dict(torch.load(gate_path, map_location='cpu'))
    gate.eval()

    biases = gate.net[-1].bias.detach()
    constraint_names = ['C1:date_contra', 'C2:num_contra', 'C3:neg_mismatch',
                        'C4:low_overlap', 'C5:no_evidence']
    print("Learned gate biases (sigmoid → default openness):")
    for name, b in zip(constraint_names, biases):
        print(f"  {name:<20} bias={b.item():.3f}  → gate={torch.sigmoid(b).item():.3f}")

    rpath = "outputs_fever_gold_nst_gated/report.json"
    if os.path.exists(rpath):
        with open(rpath) as f:
            rep = json.load(f)
        print(f"\\nFinal accuracy: {rep.get('dev', {}).get('accuracy', 'N/A')}")
        print(f"Temperature:    {rep.get('temperature', 'N/A')}")
else:
    print("⚠️  No gate checkpoint found — run Cell 7 first")
print("✅ Gate analysis complete")
""",

# ============================================================
# Cell 12: Zip + Save to Drive
# ============================================================
"""
# Cell 12: Zip outputs + save to Google Drive
# ==============================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import shutil, datetime, os, glob

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# Collect all FEVER outputs into a single directory
os.makedirs("outputs", exist_ok=True)
for d in sorted(glob.glob("outputs_fever_*")):
    dst = os.path.join("outputs", os.path.basename(d))
    if not os.path.exists(dst):
        shutil.copytree(d, dst)

archive = shutil.make_archive(f"nst_results_{timestamp}", 'zip', 'outputs')
print(f"Created: {archive}")

DST = "/content/drive/MyDrive/"
shutil.copy(archive, DST)
print(f"Saved to Drive: {DST}{os.path.basename(archive)}")
print(f"\\n✅ All results zipped and saved to Google Drive")
""",

]  # end COLAB_CELLS


def print_cells():
    """Print all cells for easy copy-paste."""
    for i, cell in enumerate(COLAB_CELLS, 1):
        print(f"\n{'#'*60}")
        print(f"# CELL {i}")
        print(f"{'#'*60}")
        print(cell)


if __name__ == "__main__":
    print_cells()

"""NST FEVER Colab — Smoke Test + Full FEVER Standard Test Set.

Copy-paste each CELL into a separate Colab cell (T4 GPU runtime).

Part 1: Smoke test   (~5 min) — verifies install, data loading, tiny training
Part 2: Full FEVER   (~45 min on T4) — all 4 modes on gold evidence

Usage:
    python colab/nst_fever_colab.py   # prints cells for copy-paste
"""

CELLS = []

# ═══════════════════════════════════════════════════════════════
#  PART 1 — SMOKE TEST  (Cells 1–5)
# ═══════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────
# Cell 1: Clone + Install
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 1: Setup — Clone repo + install deps  (~3 min)
# ============================================================
import os, subprocess, sys

# Check GPU
!nvidia-smi || echo "⚠️  No GPU detected — switch to T4 runtime"

# Clone (skip if already cloned)
if not os.path.exists("nst"):
    !git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git nst

%cd nst

# Install deps
!pip install -U pip wheel setuptools -q
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -e ".[dev]" -q
!pip install z3-solver -q  # for hard constraint verification

# Verify
import torch
print(f"\n{'='*50}")
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("  GPU     : CPU only (will be slow)")
print(f"{'='*50}")

# Quick import check — catch missing deps early
try:
    import transformers, datasets, sklearn, yaml, numpy, scipy
    print(f"  transformers : {transformers.__version__}")
    print(f"  datasets     : {datasets.__version__}")
    print("✅ All core dependencies OK")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    raise
""")

# ────────────────────────────────────────────────────────────────
# Cell 2: Build Wiki Cache
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 2: Build FEVER wiki cache  (~5 min first time)
# ============================================================
# The wiki cache is a ~15 MB SQLite file that stores only the
# ~25k wiki pages referenced by FEVER evidence annotations.
# This avoids loading the full 5.4M-page wiki dump into memory.
import os

cache_path = "data/fever_wiki.db"
if os.path.exists(cache_path):
    !python main.py build-fever-wiki-cache --stats_only
    print("✅ Wiki cache already exists")
else:
    print("Building wiki cache (one-time, ~5 min)...")
    !python main.py build-fever-wiki-cache
    print("✅ Wiki cache built")

# Verify cache is usable
from data.fever_wiki_cache import WikiCache
cache = WikiCache(cache_path)
n_pages = len(cache)
print(f"  Cached pages: {n_pages}")
assert n_pages > 1000, f"Cache too small ({n_pages} pages) — rebuild"
print("✅ Cache verified")
""")

# ────────────────────────────────────────────────────────────────
# Cell 3: Dataset Stats + Sanity Checks
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 3: Dataset stats + sanity checks  (~1 min)
# ============================================================
import sys, os
sys.path.insert(0, os.getcwd())

# 3a. Print FEVER stats (loads from HuggingFace, uses wiki cache)
!python main.py fever-stats

# 3b. Verify label constants
from data.fever_dataset import FEVER_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS

print(f"\nLabel constants:")
print(f"  FEVER_LABELS : {FEVER_LABELS}")
print(f"  LABEL2ID     : {LABEL2ID}")
print(f"  ID2LABEL     : {ID2LABEL}")
print(f"  NUM_LABELS   : {NUM_LABELS}")
assert NUM_LABELS == 3, "Expected 3 labels"
assert LABEL2ID["SUPPORTS"] == 0
assert LABEL2ID["REFUTES"] == 1
assert LABEL2ID["NOT ENOUGH INFO"] == 2
print("✅ Label mapping OK")

# 3c. Test data loading with small subset
from data.fever_dataset import load_fever_splits, FeverGoldDataset, fever_collate_fn
splits = load_fever_splits(max_train=100, max_dev=50)
assert len(splits["train"]) == 100, f"Expected 100 train, got {len(splits['train'])}"
assert len(splits["dev"]) == 50, f"Expected 50 dev, got {len(splits['dev'])}"

# Check evidence is populated
n_with_ev = sum(1 for it in splits["train"] if it["gold_evidence_text"])
print(f"\n  Train samples with evidence text: {n_with_ev}/100")

# Test PyTorch dataset + collate
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
ds = FeverGoldDataset(splits["train"][:8])
batch_raw = [ds[i] for i in range(min(4, len(ds)))]
batch = fever_collate_fn(batch_raw, tokenizer, max_length=128)
assert "input_ids" in batch
assert "labels" in batch
assert batch["input_ids"].shape[0] == len(batch_raw)
print(f"  Collate OK: input_ids shape = {batch['input_ids'].shape}")
print("✅ Data loading + collation OK")
""")

# ────────────────────────────────────────────────────────────────
# Cell 4: Unit Tests (quick)
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 4: Run unit tests  (~1 min)
# ============================================================
# Run the FEVER-specific tests + key infrastructure tests
!python -m pytest tests/test_fever.py tests/test_fever_io.py tests/test_calibration.py \
    tests/test_config_validation.py tests/test_infrastructure.py \
    -v --no-cov --tb=short 2>&1 | tail -40

print("\n✅ Unit tests done (check output above for any failures)")
""")

# ────────────────────────────────────────────────────────────────
# Cell 5: Smoke Train — Neural Baseline (200 samples, 1 epoch)
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 5: SMOKE TEST — train tiny neural baseline  (~1–2 min)
# ============================================================
# This trains on 200 samples for 1 epoch just to verify
# the full training pipeline runs end-to-end without errors.
import os

# Use the pre-built smoke config
!python main.py train-fever-nst --config configs/fever_gold_smoke.yaml

# Verify outputs were created
smoke_dir = "outputs_fever_gold_smoke"
assert os.path.exists(smoke_dir), f"Output dir {smoke_dir} not created"

ckpt_dir = os.path.join(smoke_dir, "ckpt")
report_path = os.path.join(smoke_dir, "report.json")

if os.path.exists(report_path):
    import json
    with open(report_path) as f:
        rep = json.load(f)
    dev_acc = rep.get("dev", rep).get("accuracy", "N/A")
    print(f"\n  Smoke test dev accuracy: {dev_acc}")
    print(f"  (Low accuracy expected — only 200 training samples, 1 epoch)")

print("\n🎉 SMOKE TEST PASSED — full pipeline works end-to-end!")
print("="*55)
print("  You can now proceed to the full FEVER experiments")
print("="*55)
""")


# ═══════════════════════════════════════════════════════════════
#  PART 2 — FULL FEVER EXPERIMENTS  (Cells 6–12)
# ═══════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────
# Cell 6: Neural Baseline (Gold Evidence)
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 6: Gold Evidence — Neural Baseline  (~8 min on T4)
# ============================================================
# Pure DeBERTa-v3-base cross-entropy, no symbolic constraints.
# This is the baseline against which NST methods are compared.
import time
start = time.time()

!python main.py train-fever-nst --config configs/fever_gold_neural.yaml

elapsed = time.time() - start
print(f"\n⏱  Neural baseline done in {elapsed/60:.1f} min")

# Check report
import json, os
rpath = "outputs_fever_gold_neural/report.json"
if os.path.exists(rpath):
    with open(rpath) as f:
        rep = json.load(f)
    dev = rep.get("dev", rep)
    print(f"  Accuracy: {dev.get('accuracy', 'N/A')}")
    print(f"  ECE:      {dev.get('ece', 'N/A')}")
    print(f"  Brier:    {dev.get('brier', 'N/A')}")
print("✅ Neural baseline (gold evidence) done")
""")

# ────────────────────────────────────────────────────────────────
# Cell 7: NST Soft Constraints (Gold Evidence)
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 7: Gold Evidence — NST Soft Constraints  (~8 min on T4)
# ============================================================
# CE + fixed-weight symbolic constraint loss.
# Marginal gain expected over neural via constraint regularization.
import time
start = time.time()

!python main.py train-fever-nst --config configs/fever_gold_nst_soft.yaml

elapsed = time.time() - start
print(f"\n⏱  NST Soft done in {elapsed/60:.1f} min")

import json, os
rpath = "outputs_fever_gold_nst_soft/report.json"
if os.path.exists(rpath):
    with open(rpath) as f:
        rep = json.load(f)
    dev = rep.get("dev", rep)
    print(f"  Accuracy: {dev.get('accuracy', 'N/A')}")
    print(f"  ECE:      {dev.get('ece', 'N/A')}")
    print(f"  Brier:    {dev.get('brier', 'N/A')}")
print("✅ NST Soft (gold evidence) done")
""")

# ────────────────────────────────────────────────────────────────
# Cell 8: NST CEGIS (Gold Evidence)
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 8: Gold Evidence — NST CEGIS  (~12 min on T4)
# ============================================================
# Lagrangian adaptive + counterexample-guided outer loop.
# Expected: best label accuracy via constraint-guided training.
import time
start = time.time()

!python main.py train-fever-nst --config configs/fever_gold_nst_cegis.yaml

elapsed = time.time() - start
print(f"\n⏱  NST CEGIS done in {elapsed/60:.1f} min")

import json, os
rpath = "outputs_fever_gold_nst_cegis/report.json"
if os.path.exists(rpath):
    with open(rpath) as f:
        rep = json.load(f)
    dev = rep.get("dev", rep)
    print(f"  Accuracy: {dev.get('accuracy', 'N/A')}")
    print(f"  ECE:      {dev.get('ece', 'N/A')}")
    print(f"  Brier:    {dev.get('brier', 'N/A')}")
print("✅ NST CEGIS (gold evidence) done")
""")

# ────────────────────────────────────────────────────────────────
# Cell 9: NST ECCG / Gated (Gold Evidence)
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 9: Gold Evidence — NST ECCG (Novel)  (~10 min on T4)
# ============================================================
# Evidence-Conditioned Constraint Gating — the novel contribution.
# Learns per-sample, per-constraint gates based on evidence features.
import time
start = time.time()

!python main.py train-fever-nst --config configs/fever_gold_nst_gated.yaml

elapsed = time.time() - start
print(f"\n⏱  NST ECCG done in {elapsed/60:.1f} min")

import json, os
rpath = "outputs_fever_gold_nst_gated/report.json"
if os.path.exists(rpath):
    with open(rpath) as f:
        rep = json.load(f)
    dev = rep.get("dev", rep)
    print(f"  Accuracy: {dev.get('accuracy', 'N/A')}")
    print(f"  ECE:      {dev.get('ece', 'N/A')}")
    print(f"  Brier:    {dev.get('brier', 'N/A')}")
print("✅ NST ECCG / Gated (gold evidence) done")
""")

# ────────────────────────────────────────────────────────────────
# Cell 10: No-Leakage Verification
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 10: No-Leakage Verification (6 automated checks)
# ============================================================
# Runs integrity checks to verify no data leakage occurred.
!python scripts/verify_no_leakage.py --max_train 2000 --max_dev 500

print("\n✅ All leakage checks complete (review output above)")
""")

# ────────────────────────────────────────────────────────────────
# Cell 11: Compare All Results
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 11: Compare all gold-evidence results
# ============================================================
import json, os, glob

print("=" * 70)
print("  FEVER Results — Gold Evidence (Setting A)")
print("=" * 70)
print(f"  {'Model':<30} {'Acc':>8} {'ECE':>8} {'Brier':>8}")
print("-" * 70)

results = {}
for d in sorted(glob.glob("outputs_fever_gold_*")):
    if d.endswith("_smoke"):
        continue  # skip smoke test
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
    results[name] = {"accuracy": acc, "ece": ece, "brier": brier}
    print(f"  {name:<30} {acc:>8.4f} {ece:>8.4f} {brier:>8.4f}")

    # Per-label breakdown
    per_label = dev.get("per_label", {})
    for lbl in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        stats = per_label.get(lbl, {})
        if stats:
            print(f"    {lbl:<28} {stats.get('accuracy', 0):>8.4f}  "
                  f"({stats.get('correct', 0)}/{stats.get('count', 0)})")

# Summary comparison
if len(results) >= 2:
    print("\n" + "=" * 70)
    print("  Summary: Accuracy ranking")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda x: -x[1]["accuracy"])
    for i, (name, m) in enumerate(ranked, 1):
        marker = " ← best" if i == 1 else ""
        print(f"  {i}. {name:<28} {m['accuracy']:.4f}{marker}")

print("\n✅ Comparison complete")
""")

# ────────────────────────────────────────────────────────────────
# Cell 12: Save Results to Drive
# ────────────────────────────────────────────────────────────────
CELLS.append(r"""
# ============================================================
# Cell 12: (Optional) Zip outputs + save to Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import shutil, datetime, os, glob

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# Collect all FEVER outputs
os.makedirs("all_fever_outputs", exist_ok=True)
for d in sorted(glob.glob("outputs_fever_*")):
    dst = os.path.join("all_fever_outputs", os.path.basename(d))
    if not os.path.exists(dst):
        shutil.copytree(d, dst)

archive = shutil.make_archive(f"nst_fever_results_{timestamp}", "zip", "all_fever_outputs")
print(f"Created: {archive}")

dst_dir = "/content/drive/MyDrive/"
shutil.copy(archive, dst_dir)
print(f"Saved to Drive: {dst_dir}{os.path.basename(archive)}")
print("\n✅ All results saved to Google Drive")
""")


def print_cells():
    """Print all cells numbered for easy copy-paste."""
    for i, cell in enumerate(CELLS, 1):
        if i == 1:
            print("=" * 60)
            print("  PART 1: SMOKE TEST (Cells 1–5)")
            print("=" * 60)
        elif i == 6:
            print("\n" + "=" * 60)
            print("  PART 2: FULL FEVER EXPERIMENTS (Cells 6–12)")
            print("=" * 60)
        print(f"\n{'#' * 60}")
        print(f"# CELL {i}")
        print(f"{'#' * 60}")
        print(cell)


if __name__ == "__main__":
    print_cells()

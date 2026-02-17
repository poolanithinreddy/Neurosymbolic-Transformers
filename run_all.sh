#!/usr/bin/env bash
# run_all.sh — Full experiment suite for Neural CEGIS paper.
# Generates all numbers for Tables 1–4 and Figures 1–2.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh              # full suite (~3 hrs on T4)
#   ./run_all.sh --quick      # smoke test (~15 min)
#
# Prerequisites:
#   pip install -e ".[dev]"
#   python -m pytest tests/ -q   # ensure tests pass first

set -euo pipefail

SEEDS="42,43,44"
QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
    SEEDS="42"
    echo "=== QUICK MODE: single seed, reduced epochs ==="
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Neural CEGIS — Full Experiment Suite                   ║"
echo "║  Seeds: ${SEEDS}                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Dataset Statistics ────────────────────────────────────
echo "▸ [1/8] Dataset statistics..."
python main.py multi-digit-stats
python main.py kinship-stats
echo "  ✓ Done"

# ── 2. Multi-Digit Addition — All Methods ────────────────────
echo ""
echo "▸ [2/8] Multi-digit addition experiments..."

echo "  → Pure Neural"
python main.py multi-seed --task train-multi-digit \
    --config configs/multi_digit_neural.yaml --seeds "$SEEDS"

echo "  → NST-Soft"
python main.py multi-seed --task train-multi-digit \
    --config configs/multi_digit_soft.yaml --seeds "$SEEDS"

echo "  → NST-Lagrangian"
python main.py multi-seed --task train-multi-digit \
    --config configs/multi_digit_lagrangian.yaml --seeds "$SEEDS"

echo "  → Neural CEGIS"
python main.py multi-seed --task train-cegis \
    --config configs/multi_digit_cegis.yaml --seeds "$SEEDS"

echo "  ✓ Multi-digit done"

# ── 3. Baselines ─────────────────────────────────────────────
echo ""
echo "▸ [3/8] Controlled baselines..."

echo "  → Random Replay"
python main.py baseline --method random-replay \
    --config configs/multi_digit_random_replay.yaml --seeds "$SEEDS"

echo "  → Hard Example Mining"
python main.py baseline --method hard-mining \
    --config configs/multi_digit_hard_mining.yaml --seeds "$SEEDS"

echo "  → Same Budget"
python main.py baseline --method same-budget \
    --config configs/multi_digit_same_budget.yaml --seeds "$SEEDS"

echo "  ✓ Baselines done"

# ── 4. Kinship — All Methods ─────────────────────────────────
echo ""
echo "▸ [4/8] Kinship experiments..."

echo "  → Pure Neural"
python main.py multi-seed --task train-kinship \
    --config configs/kinship_neural.yaml --seeds "$SEEDS"

echo "  → NST-Lagrangian"
python main.py multi-seed --task train-kinship \
    --config configs/kinship_lagrangian.yaml --seeds "$SEEDS"

echo "  → Neural CEGIS"
python main.py multi-seed --task train-kinship-cegis \
    --config configs/kinship_cegis.yaml --seeds "$SEEDS"

echo "  ✓ Kinship done"

# ── 5. Latency Benchmark ─────────────────────────────────────
echo ""
echo "▸ [5/8] Inference latency benchmark..."
mkdir -p results
python scripts/benchmark_latency.py --n_samples 500 --device cpu \
    --json results/latency_cpu.json
echo "  ✓ Latency done"

# ── 6. Generate Plots ────────────────────────────────────────
echo ""
echo "▸ [6/8] Generating plots..."
# Alignment phase plot (Lagrangian)
if [ -d "outputs_train-multi-digit_multiseed" ]; then
    python scripts/plot_alignment.py \
        --logdir outputs_train-multi-digit_multiseed/seed_42 \
        --outdir figures/ || echo "  ⚠ Lagrangian plot skipped (no logs)"
fi
# CEGIS convergence plot
if [ -d "outputs_train-cegis_multiseed" ]; then
    python scripts/plot_alignment.py \
        --logdir outputs_train-cegis_multiseed/seed_42 \
        --cegis --outdir figures/ || echo "  ⚠ CEGIS plot skipped (no logs)"
fi
echo "  ✓ Plots done"

# ── 7. Export Tables ──────────────────────────────────────────
echo ""
echo "▸ [7/8] Exporting results tables..."
python scripts/export_tables.py --task multi_digit --format latex \
    --outdir results/ 2>/dev/null || echo "  ⚠ Multi-digit table skipped"
python scripts/export_tables.py --task multi_digit --format markdown \
    --outdir results/ 2>/dev/null || echo "  ⚠ Multi-digit table skipped"
python scripts/export_tables.py --task kinship --format latex \
    --outdir results/ 2>/dev/null || echo "  ⚠ Kinship table skipped"
python scripts/export_tables.py --task kinship --format markdown \
    --outdir results/ 2>/dev/null || echo "  ⚠ Kinship table skipped"
echo "  ✓ Tables done"

# ── 8. Test Suite ─────────────────────────────────────────────
echo ""
echo "▸ [8/8] Running test suite..."
python -m pytest tests/ -q --tb=short
echo "  ✓ Tests done"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ All experiments complete!                            ║"
echo "║                                                          ║"
echo "║  Results:    results/*.json, results/*.tex, results/*.md ║"
echo "║  Figures:    figures/*.pdf                                ║"
echo "║  Checkpoints: outputs_*/ckpt/                            ║"
echo "╚══════════════════════════════════════════════════════════╝"

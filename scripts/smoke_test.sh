#!/usr/bin/env bash
# Lightweight smoke test — verifies the project installs and runs correctly.
# No GPU required. Completes in under 60 seconds on any modern CPU.
#
# Usage:
#   bash scripts/smoke_test.sh           # run all checks
#   bash scripts/smoke_test.sh --fast    # skip test suite, run core commands only
#
# Exit code: 0 = all checks passed, non-zero = failure.

set -uo pipefail

FAST_MODE=0
for arg in "$@"; do
  [[ "$arg" == "--fast" ]] && FAST_MODE=1
done

PY="${PYTHON:-python3}"
PASS=0
FAIL=0

ok()  { echo "  [PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo ""
echo "=== NST Smoke Test ==="
echo ""

# ── 1. Python version ───────────────────────────────────────────────────────
echo "1. Environment"
PY_VER=$($PY -c "import sys; ok=sys.version_info >= (3,10); print(sys.version_info.major, sys.version_info.minor, ok)")
PY_MINOR=$(echo "$PY_VER" | awk '{print $2}')
PY_OK=$(echo "$PY_VER" | awk '{print $3}')
PY_LABEL=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PY_OK" == "True" ]]; then
  ok "Python $PY_LABEL"
else
  fail "Python $PY_LABEL (requires >=3.10)"
fi

# ── 2. Core imports ─────────────────────────────────────────────────────────
echo ""
echo "2. Core imports"
$PY -c "import torch" && ok "torch" || fail "torch"
$PY -c "import transformers" && ok "transformers" || fail "transformers"
$PY -c "from models.nst_multi_digit import MultiDigitModel" && ok "models.nst_multi_digit" || fail "models.nst_multi_digit"
$PY -c "from training.cegis import CEGISTrainer, CEGISConfig" && ok "training.cegis" || fail "training.cegis"
$PY -c "from symbolic.lagrangian import lagrangian_loss" && ok "symbolic.lagrangian" || fail "symbolic.lagrangian"
$PY -c "from data.multi_digit_addition import MultiDigitAdditionDataset" && ok "data.multi_digit_addition" || fail "data.multi_digit_addition"
$PY -c "from data.kinship import KinshipDataset" && ok "data.kinship" || fail "data.kinship"
$PY -c "from eval.rulecheck import rule_satisfaction_report" && ok "eval.rulecheck" || fail "eval.rulecheck"

# ── 3. Dataset statistics (no training, no model download) ──────────────────
echo ""
echo "3. Dataset statistics"
$PY main.py multi-digit-stats > /dev/null && ok "multi-digit-stats" || fail "multi-digit-stats"
$PY main.py kinship-stats > /dev/null && ok "kinship-stats" || fail "kinship-stats"

# ── 4. Unit tests ────────────────────────────────────────────────────────────
if [[ $FAST_MODE -eq 0 ]]; then
  echo ""
  echo "4. Unit tests"
  PYTEST_RC=0
  TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 $PY -m pytest tests/ -q --no-header --tb=short \
    2>&1 || PYTEST_RC=$?
  if [[ $PYTEST_RC -eq 0 ]]; then
    ok "pytest: all tests passed"
  else
    fail "pytest exited with code $PYTEST_RC (see output above)"
  fi
fi

# ── 5. Multi-digit smoke training ────────────────────────────────────────────
echo ""
echo "5. Multi-digit smoke training (200 samples, 2 epochs, CPU)"
SMOKE_OUT="outputs_smoke_multi_digit"
$PY main.py train-multi-digit --config configs/multi_digit_smoke.yaml --outdir "$SMOKE_OUT" > /dev/null \
  && ok "train-multi-digit smoke" || fail "train-multi-digit smoke"

if [[ -f "$SMOKE_OUT/report.json" ]]; then
  ok "report.json written"
else
  fail "report.json not found"
fi

# ── 6. Latency benchmark (CPU, 50 samples) ──────────────────────────────────
echo ""
echo "6. Inference latency (50 samples, CPU)"
$PY scripts/benchmark_latency.py --n_samples 50 --device cpu --json /tmp/latency_smoke.json > /dev/null \
  && ok "benchmark_latency.py" || fail "benchmark_latency.py"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
echo ""

if [[ $FAIL -gt 0 ]]; then
  echo "Smoke test FAILED. See failures above."
  exit 1
fi
echo "Smoke test PASSED."
exit 0

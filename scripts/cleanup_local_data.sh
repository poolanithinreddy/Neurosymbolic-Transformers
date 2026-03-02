#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
#  scripts/cleanup_local_data.sh
#  Remove generated data artifacts and caches from the repo.
#
#  Usage:
#    bash scripts/cleanup_local_data.sh          # repo-local only
#    bash scripts/cleanup_local_data.sh --hf     # + HuggingFace caches
#    bash scripts/cleanup_local_data.sh --dry    # show what would be deleted
# ──────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY=false
CLEAR_HF=false

for arg in "$@"; do
  case "$arg" in
    --hf)   CLEAR_HF=true ;;
    --dry)  DRY=true ;;
    -h|--help)
      echo "Usage: $0 [--hf] [--dry]"
      echo "  --hf   Also clear ~/.cache/huggingface/{datasets,hub}"
      echo "  --dry   Show what would be deleted without deleting"
      exit 0
      ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# ── Helpers ─────────────────────────────────────────────
remove() {
  local target="$1"
  if [ -e "$target" ]; then
    local size
    size="$(du -sh "$target" 2>/dev/null | cut -f1)"
    if $DRY; then
      echo "  [dry-run] would delete: $target ($size)"
    else
      rm -rf "$target"
      echo "  deleted: $target ($size)"
    fi
  fi
}

echo "=== NST Cleanup ==="
echo "Repo root: $REPO_ROOT"
$DRY && echo "(DRY RUN — nothing will be deleted)"
echo ""

# ── 1. Output directories ──────────────────────────────
echo "── Output directories ──"
for d in "$REPO_ROOT"/outputs_*; do
  [ -e "$d" ] && remove "$d"
done
remove "$REPO_ROOT/outputs"
remove "$REPO_ROOT/runs"
remove "$REPO_ROOT/wandb"

# ── 2. Data caches (generated at runtime) ──────────────
echo "── Data caches ──"
remove "$REPO_ROOT/data/fever_wiki.db"
remove "$REPO_ROOT/data/fever_wiki_manifest.json"
# Any other wiki dumps or large files
for f in "$REPO_ROOT"/data/wiki* "$REPO_ROOT"/data/*wiki* "$REPO_ROOT"/data/*.jsonl "$REPO_ROOT"/data/*.zip; do
  [ -e "$f" ] && remove "$f"
done

# ── 3. Python caches ───────────────────────────────────
echo "── Python caches ──"
find "$REPO_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$REPO_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "  cleared __pycache__ and .pytest_cache"

# ── 4. HuggingFace caches (optional) ───────────────────
if $CLEAR_HF; then
  echo "── HuggingFace caches ──"
  remove "$HOME/.cache/huggingface/datasets"
  remove "$HOME/.cache/huggingface/hub"
  mkdir -p "$HOME/.cache/huggingface/datasets" "$HOME/.cache/huggingface/hub"
fi

# ── Summary ─────────────────────────────────────────────
echo ""
echo "=== Disk usage after cleanup ==="
du -sh "$REPO_ROOT" 2>/dev/null
du -sh "$REPO_ROOT/data" 2>/dev/null
if $CLEAR_HF; then
  du -sh "$HOME/.cache/huggingface/datasets" "$HOME/.cache/huggingface/hub" 2>/dev/null
fi
echo "=== Done ==="

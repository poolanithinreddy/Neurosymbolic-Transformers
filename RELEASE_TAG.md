# Release: Neural CEGIS v4.0 — Submission-Ready

**Date:** 2026-02-17

## What Changed

### New Features
- **Neural CEGIS paper draft** (`PAPER.md`): full submission-quality write-up with System 2 framing, Proposition 1 (augmented Lagrangian convergence), and all experiment tables (awaiting numbers from full runs).
- **Controlled baselines** (`training/baselines.py`): random replay, hard-example mining, and same-budget training — three ablation baselines that isolate CEGIS's contribution.
- **Latency benchmark** (`scripts/benchmark_latency.py`): wall-clock inference timing for neural/soft/lagrangian/Z3 modes.
- **Alignment plot** (`scripts/plot_alignment.py`): dual-axis "Price of Logic" figure (CSR vs λ trajectory) and CEGIS convergence plot. Outputs both PNG and PDF.
- **Table export** (`scripts/export_tables.py`): multi-seed mean±std tables in LaTeX and Markdown.
- **CLUTRR dataset loader** (`data/clutrr.py`): HuggingFace + synthetic fallback for natural-language kinship reasoning.
- **One-shot experiment runner** (`run_all.sh`): 8-stage script covering all paper experiments, with `--quick` smoke mode.

### CLI Updates (main.py)
- Added 4 new commands: `baseline`, `latency`, `plot`, `export-tables` (18 total).

### Configs
- Added 3 baseline YAML configs: `multi_digit_random_replay.yaml`, `multi_digit_hard_mining.yaml`, `multi_digit_same_budget.yaml`.

### Documentation
- `README.md`: complete rewrite — concise, submission-quality, 18-command CLI reference.
- `colab/README_COLAB.md`: step-by-step Colab reproduction guide with copy-paste cells.
- `colab/nst_playbook.py`: updated to v4 with baseline + latency + table cells.

### Bug Fixes
- Fixed YAML `1e-3` parsed as string causing `TypeError` in `AdamW` (added `float()` casts).
- Fixed baseline config resolution (baselines now read from `training`/`lagrangian`/`baseline` YAML sections, not just `cegis`).
- Fixed `--outdir` CLI flag for plot and table export scripts.
- Fixed Colab playbook JSON field name mismatches.
- Cleaned up `.gitignore` (deduplicated, added `outputs_*/`, `figures/`, `results/`).

### Infrastructure
- `Makefile`: 6 new targets (`experiments`, `baselines`, `latency`, `plots`, `tables`, `all-paper`).
- `pyproject.toml`: added `matplotlib` and `numpy` to `[dev]` extras.
- All 119 tests pass.

## Remaining TODOs (require GPU time)
- Run `./run_all.sh` on Colab T4 (~3 hours) to fill Tables 1–4 in PAPER.md.
- Verify all `— VERIFY` citations in PAPER.md against actual publication metadata.
- Generate final figures from real training logs.

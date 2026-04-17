# Changelog

## 0.3.0

### Constraint engine overhaul (v2.1 — "precision > recall")
- **NumericalConstraint**: requires shared entities + quantity context words ("total", "population", etc.) to fire; eliminates false positives from incidental number mentions
- **NegationConstraint**: requires shared content words between claim and evidence; prevents false negation detection on unrelated sentences; uses strict antonym pairs
- **EntityOverlapConstraint**: dual-metric approach combining entity overlap AND content word overlap; both must be low to fire, reducing false NEI bias
- **TemporalConstraint**: requires shared entities between claim/evidence; prevents false temporal conflicts from unrelated time references
- **HedgeModalityConstraint**: stricter thresholds (50%+ hedge words required) to avoid false positives on normal academic language
- **NEW MutualExclusionConstraint** (C7): detects categorical assignment conflicts ("X is Y" vs "X is Z") — captures a pattern DeBERTa often misses

### Training improvements
- **Uncertainty-focused constraint loss**: constraints only influence the gradient when the model is *uncertain* (high entropy) AND the constraint *disagrees* with the prediction; avoids overriding confident correct predictions
- **Gentler warmup schedule**: starts at 5% weight (was 10%), ramps gradually through training phases
- Updated default `n_constraints` from 6 to 7 throughout model and config files

### Packaging & docs
- Synced version to 0.3.0 in both `pyproject.toml` and `__init__.py`
- Updated author to Nithin Reddy Poola
- Added fair neural-large baseline config (`fever_gold_neural_large.yaml`)
- Updated README and RESULTS.md to reflect 7-constraint architecture
- Added NST-VERI v3 row to results tracker

## 0.1.0

### Core features
- Neural CEGIS training loop with augmented Lagrangian (adaptive λ)
- Evidence-Conditioned Constraint Gating (ECCG) for FEVER
- Multi-digit addition benchmark with carry-propagation constraints
- Kinship relational reasoning benchmark with compositional depth split
- FEVER fact verification pipeline (Setting A: gold evidence, Setting B: BM25 retrieval)
- Controlled baselines: random replay, hard-example mining, same-budget training
- Multi-seed runner for mean ± std results
- Inference latency benchmark, alignment phase plots, LaTeX/Markdown table export
- One-click Colab playbooks (nst_playbook.py, fever_playbook.py)
- 232-test unit test suite

### Infrastructure
- `pyproject.toml` with proper package discovery and editable install
- `pytest` with `importmode = importlib` and `pythonpath = ["."]`
- `scripts/smoke_test.sh` — full stack verification in <60s, no GPU required
- `configs/multi_digit_smoke.yaml` — CPU smoke config (200 samples, 2 epochs)
- `Makefile` targets: setup, test, smoke, experiments, baselines, latency, plots, tables
- CI workflow (GitHub Actions, ubuntu-latest, Python 3.11)

### Bug fixes
- Fixed YAML float parsing (1e-3 was parsed as string, causing TypeError in AdamW)
- Fixed editable install: pyproject.toml package discovery now finds all actual packages
- Fixed pytest: added importmode=importlib to avoid root __init__.py collision
- Fixed test_fever.py and test_eccg.py: removed HuggingFace network dependency from fixtures

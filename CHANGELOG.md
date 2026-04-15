# Changelog

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

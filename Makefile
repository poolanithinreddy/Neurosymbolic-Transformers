PY=python3
SEEDS=42,43,44

.PHONY: setup fmt lint test smoke colab experiments baselines latency plots tables all-paper

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .[dev]
	$(PY) -m spacy download en_core_web_sm || true

fmt:
	ruff format .
	black -l 100 .

lint:
	ruff check .
	mypy models training symbolic eval logic || true

test:
	TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 pytest -q

smoke:
	bash scripts/smoke_test.sh

colab:
	@echo "See colab/README_COLAB.md for copy-paste Colab instructions."

# ── Paper experiment targets ──────────────────────────────────

experiments:
	@echo "▸ Multi-digit: neural, soft, lagrangian, CEGIS"
	$(PY) main.py multi-seed --task train-multi-digit --config configs/multi_digit_neural.yaml --seeds $(SEEDS)
	$(PY) main.py multi-seed --task train-multi-digit --config configs/multi_digit_soft.yaml --seeds $(SEEDS)
	$(PY) main.py multi-seed --task train-multi-digit --config configs/multi_digit_lagrangian.yaml --seeds $(SEEDS)
	$(PY) main.py multi-seed --task train-cegis --config configs/multi_digit_cegis.yaml --seeds $(SEEDS)
	@echo "▸ Kinship: neural, lagrangian, CEGIS"
	$(PY) main.py multi-seed --task train-kinship --config configs/kinship_neural.yaml --seeds $(SEEDS)
	$(PY) main.py multi-seed --task train-kinship --config configs/kinship_lagrangian.yaml --seeds $(SEEDS)
	$(PY) main.py multi-seed --task train-kinship-cegis --config configs/kinship_cegis.yaml --seeds $(SEEDS)

baselines:
	$(PY) main.py baseline --method random-replay --config configs/multi_digit_random_replay.yaml --seeds $(SEEDS)
	$(PY) main.py baseline --method hard-mining --config configs/multi_digit_hard_mining.yaml --seeds $(SEEDS)
	$(PY) main.py baseline --method same-budget --config configs/multi_digit_same_budget.yaml --seeds $(SEEDS)

latency:
	mkdir -p results
	$(PY) scripts/benchmark_latency.py --n_samples 500 --device cpu --json results/latency_cpu.json

plots:
	mkdir -p figures
	$(PY) scripts/plot_alignment.py --logdir outputs_train-multi-digit_multiseed/seed_42 --outdir figures/ || true
	$(PY) scripts/plot_alignment.py --logdir outputs_train-cegis_multiseed/seed_42 --cegis --outdir figures/ || true

tables:
	mkdir -p results
	$(PY) scripts/export_tables.py --task multi_digit --format latex --outdir results/ || true
	$(PY) scripts/export_tables.py --task multi_digit --format markdown --outdir results/ || true
	$(PY) scripts/export_tables.py --task kinship --format latex --outdir results/ || true
	$(PY) scripts/export_tables.py --task kinship --format markdown --outdir results/ || true

all-paper: experiments baselines latency plots tables
	@echo "✅ All paper artifacts generated."


PY=python3

.PHONY: setup fmt lint test smoke colab

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .[dev]
	$(PY) -m spacy download en_core_web_sm || true

fmt:
	ruff format .
	black -l 100 .

lint:
	ruff check .
	mypy nst || true

test:
	pytest -q --cov=nst

smoke:
	$(PY) training/train.py --config configs/mac_quick.yaml --task fever --outdir outputs_quick
	$(PY) eval/fever.py --ckpt outputs_quick/ckpt --data data/fever.tsv --split dev --report outputs_quick/fever_dev.json --device cpu

colab:
	@echo "Run the cells in colab/colab_commands.md inside Colab."

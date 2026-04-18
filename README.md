# Neuro-Symbolic Transformers (NST)

**Neural CEGIS: Counterexample-Guided Training for Constraint-Satisfying Neural Networks**

NST is a training framework that adapts counterexample-guided inductive synthesis (CEGIS) — the core loop in formal program verification — to gradient-based learning. A symbolic verifier finds inputs where a trained model violates domain constraints, feeds them back as targeted training data, and the loop repeats until violations reach zero. Combined with an augmented Lagrangian that automatically learns the constraint–task tradeoff (the "price of logic," λ\*), this closes the gap between symbolic correctness and neural flexibility.

---

## Why This Matters

Most neuro-symbolic methods inject constraints as a soft loss term with a fixed weight λ. This has two well-known problems: λ requires task-specific tuning, and it provides no feedback on *where* the model fails. Standard regularisation spreads the penalty uniformly across all training examples.

Neural CEGIS addresses both:
- **Targeted counterexamples**: the verifier identifies specific inputs that violate constraints and concentrates training on them, not random examples.
- **Adaptive λ**: the dual variable automatically rises when constraints are violated and falls when they are satisfied, eliminating manual tuning.

The result: a training regime where constraint satisfaction improves monotonically across verification rounds, and convergence is measurable by the residual counterexample count.

---

## Core Contributions

1. **Neural CEGIS loop** — wraps a standard Lagrangian training loop inside a verify → augment → retrain cycle. Converges when no counterexamples remain.
2. **Augmented Lagrangian with learned λ\*** — the dual variable at equilibrium is the "price of logic": how much task accuracy the model trades for each unit of constraint satisfaction.
3. **Evidence-Conditioned Constraint Gating (ECCG)** — per-sample, per-constraint reliability gates that learn when noisy symbolic extractors are informative.
4. **Three benchmarks**: multi-digit addition with carry propagation, kinship reasoning with distractors, and FEVER fact verification.
5. **Controlled baselines**: random replay, hard-example mining, and same-budget training isolate the effect of constraint-targeted counterexamples.
6. **GroundedVerifier API** — a reusable, pip-installable verification layer that wraps any NLI model with symbolic constraints and ECCG gating.

---

## GroundedVerifier — Quick Start

```python
from nst import GroundedVerifier

# Wrap any HuggingFace NLI model with symbolic verification
verifier = GroundedVerifier(model_name="microsoft/deberta-v3-base")

result = verifier.verify(
    claim="The Eiffel Tower is 500 meters tall.",
    evidence="The Eiffel Tower is 330 metres tall.",
)

print(result.label)          # "REFUTES"
print(result.confidence)     # 0.87
print(result.abstain)        # False
print(result.constraint_details)  # Per-constraint diagnostics

# Latency benchmark
bench = verifier.benchmark_latency()
print(f"P95: {bench['p95_ms']:.1f} ms, constraint overhead: {bench['overhead_constraints_pct']:.1f}%")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Neural CEGIS Outer Loop                                │
│  ┌───────────────┐    ┌──────────────┐                  │
│  │ LEARNER       │◄───│ CE Buffer    │                  │
│  │ (Lagrangian   │    │ (constraint  │                  │
│  │  inner loop)  │    │  violations) │                  │
│  └───────┬───────┘    └──────▲───────┘                  │
│          │                   │                          │
│          ▼                   │                          │
│  ┌───────────────┐    ┌──────┴───────┐                  │
│  │ Trained Model │───►│ VERIFIER     │                  │
│  │               │    │ (symbolic    │                  │
│  │               │    │  constraint  │                  │
│  │               │    │  checker)    │                  │
│  └───────────────┘    └──────────────┘                  │
│  Terminates when CE count → 0                           │
└─────────────────────────────────────────────────────────┘
```

The LEARNER minimises the augmented Lagrangian:

```
L(θ, λ) = L_task(θ) + λ·(L_logic(θ) − ε) + ρ/2·[max(0, L_logic(θ) − ε)]²
λ ← max(0, λ + α·(L_logic(θ) − ε))    # dual update after each epoch
```

---

## Project Structure

```
nst/
├── main.py                 # Unified CLI (24 commands)
├── configs/                # 30+ YAML experiment configs
├── models/                 # CNN perception, Transformer, NST-VERI v1/v2 models
├── training/               # Training loops, CEGIS, Lagrangian, VERI v2, multi-seed
├── symbolic/               # Constraint loss, Lagrangian state, ECCG gate, rule engine
├── data/                   # Dataset generators (digit addition, kinship, CLUTRR, FEVER)
├── eval/                   # Evaluation, calibration, temperature scaling, leakage checks
├── scripts/                # Benchmark, plots, table export, cleanup, smoke test
├── results/                # Results aggregation (LaTeX/Markdown table generation)
├── tests/                  # 233 unit tests
├── colab/                  # One-click Colab playbooks
├── PAPER.md                # Technical write-up
├── RUNBOOK.md              # FEVER experiment playbook
├── RESULTS.md              # Results tracker
└── CODE_REVIEW.md          # Codebase analysis and design decisions
```

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.2. No GPU required for the smoke test or unit tests.

### macOS / Apple Silicon

MPS is supported via `device: "auto"` in all configs.

```bash
# If you hit MPS OOM during large FEVER runs:
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python main.py train-fever-nst \
    --config configs/fever_mac_neural.yaml
```

**Common issues:**

| Issue | Fix |
|-------|-----|
| `sentencepiece` import error | `pip install sentencepiece>=0.1.99 protobuf>=4.0` |
| `fp16` crash on MPS | Set `fp16: false` in config |
| DeBERTa tokenizer warning | Benign — overflowing tokens from `DebertaV2Tokenizer`; safe to ignore |
| `tiktoken` import error | `pip install tiktoken` |

---

## Quick Start

```bash
# Run the full smoke test (no GPU, <60s)
bash scripts/smoke_test.sh

# Dataset statistics (no training)
python main.py multi-digit-stats
python main.py kinship-stats

# Train Neural CEGIS on multi-digit addition
python main.py train-cegis --config configs/multi_digit_cegis.yaml

# Train Neural CEGIS on kinship reasoning
python main.py train-kinship-cegis --config configs/kinship_cegis.yaml

# Run all unit tests
TRANSFORMERS_OFFLINE=1 python -m pytest tests/ -v
```

---

## Smoke Test

Verifies the full stack works on any machine without a GPU:

```bash
bash scripts/smoke_test.sh
```

What it checks:
1. Python version ≥ 3.10
2. Core package imports (torch, transformers, all NST modules)
3. Dataset statistics generation
4. Full unit test suite (233 tests)
5. Multi-digit smoke training (200 samples, 2 epochs, CPU, ~2s)
6. Inference latency benchmark (50 samples, CPU)

Expected output: `15 passed, 0 failed` in under 60 seconds.

---

## Full Experiment Suite

```bash
# Full suite — all methods, 3 seeds (~3h on Colab T4)
chmod +x run_all.sh && ./run_all.sh

# Or via Makefile:
make experiments      # multi-digit + kinship, all methods, 3 seeds
make baselines        # random-replay, hard-mining, same-budget
make latency          # inference latency benchmark (CPU)
make plots            # alignment phase + CEGIS convergence figures
make tables           # LaTeX + Markdown results tables
make all-paper        # everything above
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `train` | Train single-digit addition model |
| `eval` | Evaluate digit-addition checkpoint |
| `train-multi-digit` | Train multi-digit addition (neural/soft/lagrangian) |
| `train-cegis` | Train multi-digit with Neural CEGIS |
| `train-kinship` | Train kinship relational reasoning |
| `train-kinship-cegis` | Train kinship with Neural CEGIS |
| `multi-seed` | Multi-seed run for mean ± std |
| `baseline` | Run controlled baseline (random-replay / hard-mining / same-budget) |
| `ablation` | Run all single-digit ablation experiments |
| `results` | Generate results tables from output directories |
| `export-tables` | Export multi-seed tables (LaTeX/Markdown) |
| `latency` | Benchmark inference latency |
| `plot` | Generate alignment/convergence plots |
| `multi-digit-stats` | Multi-digit dataset statistics |
| `kinship-stats` | Kinship dataset statistics |
| `train-fever-nst` | Train FEVER NLI with DeBERTa + NST constraints |
| `train-fever-veri` | Train NST-VERI v1 (3-phase, heuristic constraints) |
| `train-fever-veri-v2` | **Train NST-VERI v2** (learned multi-task, focal loss) |
| `build-fever-wiki-cache` | Build SQLite wiki cache (avoids OOM) |
| `fever-stats` | FEVER dataset statistics + split hashes |
| `eval-fever` | Evaluate a FEVER NLI checkpoint |
| `export-fever-tables` | Export FEVER results as Markdown tables |
| `leakage-check` | Run data leakage checks on FEVER splits |

---

## Configs

| Config | Method | Task |
|--------|--------|------|
| `multi_digit_neural.yaml` | Pure neural baseline | Multi-digit |
| `multi_digit_soft.yaml` | Soft constraints (fixed λ) | Multi-digit |
| `multi_digit_lagrangian.yaml` | Augmented Lagrangian (adaptive λ) | Multi-digit |
| `multi_digit_cegis.yaml` | **Neural CEGIS** | Multi-digit |
| `multi_digit_random_replay.yaml` | Random replay baseline | Multi-digit |
| `multi_digit_hard_mining.yaml` | Hard mining baseline | Multi-digit |
| `multi_digit_same_budget.yaml` | Same budget baseline | Multi-digit |
| `multi_digit_smoke.yaml` | Smoke test (200 samples, CPU) | Multi-digit |
| `kinship_neural.yaml` | Pure neural baseline | Kinship |
| `kinship_lagrangian.yaml` | Augmented Lagrangian | Kinship |
| `kinship_cegis.yaml` | **Neural CEGIS** | Kinship |
| `fever_gold_neural.yaml` | Pure neural baseline | FEVER (Gold Evidence) |
| `fever_gold_nst_soft.yaml` | Soft constraints (fixed λ) | FEVER (Gold Evidence) |
| `fever_gold_nst_cegis.yaml` | **Neural CEGIS** | FEVER (Gold Evidence) |
| `fever_gold_nst_gated.yaml` | **ECCG** (gated constraints) | FEVER (Gold Evidence) |
| `fever_gold_nst_veri.yaml` | **NST-VERI** (verification-enhanced) | FEVER (Gold Evidence) |
| `fever_gold_nst_veri_a100.yaml` | **NST-VERI** (A100 optimized) | FEVER (Gold Evidence) |
| `fever_veri_v2_10k_a100.yaml` | **NST-VERI v2** (10K, learned multi-task) | FEVER (Gold Evidence) |
| `fever_veri_v2_full_a100.yaml` | **NST-VERI v2** (full data, DeBERTa-large) | FEVER (Gold Evidence) |
| `fever_gold_smoke.yaml` | Smoke test (200 samples) | FEVER (Gold Evidence) |
| `fever_micro_smoke.yaml` | Micro smoke (50 samples, tiny BERT) | FEVER |
| `fever_pipeline_neural.yaml` | Pure neural baseline | FEVER (Full Pipeline) |
| `fever_pipeline_nst_cegis.yaml` | **Neural CEGIS** | FEVER (Full Pipeline) |

---

## FEVER Fact Verification

NST applies Neural CEGIS to the FEVER fact verification task, classifying claim–evidence pairs as SUPPORTS / REFUTES / NOT ENOUGH INFO.

**Two evaluation settings:**
- **Setting A (Gold Evidence)**: oracle evidence → Label Accuracy. Isolates NLI capability.
- **Setting B (Full Pipeline)**: BM25 retrieval → NLI → FEVER Score. End-to-end pipeline.

**Architecture**: DeBERTa-v3-base/large (184M–442M params, LoRA r=16) with learned multi-task verification:

**NST-VERI v2** (current flagship) replaces heuristic constraint integration with learned verification primitives:
- **FocalCrossEntropy**: Per-class gamma (REFUTES γ=3.0) to address the REFUTES accuracy bottleneck
- **ContradictionHead**: Separate AttentionPool learning entity-level conflict patterns (silver label: REFUTES→1)
- **EvidenceRelevanceHead**: Separate AttentionPool learning evidence sufficiency (silver label: NEI→0)
- **RecalibrationNetwork**: Signal-only correction (7-dim input: NLI logits + aux probs + confidence + entropy), not a second classifier
- **Supervised Contrastive**: Class-prototype contrastive learning for representation shaping
- **Symmetric R-Drop**: Bidirectional KL with all losses averaged across both forward passes
- **Symbolic Fusion** (inference-time): Optional constraint-based logit correction for uncertain predictions

Training uses a two-phase schedule:
- **Phase 1** (epochs 0–1): NLI warmup with low-weight auxiliary heads
- **Phase 2** (epochs 2+): Full multi-task with R-Drop, contrastive, and ramped auxiliary weights

```bash
# Build wiki cache (one-time, ~4 min, ~25 MB SQLite)
python main.py build-fever-wiki-cache

# FEVER smoke test (tiny BERT, 50 samples, <2 min on CPU)
python main.py train-fever-nst --config configs/fever_micro_smoke.yaml

# NST-VERI v2 — 10K development loop (A100, ~20 min)
python main.py train-fever-veri-v2 --config configs/fever_veri_v2_10k_a100.yaml

# NST-VERI v2 — Full data with DeBERTa-v3-large (A100, ~2h)
python main.py train-fever-veri-v2 --config configs/fever_veri_v2_full_a100.yaml

# Full training — NST-CEGIS (requires GPU, ~3h on T4)
python main.py train-fever-nst --config configs/fever_gold_nst_cegis.yaml

# Multi-seed for error bars
python main.py multi-seed --task train-fever-veri-v2 \
    --config configs/fever_veri_v2_10k_a100.yaml --seeds 42,43,44

# Evaluate a checkpoint
python main.py eval-fever --ckpt outputs_fever_gold_neural/ckpt/best_model.pt
```

---

## Reproducibility

| Item | Detail |
|------|--------|
| Hardware | Colab T4 (16 GB VRAM) or macOS with MPS |
| Seeds | {42, 43, 44} — results reported as mean ± std |
| Framework | PyTorch ≥ 2.2, Python ≥ 3.10 |
| Tests | 233 tests, `pytest tests/ -v` |
| License | Apache-2.0 |

---

## Data & Storage

| Artifact | Path | Committed? |
|----------|------|------------|
| Fixture TSVs (toy data) | `data/*.tsv` | Yes — small test fixtures |
| Dataset generators | `data/*.py` | Yes — source code |
| FEVER wiki cache | `data/fever_wiki.db` | No — build at runtime |
| Training outputs | `outputs_*/` | No — gitignored |
| HuggingFace cache | `~/.cache/huggingface/` | No — system cache |

```bash
# Clean up repo-local outputs and caches
bash scripts/cleanup_local_data.sh

# Also wipe HuggingFace system cache (~1–20 GB)
bash scripts/cleanup_local_data.sh --hf

# Preview what would be removed (dry run)
bash scripts/cleanup_local_data.sh --dry
```

> **Warning**: do not commit datasets, caches, or model checkpoints. The `.gitignore` blocks `*.db`, `*.safetensors`, `*.pt`, `*.bin`, and `outputs_*/`. If you accidentally stage a large file: `git rm --cached <file>`

---

## Colab

See `colab/nst_playbook.py` for copy-paste cells. Runs the full experiment suite in ~40 minutes on a T4.

For FEVER, see `colab/fever_playbook.py` (12 cells, ~45 min on T4).

```python
!git clone https://github.com/poolanithinreddy/Neurosymbolic-Transformers.git nst
%cd nst
!pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -e ".[dev]" -q
!python main.py build-fever-wiki-cache
!python main.py train-cegis --config configs/multi_digit_cegis.yaml
```

---

## Limitations

- **Synthetic benchmarks**: multi-digit addition and kinship are controlled environments. Real domains with messier constraint structures may require more engineering work on the verifier.
- **Training cost**: CEGIS adds roughly 2× training time due to verification rounds. For very large models this overhead can become significant.
- **Sampling-based verification**: the verifier samples from the training set and cannot exhaustively cover the input space, so "zero counterexamples" is evidence of constraint satisfaction, not a formal proof.
- **Domain-specific verifier**: the current verifier requires hand-specified constraint rules. Generalising to arbitrary domains requires writing new constraint modules.
- **Results pending**: FEVER full-run numbers (Table 1, 2 in PAPER.md) require GPU time (~3h per config). All infrastructure is in place; run `make experiments` to populate them.

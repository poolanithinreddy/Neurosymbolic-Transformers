# NST Results Tracker

> **Status**: Full training pipeline operational. GroundedVerifier API available.
> NST-VERI constraint warmup bug found and fixed. Fair comparison configs added.
>
> **Honesty policy**: All reported numbers are from actual training runs.
> Placeholders marked with "—" have not been measured yet.
> If the real result is 84%, we report 84%. Negative results are reported too.

---

## Multi-Digit Addition

**Model**: CNN encoder (MNIST-style digits, 28×28)  
**Task**: two-digit + two-digit addition with carry propagation  
**Compositional split**: train on no-carry pairs, test on ≥1 carry (Comp) and 2-carry (Hard)  
**Seeds**: {42, 43, 44}, results as mean ± std

### Smoke Tests

| Config | Mode | Samples | Epochs | Runtime |
|--------|------|---------|--------|---------|
| `multi_digit_smoke` | Lagrangian | 200 | 2 | ~2s CPU |

### Full Runs

| Model | Sum Acc (IID) | Sum Acc (Comp) | Sum Acc (Hard) | CSR (Comp) | Comp Gap |
|-------|--------------|----------------|----------------|------------|----------|
| Pure Neural | — | — | — | — | — |
| NST-Soft (λ=0.5) | — | — | — | — | — |
| NST-Lagrangian | — | — | — | — | — |
| Random Replay | — | — | — | — | — |
| Hard Mining | — | — | — | — | — |
| Same Budget | — | — | — | — | — |
| **NST-CEGIS** | — | — | — | — | — |

*CSR = Constraint Satisfaction Rate (fraction of predictions satisfying carry rule)*

---

## Kinship Relational Reasoning

**Model**: Transformer encoder (2 layers, 128-dim, 4 heads)  
**Task**: kinship relation classification from chain premises  
**Compositional split**: train depth 1–3, test depth 4–6  

### Full Runs

| Model | Acc (IID) | Acc (Comp) | CSR (Comp) | Comp Gap |
|-------|-----------|------------|------------|----------|
| Pure Neural | — | — | — | — |
| NST-Lagrangian | — | — | — | — |
| **NST-CEGIS** | — | — | — | — |

---

## FEVER Fact Verification

**Model**: DeBERTa-v3-base (184M) / DeBERTa-v3-large (405M) with LoRA  
**Task**: SUPPORTS / REFUTES / NOT ENOUGH INFO classification  
**Wiki cache**: 14,363 pages (~98.8% coverage)  
**Seeds**: {42, 43, 44}  
**Key fix**: max_length=384 for all configs (previously 256 for some — evidence was truncated)

### Available Training Modes

| Mode | Description | Config |
|------|-------------|--------|
| Neural | Pure DeBERTa NLI (no symbolic) | `fever_gold_neural.yaml` |
| Soft | Fixed-weight symbolic constraints (v1) | `fever_gold_nst_soft.yaml` |
| Lagrangian | Adaptive λ dual-variable constraints | `fever_gold_lagrangian.yaml` |
| CEGIS | Counterexample-guided refinement | `fever_gold_nst_cegis.yaml` |
| ECCG/Gated | Per-sample constraint gating | `fever_gold_nst_gated.yaml` |
| **VERI** | **3-phase verification-enhanced (flagship)** | `fever_gold_nst_veri_a100.yaml` |

### Smoke Tests

| Config | Model | Train | Dev | Epochs | dev_acc | Runtime |
|--------|-------|-------|-----|--------|---------|---------|
| `fever_micro_smoke` | BERT-tiny (4.4M) | 50 | 25 | 2 | ~0.28 | <30s CPU |
| `fever_gold_smoke` | DeBERTa-v3-base | 200 | 100 | 1 | ~0.32 | ~30s GPU |
| `fever_veri_smoke` | DeBERTa-v3-base+VERI | 200 | 100 | 3 | — | ~2 min CPU |

### Setting A: Gold Evidence

> **Run date**: 2026-04-16, A100-SXM4-40GB, seed=42
> Evidence: SQLite wiki cache (14,363 pages), 53% of claims have >30 char evidence text

| Mode | Config | Label Acc | ECE ↓ | Brier ↓ | DevTest | Time | Notes |
|------|--------|-----------|-------|---------|---------|------|-------|
| Neural (base) | `fever_gold_neural` | **0.8378** | **0.0401** | **0.2395** | **0.8350** | 38m | DeBERTa-v3-base, 184M, full FT |
| Neural (large) | `fever_gold_neural_large` | — | — | — | — | — | DeBERTa-v3-large+LoRA, **fair baseline** |
| NST-VERI v1 | `fever_gold_nst_veri` | 0.8384 | 0.0423 | 0.2369 | 0.8320 | 124m | **Bug: constraints never fired** |
| **NST-VERI v2** | `fever_gold_nst_veri` | — | — | — | — | — | **Fixed warmup, patience=10** |
| Ablation: no cst | `ablation_no_constraints` | — | — | — | — | — | VERI without Phase 3 constraints |
| Ablation: fixed λ | `ablation_fixed_lambda` | — | — | — | — | — | VERI without ECCG gating |
| Ablation: no ctr | `ablation_no_contrastive` | — | — | — | — | — | VERI without Phase 2 contrastive |

**Per-label breakdown (dev):**

| Mode | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|------|----------|---------|-----------------|
| Neural | 0.8670 (n=6014) | 0.8202 (n=5969) | 0.8259 (n=6015) |
| NST-VERI | 0.8964 (n=6014) | 0.7953 (n=5969) | 0.8231 (n=6015) |

**Honest assessment:**
- **NST-VERI and Neural Baseline are effectively tied** on overall accuracy (+0.06%, within noise)
- NST-VERI has better SUPPORTS accuracy (+2.9%) but worse REFUTES (-2.5%)
- Brier score favors VERI slightly (0.2369 vs 0.2395) but ECE is slightly worse
- **Critical bug found**: constraint loss (`cst`) was always 0.0 — constraints never activated
  - Root cause: `_constraint_warmup(epoch=2, epochs=5)` returned 0.0 at Phase 3 start
  - Early stopping (patience=6) fired at step 5000 before constraints ramped up
  - **Fix applied**: warmup now starts at non-zero from Phase 3 start; patience increased to 10
- NST-VERI used a larger model (DeBERTa-v3-large 405M vs base 184M) yet didn't outperform
- Result represents **the neural backbone's contribution, not symbolic constraints**
- Single seed, no variance estimate — results are directional, not definitive

### Setting B: Retrieved Evidence (Full Pipeline)

| Mode | Config | FEVER Score | Label Acc | Recall@5 |
|------|--------|-------------|-----------|----------|
| Neural | `fever_pipeline_neural` | — | — | — |
| CEGIS | `fever_pipeline_nst_cegis` | — | — | — |

### NST-VERI Architecture

The flagship model combines:
1. **DeBERTa-v3-large backbone** with LoRA (r=16, α=32) — efficient fine-tuning
2. **K=6 verification heads** — auxiliary binary classifiers bridging neural→symbolic
3. **Residual correction** — zero-init logit adjustment from verification signals
4. **Supervised contrastive head** — representation shaping via class prototypes
5. **Adaptive per-sample λ** — learns when to trust constraints per example
6. **Focal loss** — focuses training on hard REFUTES/NEI boundary cases
7. **Constraint engine v2** — 6 probabilistic constraints (numerical, negation, entity overlap, sufficiency, temporal, hedge)

Training proceeds in 3 phases:
- **Phase 1** (20% of epochs): NLI + auxiliary heads only
- **Phase 2** (20%): + contrastive loss on high-confidence constraint examples
- **Phase 3** (60%): + adaptive constraint loss with curriculum warmup

---

## Quick Start

```bash
# Smoke test (verify pipeline works)
python main.py train-fever-veri --config configs/fever_veri_smoke.yaml

# Rapid development (10K subset, ~15 min GPU)
python main.py train-fever-nst --config configs/fever_rapid_10k.yaml

# Full NST-VERI on A100
python main.py train-fever-veri --config configs/fever_gold_nst_veri_a100.yaml

# Evaluate checkpoint
python eval/fever.py --ckpt outputs_fever_gold_nst_veri/ckpt --model_type veri
```

---

## Integrity

- Split hashes logged in every `report.json`
- Leakage guard enforced in `FeverPipelineDataset`
- CEGIS mines counterexamples from training set only
- Test set never touched during training or hyperparameter selection
- 232+ unit tests pass (`pytest tests/`)
- All configs use max_length=384 (no evidence truncation)
- Post-hoc temperature scaling for calibration (learned on dev set)

---

## GroundedVerifier API

The reusable verification layer — the engineering artifact of this project.

```python
from nst import GroundedVerifier

verifier = GroundedVerifier(model_name="microsoft/deberta-v3-base")
result = verifier.verify("Paris is the capital of France",
                          "Paris is the capital city of France.")
print(result.label, result.confidence, result.abstain)
# result.constraint_details shows per-constraint diagnostics
```

**Features:**
- Wraps any HuggingFace NLI model
- 6 probabilistic constraints (ConstraintEngineV2)
- ECCG per-sample gating (465 params)
- Calibrated abstention on insufficient evidence
- Latency benchmarking (`verifier.benchmark_latency()`)
- Save/load serialization

---

## Key Questions (Experimental Agenda)

1. **Does the constraint warmup fix actually make constraints fire?** (NST-VERI v2)
2. **Does VERI beat neural when using the SAME backbone?** (Neural-Large vs VERI v2)
3. **Which components matter?** (ablation: ±constraints, ±contrastive, ±ECCG)
4. **What is the latency overhead of symbolic verification?** (GroundedVerifier benchmark)
5. **Do constraints improve calibration even if accuracy is tied?** (ECE, Brier comparison)

# NST Results Tracker

> **Status**: Full training pipeline operational. NST-VERI flagship method now wired.
> Run `python main.py train-fever-veri --config configs/fever_gold_nst_veri_a100.yaml` for full results.
>
> **Honesty policy**: All reported numbers are from actual training runs.
> Placeholders marked with "—" have not been measured yet.
> If the real result is 84%, we report 84%.

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

| Mode | Config | Label Acc | ECE ↓ | Brier ↓ | Notes |
|------|--------|-----------|-------|---------|-------|
| Neural | `fever_gold_neural` | — | — | — | Baseline |
| Soft (λ=0.1) | `fever_gold_nst_soft` | — | — | — | Fixed constraint weight |
| Lagrangian | `fever_gold_lagrangian` | — | — | — | Adaptive λ |
| CEGIS | `fever_gold_nst_cegis` | — | — | — | Counterexample-guided |
| ECCG | `fever_gold_nst_gated` | — | — | — | Per-sample gating |
| **NST-VERI** | `fever_gold_nst_veri_a100` | — | — | — | **3-phase: NLI→contrastive→constraints** |

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

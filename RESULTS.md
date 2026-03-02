# FEVER Results — Neurosymbolic Transformers

> **Branch:** `fever-sota`  
> **Model:** DeBERTa-v3-base (184M params)  
> **Dataset:** FEVER v1.0 (train: ~145K, labelled_dev: ~19K)  
> **Wiki Cache:** 14,363 pages (98.8% coverage)

---

## Setting A: Gold Evidence (Oracle NLI)

Gold evidence sentences provided by FEVER annotations.
Measures NLI classification accuracy in isolation.

### Smoke Tests (Validation Only)

| Config | Model | Train | Dev | Epochs | dev_acc | Time |
|--------|-------|-------|-----|--------|---------|------|
| `fever_micro_smoke` | BERT-tiny (4.4M) | 50 | 25 | 2 | 0.28 | 8s |
| `fever_gold_smoke` | DeBERTa-v3-base | 200 | 100 | 1 | 0.32 | 17s |

### Full Runs

| Mode | Config | Label Acc | ECE ↓ | Brier ↓ | dev_test Acc | Notes |
|------|--------|-----------|-------|---------|--------------|-------|
| Neural | `fever_gold_neural` | — | — | — | — | Baseline (no constraints) |
| Soft (λ=0.1) | `fever_gold_nst_soft` | — | — | — | — | Fixed constraint weight |
| Lagrangian | `fever_gold_lagrangian` | — | — | — | — | Adaptive λ |
| CEGIS | `fever_gold_nst_cegis` | — | — | — | — | Counterexample-guided |
| **ECCG (novel)** | `fever_gold_nst_gated` | — | — | — | — | Evidence-Conditioned Gating |

*Full runs pending — requires GPU for reasonable training time (~3h per config on T4).*

---

## Setting B: Retrieved Evidence (Full Pipeline)

Evidence retrieved via BM25 from Wikipedia dump.
Gold evidence **NEVER** accessed. Measures end-to-end performance.

| Mode | Config | FEVER Score | Label Acc | Recall@5 | Notes |
|------|--------|-------------|-----------|----------|-------|
| Neural | `fever_pipeline_neural` | — | — | — | Pending |
| CEGIS | `fever_pipeline_nst_cegis` | — | — | — | Pending |

---

## Ablations

### Constraint Ablation (ECCG mode, Setting A)

| Constraints | Label Acc | ECE ↓ | Δ vs full |
|-------------|-----------|-------|-----------|
| All 5 (C1–C5) | — | — | — |
| No C4 (entity overlap) | — | — | — |
| No C5 (empty evidence) | — | — | — |
| No C1+C2 (date+number) | — | — | — |
| None (= neural) | — | — | — |

### Multi-Seed (mean ± std, n=3)

| Mode | Label Acc | ECE ↓ | Seeds |
|------|-----------|-------|-------|
| Neural | — | — | 42, 1337, 2024 |
| ECCG | — | — | 42, 1337, 2024 |

---

## Key Fixes Applied (fever-sota branch)

1. **Tokenization fix:** `fever_collate_fn` now uses proper sentence-pair encoding
   (`tokenizer(claim, evidence)`) instead of literal `" [SEP] "` concatenation.
   DeBERTa-v3 uses `</s>` as separator, not `[SEP]`.

2. **Full reproducibility:** numpy + torch + DataLoader seeds, `cudnn.deterministic=True`,
   `cudnn.benchmark=False`.

3. **Step-level evaluation:** `eval_every_steps` now evaluates at actual step intervals
   (not epoch-level), with proper early stopping.

4. **Dev/dev_test split:** `dev_test_ratio: 0.1` holds out 10% of labelled_dev for
   final evaluation, preventing overfitting through repeated dev evaluation.

5. **FEVER Score metric:** Official shared-task scoring (label correct AND sufficient
   evidence) for pipeline mode.

6. **Wiki cache:** Full 14,363-page SQLite cache (98.8% coverage of referenced pages).

7. **BM25 improvement:** Stopword removal + minimum token length for better retrieval.

8. **DeBERTa-v3 tokenizer:** Explicit `DebertaV2Tokenizer` (slow) to avoid
   tiktoken/protobuf conversion failures in transformers 4.46.x.

---

## Integrity

- Split hashes logged in every `report.json`
- Leakage guard enforced in `FeverPipelineDataset`
- CEGIS mines counterexamples from training set only
- Test set never touched
- All 193 unit tests pass

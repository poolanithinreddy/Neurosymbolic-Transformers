# NST — Comprehensive Code Review

**Repository:** `/Users/nithinreddy/Documents/Neurosymbolic Transformers/nst`  
**Date:** 2025-07-11  
**Reviewed by:** GitHub Copilot (Claude Opus 4.6)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Full Python File Inventory](#2-full-python-file-inventory)
3. [Eval Module](#3-eval-module)
4. [Data Module](#4-data-module)
5. [Models Module](#5-models-module)
6. [Training Module](#6-training-module)
7. [Symbolic Module](#7-symbolic-module)
8. [Logic / Decoding / Grounding / Retrieval](#8-supporting-modules)
9. [Tests](#9-tests)
10. [Scripts](#10-scripts)
11. [Documentation](#11-documentation)
12. [GroundedVerifier (API surface)](#12-grounded-verifier)
13. [Cross-Cutting Issues](#13-cross-cutting-issues)
14. [TODO/FIXME/HACK Audit](#14-todo-fixme-hack-audit)
15. [Dead Code](#15-dead-code)
16. [Import Cycle Analysis](#16-import-cycle-analysis)
17. [Test Coverage Assessment](#17-test-coverage-assessment)
18. [Prioritised Recommendations](#18-prioritised-recommendations)

---

## 1. Executive Summary

The NST codebase is a **research-quality PyTorch framework** implementing Neural CEGIS (Counterexample-Guided Inductive Synthesis) for neurosymbolic training. The primary application is FEVER fact verification using DeBERTa-v3 with symbolic constraints. The codebase spans **106 Python files** across 15 directories.

**Strengths:**
- Extremely well-documented (inline comments, docstrings, module headers)
- Clean separation of concerns (data/models/training/symbolic/eval)
- Three generations of architecture: v1 (heuristic constraints), v1-VERI (verification heads), v2 (learned heads)
- Defensive coding: NaN guards, leakage checks, split hashing
- Rich baseline suite (random replay, hard mining, same-budget controls)
- `GroundedVerifier` API is production-grade

**Critical Issues:**
1. **Massive code duplication**: `train_fever_veri.py` and `train_fever_nst.py` share ~60% identical code (config parsing, dataloader construction, eval loops)
2. **Hardcoded high-precision constraint indices** `{0, 3, 4}` in `nst_veri.py` predict() — fragile if constraints are reordered
3. **`eval/fliptest.py` is a complete stub** — dead code writing `{"status": "not-implemented"}`
4. **Deprecated API**: `scipy.stats.binom_test` used in `significance.py` (removed in SciPy 1.12)
5. **RESULTS.md mostly unfilled** — most cells are "—", suggesting incomplete experiment campaign

---

## 2. Full Python File Inventory

**106 Python files total:**

### Root (3)
| File | Purpose |
|------|---------|
| `__init__.py` | Package init, exports `GroundedVerifier` |
| `main.py` | Unified CLI entry point (~500 lines, 20+ subcommands) |
| `grounded_verifier.py` | Production API: NLI + constraints + gating + abstention |

### data/ (8)
| File | Purpose |
|------|---------|
| `__init__.py` | Empty |
| `digit_addition.py` | MNIST-based digit addition dataset |
| `multi_digit_addition.py` | Multi-digit variant with carry constraints |
| `kinship.py` | CLUTRR-style kinship relational reasoning |
| `clutrr.py` | Raw CLUTRR loading |
| `fever_dataset.py` | Core FEVER data loading (500+ lines, gold + pipeline modes) |
| `fever_wiki_cache.py` | SQLite cache for wiki pages (~250 lines) |
| `aliases.tsv`, `entities.tsv`, `facts.tsv`, `fever.tsv` | Static data files |

### models/ (6+)
| File | Purpose |
|------|---------|
| `fever_nli.py` | DeBERTa builder + `FeverNLIWrapper` |
| `nst_veri.py` | Flagship v1 model: backbone + VerificationHeads + ContrastiveHead + constraint fusion |
| `nst_veri_v2.py` | V2 redesign: learned ContradictionHead + EvidenceRelevanceHead + RecalibrationNetwork |
| `verification_heads.py` | K auxiliary binary classifiers with residual correction |
| `contrastive_head.py` | Supervised contrastive loss via class prototypes |
| `nst_multi_digit.py` | Multi-digit addition model (not reviewed in detail) |

### training/ (12+)
| File | Purpose |
|------|---------|
| `train_fever_nst.py` | 5-mode FEVER training (neural/soft/lagrangian/cegis/gated) |
| `train_fever_veri.py` | Flagship 3-phase NST-VERI training loop |
| `train_fever_veri_v2.py` | V2: learned multi-task training |
| `train_nst.py` | Digit addition training |
| `train_multi_digit.py` | Multi-digit training |
| `train_kinship.py` | Kinship training |
| `cegis.py` | Generic Neural CEGIS training loop |
| `baselines.py` | Random replay / hard mining / same-budget controls |
| `adaptive_lambda.py` | Per-sample constraint weighting network |
| `model_setup.py` | GPU-tier auto-configuration |
| `config_validation.py` | YAML safe_load numeric casting |
| `multi_seed.py` | Multi-seed runner for error bars |
| `pretrain_mnli.py` | MNLI pre-fine-tuning for FEVER transfer |
| `losses.py` | Simple loss utilities (BCE, classification, logic) |
| `cagrad.py` | Conflict-Averse Gradient descent (~15 lines) |
| `train.py` | Legacy T5 training |

### symbolic/ (8+)
| File | Purpose |
|------|---------|
| `constraints_v2.py` | 7-constraint engine (Numerical, Negation, EntityOverlap, EvidenceSufficiency, Temporal, HedgeModality, MutualExclusion) |
| `fever_constraint_loss.py` | V1 differentiable 5-constraint Horn-clause loss |
| `fever_constraints.py` | Structured fact extraction (regex-based) |
| `constraint_gating.py` | ECCG: per-sample per-constraint gates |
| `lagrangian.py` | Augmented Lagrangian dual-variable optimizer |
| `constraint_solver.py` | Soft constraints + Z3 hard verification |
| `rule_engine.py` | YAML rule loader + t-norm evaluator |
| `multi_digit_constraints.py` | Multi-digit-specific constraints |

### eval/ (12)
| File | Purpose |
|------|---------|
| `fever.py` | Unified FEVER evaluation CLI |
| `fever_metrics.py` | Accuracy, confusion matrix, FEVER score, error decomposition |
| `calibration_metrics.py` | ECE, Brier score, reliability diagrams |
| `calibration.py` | CLI wrapper around calibration_metrics |
| `temperature_scaling.py` | Post-hoc temperature scaling |
| `significance.py` | Bootstrap CI, McNemar's test |
| `leakage_check.py` | 4-check data leakage verification |
| `eval_nst.py` | Digit addition evaluation |
| `cogs.py` | COGS compositional generalization |
| `truthfulqa.py` | TruthfulQA evaluation |
| `fliptest.py` | **STUB — not implemented** |
| `rulecheck.py` | Rule satisfaction for digit/kinship tasks |

### decoding/ (3), logic/ (2+), grounding/ (1), retrieval/ (1)
| File | Purpose |
|------|---------|
| `decoding/hard_masks.py` | Logit masking for T5 generation (label-constrained) |
| `decoding/rerank.py` | LM + rule score mixing (3-line function) |
| `logic/logic.py` | Product t-norm primitives (neg, t_and, t_or, imply, horn_truth, horn_violation) |
| `grounding/__init__.py` | Empty |
| `retrieval/__init__.py` | Empty |

### tests/ (17), scripts/ (10+), configs/ (50+), colab/ (6)
See sections 9, 10, 11 below.

---

## 3. Eval Module

### 3.1 `eval/fever.py`
- **Purpose:** Unified FEVER evaluation — supports DeBERTa (primary) and T5 (legacy)
- **Key functions:** `evaluate_deberta()`, `evaluate_t5()`, `evaluate()`
- **Supports model_type:** `"nli"` (plain NLI wrapper) and `"veri"` (NST-VERI)
- **Issues:** None significant. Clean dispatch pattern.

### 3.2 `eval/fever_metrics.py`
- **Purpose:** Core FEVER metrics
- **Key functions:** `label_accuracy()`, `confusion_matrix()`, `retrieval_recall_at_k()`, `fever_score()` (official shared-task metric), `error_decomposition()`, `integrity_check()`, `format_results_table()`
- **Issues:** None. Well-structured, comprehensive.

### 3.3 `eval/calibration_metrics.py`
- **Purpose:** Calibration measurement
- **Key functions:** `expected_calibration_error()` (15-bin ECE), `brier_score()`, `reliability_diagram_data()`, `noise_robustness_eval()`, `collect_calibration_data()`
- **Issues:** None. Clean implementation.

### 3.4 `eval/temperature_scaling.py`
- **Purpose:** Post-hoc temperature scaling via LBFGS
- **Key classes:** `TemperatureScaler`
- **Key functions:** `learn_temperature()`, `apply_temperature()`
- **⚠ Hardcoded value:** Initial temperature set to **1.5** (not the standard 1.0). This biases the optimizer; however, LBFGS typically converges regardless. Worth documenting the rationale.

### 3.5 `eval/significance.py`
- **Purpose:** Statistical significance testing
- **Key functions:** `bootstrap_ci()`, `paired_bootstrap()`, `mcnemars_test()`, `aggregate_seeds()`, `accuracy_with_ci()`, `ece_with_ci()`
- **🐛 Bug:** Uses `scipy.stats.binom_test` (line 155), which was **deprecated in SciPy 1.7** and **removed in SciPy 1.12**. Must migrate to `scipy.stats.binomtest`.

### 3.6 `eval/fliptest.py`
- **🔴 Dead code.** Entire implementation is a stub:
  ```python
  rep = {"status": "not-implemented"}
  ```
  Should be either implemented or deleted.

### 3.7 `eval/truthfulqa.py`
- **Purpose:** TruthfulQA evaluation of T5 models
- **⚠ Weak metric:** `score_truthfulness()` uses crude token-overlap heuristic. For any token from the reference appearing in the prediction, it counts as a "hit". This is a poor proxy for factual correctness. Consider BERTScore or entailment-based scoring.
- **Design note:** This is a T5-only evaluation; irrelevant for the DeBERTa-based FEVER pipeline.

### 3.8 `eval/rulecheck.py`
- **Purpose:** Rule-satisfaction checking for digit_add and kinship
- **Limitation:** Does **not** handle FEVER task at all. Only useful for the non-FEVER experiments.

### 3.9 `eval/cogs.py`
- **Purpose:** COGS compositional generalization evaluation
- **Limitation:** T5-only. Loads TSV, batch-generates, computes exact match + token-level F1.

### 3.10 `eval/leakage_check.py`
- **Purpose:** 4 checks: split disjointness, fuzzy claim overlap (n-gram Jaccard), constraint independence, full leakage report
- **Issues:** None. Thorough and well-designed.

### 3.11 `eval/eval_nst.py`
- **Purpose:** Digit addition evaluation with IID + compositional splits + Z3 hard-constraint support
- **Issues:** None.

### 3.12 `eval/calibration.py`
- **Purpose:** CLI wrapper around calibration_metrics
- **⚠ Mostly placeholder:** Main function loads from checkpoint if available, but implementation is skeletal.

---

## 4. Data Module

### 4.1 `data/fever_dataset.py` (~500 lines)
- **Purpose:** Core FEVER data loading—the most critical data file
- **Key constants:** `FEVER_LABELS`, `LABEL2ID`, `ID2LABEL`, `NUM_LABELS=3`
- **Key functions:**
  - `_normalise_label()` — handles all label variants (SUPPORTED→SUPPORTS, etc.)
  - `_build_wiki_page_map()` — SQLite-first, JSONL fallback
  - `load_fever_splits()` — HF datasets loading, handles flat-row and nested evidence formats
  - `_backfill_nei_evidence()` — prevents "no evidence → NEI" shortcut (leakage guard)
  - `_concat_evidence_sentences()` — resolves evidence via wiki page map
- **Key classes:** `FeverGoldDataset`, `FeverPipelineDataset` (with BM25 retrieval), `_WikiCacheAdapter`
- **Strengths:**
  - Sophisticated evidence resolution pipeline
  - Split hashing for reproducibility
  - Explicit leakage prevention
- **⚠ Complexity risk:** The `load_fever_splits()` function handles multiple HF dataset formats (flat, nested, varying field names) with multiple fallback paths. Fragile if HF changes dataset schemas.

### 4.2 `data/fever_wiki_cache.py` (~250 lines)
- **Purpose:** SQLite-backed O(1) wiki page cache
- **Key functions:** `build_wiki_cache()` (two-pass: scan annotations → stream wiki_pages → SQLite)
- **Design rationale:** Solves Colab OOM by avoiding loading all wiki pages into RAM
- **Issues:** None. Clean SQLite usage with WAL mode.

### 4.3 Other data files
- `digit_addition.py`, `multi_digit_addition.py`, `kinship.py`, `clutrr.py` — standard dataset classes for non-FEVER tasks. Not reviewed in detail.

---

## 5. Models Module

### 5.1 `models/fever_nli.py`
- **Purpose:** Model construction + inference wrapper
- **Key functions:** `build_fever_model()` — builds DeBERTa + tokenizer with LoRA/gradient checkpointing
- **Key classes:** `FeverNLIWrapper` — wraps HF SequenceClassification model
  - `forward()` returns dict with logits, probs, loss
  - `predict()` returns labels + probs
  - `get_label_probs()` for calibration
- **⚠ Minor:** Handles class_weight dtype mismatch explicitly (float32 cast). Robust.

### 5.2 `models/nst_veri.py` (Flagship V1)
- **Purpose:** Main model combining backbone + verification heads + contrastive head + constraint fusion
- **Key class:** `NSTVeriModel`
  - `forward()` computes 4 losses: NLI, auxiliary verification, contrastive, constraint KL
  - `predict()` has inference-time constraint fusion
- **🔴 Hardcoded indices:**
  ```python
  HIGH_PRECISION = {0, 3, 4}  # Numerical, EvidenceSufficiency, Temporal
  ```
  These indices assume a specific constraint ordering in `ConstraintEngineV2`. **If constraints are reordered or added, this silently breaks.** Should reference constraints by name.
- **⚠ KL loss clamped at 10.0** — reasonable but undocumented rationale.
- **⚠ `predict()` constraint alpha-blending:** Uses `constraint_alpha=0.15` default. This is a hyperparameter baked into inference; ideally it should come from config.

### 5.3 `models/nst_veri_v2.py` (V2 Redesign)
- **Purpose:** Replaces heuristic constraints with learned neural heads
- **Key classes:** `AttentionPool`, `ContradictionHead`, `EvidenceRelevanceHead`, `RecalibrationNetwork`, `ContrastiveHead`, `NSTVeriModelV2`
- **Architecture:** NLI backbone → [Contradiction, Relevance, Contrastive] heads → RecalibrationNetwork → final logits
- **Key design decisions:**
  - Silver labels: REFUTES→contradiction=1, SUPPORTS/NEI→contradiction=0
  - RecalibrationNetwork starts at near-zero scale (σ(-3.0) ≈ 0.047)
  - R-Drop: two forward passes with different dropout masks
- **Strengths:** Much cleaner than v1; no dependency on regex-based constraint engine at training time
- **⚠ R-Drop cost:** When `use_rdrop=True`, training does **2× forward passes** per batch. This doubles compute but is a deliberate design choice documented in the module header.

### 5.4 `models/verification_heads.py`
- **Purpose:** K auxiliary binary classifiers predicting symbolic properties from [CLS]
- **Key class:** `VerificationHeads`
  - K independent heads (Linear → ReLU → Linear → Sigmoid)
  - Residual correction with zero-initialized scale
  - `verification_loss()` uses masked BCE (only computes loss where constraint signal fires)
- **Issues:** None. Clean design.

### 5.5 `models/contrastive_head.py`
- **Purpose:** Supervised contrastive learning via class prototypes
- **Key class:** `ContrastiveConstraintHead`
  - Learnable prototype vectors (one per class)
  - Two-layer projection head → cosine similarity → temperature scaling
- **Issues:** None. Standard approach.

---

## 6. Training Module

### 6.1 `training/train_fever_veri.py` (Flagship, ~500 lines)
- **Purpose:** 3-phase NST-VERI training loop
- **Key components:**
  - `FocalLoss` class (γ=2.0 default)
  - `_get_phase()` — Phase 1 (20%): NLI+aux, Phase 2 (20%): +contrastive, Phase 3 (60%): +constraints
  - `_constraint_warmup()` — cosine warmup schedule within Phase 3
  - `train_fever_veri()` — main entry
- **🔴 Config parsing duplication:** ~150 lines of config extraction (nested/flat key resolution, type casting, default values) are nearly identical to `train_fever_nst.py`. This should be refactored into a shared `_parse_training_config()` function.
- **🔴 Historical bug documented in RESULTS.md:** `_constraint_warmup(epoch=2, epochs=5)` returned 0.0 at Phase 3 start, meaning constraints never activated before early stopping fired. Fix was applied but indicates the warmup schedule was fragile.
- **⚠ Multiple learning rate groups:** lr, lr_lora, lr_heads, lr_gate — 4 separate LRs. Complex but justified for different component types.

### 6.2 `training/train_fever_nst.py` (~400 lines)
- **Purpose:** 5-mode FEVER training (neural/soft/lagrangian/cegis/gated)
- **Key function:** `train_fever_nst()`
- **Key internal:** `_mine_counterexamples()` for CEGIS mode
- **🔴 Code duplication:** Shares ~60% of code with `train_fever_veri.py` — config parsing, dataloader construction, eval loops, optimiser setup. Should share a common training infrastructure.

### 6.3 `training/train_fever_veri_v2.py`
- **Purpose:** V2 multi-task training
- **Key design:** Simpler 2-phase schedule (warmup → full multi-task + R-Drop)
- **Strengths:** Cleaner than v1; less config surface area
- **⚠ Still has similar config-parsing boilerplate.

### 6.4 `training/cegis.py` (~300 lines)
- **Purpose:** Generic Neural CEGIS trainer
- **Key classes:** `CEGISConfig`, `CEGISLog`, `CEGISTrainer`
- **Design:** Generic over model and domain — requires `verify_fn` and `ce_to_dataset_fn` callbacks. Clean separation.
- **Issues:** None significant.

### 6.5 `training/baselines.py`
- **Purpose:** Three controlled baselines for fair comparison
  1. **Random Replay** — random samples instead of counterexamples (controls for data augmentation)
  2. **Hard Example Mining** — highest-loss examples (controls for curriculum learning)
  3. **Same Budget** — extra epochs without replay (controls for compute)
- **Strengths:** These are essential for academic credibility. Well-designed.
- **Issues:** None.

### 6.6 `training/adaptive_lambda.py`
- **Purpose:** Per-sample constraint weighting network
- **Key class:** `AdaptiveLambdaModule`
  - Input: constraint_fires(K) + confidence(K) + entropy(1) + max_conf(1) + agreement(1)
  - Output: lambda_per_sample(B,) and gate_weights(B,K)
- **⚠ Init comment mismatch:** Comment says "sigmoid(0.0) = 0.5, so lambda starts at ~0.5 × lambda_max" but then says "Initialised to output near-zero λ (start unconstrained)". These contradict. The actual init `bias=0.0` → sigmoid(0)=0.5 → λ=0.25 (for lambda_max=0.5). This is moderate, not "near-zero".

### 6.7 `training/model_setup.py`
- **Purpose:** GPU-tier auto-configuration
- **Key functions:** `detect_gpu_tier()`, `setup_gpu_optimizations()`, `build_nst_veri_model()`
- **Tier config:** H100/A100-80GB → large+LoRA r=16 bs=128; T4-16GB → base+LoRA r=8 bs=16; MPS/CPU → base full FT bs=8
- **Issues:** None. Very practical for reproducibility across hardware.

### 6.8 `training/config_validation.py`
- **Purpose:** YAML safe_load numeric fix (YAML parses `1e-3` as string, not float)
- **Key functions:** `cast_value()`, `validate_section()`, `load_and_validate_config()`
- **Issues:** None. Solves a very real YAML footgun.

### 6.9 `training/multi_seed.py`
- **Purpose:** Multi-seed runner for error bars
- **Issues:** None. Standard implementation.

### 6.10 `training/pretrain_mnli.py`
- **Purpose:** MNLI pre-fine-tuning with FEVER-aligned label mapping
- **Label mapping:** MNLI(entailment→SUPPORTS, neutral→NEI, contradiction→REFUTES)
- **Issues:** None. Clean transfer learning approach.

### 6.11 `training/losses.py`
- **Purpose:** Minimal loss utilities
- **Functions:** `task_loss_classification()`, `concept_bce()`, `logic_loss()`
- **Issues:** Very thin — only 18 lines. Could be inlined where used.

### 6.12 `training/cagrad.py`
- **Purpose:** Conflict-Averse Gradient descent
- **Issues:** Only ~15 lines. Minimal implementation — illustrative, not production. The conflict threshold `c=0.3` is hardcoded.

---

## 7. Symbolic Module

### 7.1 `symbolic/constraints_v2.py` (~700 lines)
- **Purpose:** 7-constraint probabilistic engine (THE core symbolic component)
- **Key classes:** `NumericalConstraint`, `NegationConstraint`, `EntityOverlapConstraint`, `EvidenceSufficiencyConstraint`, `TemporalConstraint`, `HedgeModalityConstraint`, `MutualExclusionConstraint`
- **Orchestrator:** `ConstraintEngineV2.evaluate_batch()` → returns fires/confidence/direction tensors
- **Design philosophy:** "PRECISION > RECALL" — constraints only fire when confident
- **⚠ `min_evidence_words=8`** gate skips title-only evidence. Good design but hardcoded.
- **⚠ Regex-based extraction:** All constraints use regex patterns for signal extraction (dates, numbers, negation cues, entities). This is by design (lightweight, no model deps) but inherently noisy.

### 7.2 `symbolic/fever_constraint_loss.py`
- **Purpose:** V1 constraint system — 5 differentiable Horn-clause constraints using product t-norm
- **Key functions:** `fever_constraint_loss()`, `verify_fever_constraints()`
- **Issues:** This is the **v1 predecessor** to `constraints_v2.py`. Both coexist in the codebase. Not dead code — still used by `train_fever_nst.py` modes.

### 7.3 `symbolic/fever_constraints.py`
- **Purpose:** Structured fact extraction from claim+evidence text
- **Key class:** `StructuredFacts` dataclass
- **Key function:** `extract_structured_facts()` — regex-based extraction of numbers, dates, negation, entities
- **Issues:** None. Clean extraction with well-documented patterns.

### 7.4 `symbolic/constraint_gating.py`
- **Purpose:** ECCG (Evidence-Conditioned Constraint Gating)
- **Key class:** `ConstraintGate` — lightweight gating network (<1K params)
- **Key function:** `gated_fever_constraint_loss()` — applies learned gates to v1 constraint violations
- **Constants:** `_N_SIGNALS = 7`, `_N_CONSTRAINTS = 5`
- **Issues:** None. Well-designed with clear documentation of the novel concept.

### 7.5 `symbolic/lagrangian.py`
- **Purpose:** Augmented Lagrangian dual-variable optimizer
- **Key classes:** `LagrangianState`, `MultiConstraintLagrangian`
- **Key functions:** `lagrangian_loss()`, `update_dual_variable()`, `price_of_logic()`, `save_lambda_trajectory()`
- **Issues:** None. Clean, well-documented implementation with convergence theory references.

### 7.6 `symbolic/constraint_solver.py`
- **Purpose:** Soft differentiable constraints (digit addition) + Z3 hard verification
- **Key functions:** `sum_constraint_soft()` (discrete convolution for expected sum), `hard_constraint_verify()` (Z3 SAT), `hard_constraint_batch()`
- **⚠ Z3 optional dependency:** Graceful fallback to arithmetic repair if Z3 not installed.

### 7.7 `symbolic/rule_engine.py`
- **Purpose:** YAML rule loading + t-norm evaluation
- **Key functions:** `load_rules()`, `evaluate_rule()`, `evaluate_all_rules()`
- **Issues:** None. Generic and reusable.

---

## 8. Supporting Modules

### 8.1 `logic/logic.py`
- **Purpose:** Core fuzzy logic primitives
- **Functions:** `neg()`, `t_and()`, `t_or()`, `imply()` (Reichenbach), `prod_many()`, `horn_truth()`, `horn_violation()`
- **Issues:** None. Short (45 lines), correct. Uses log-space product for numerical stability.

### 8.2 `decoding/hard_masks.py`
- **Purpose:** Constrain T5 generation logits to label tokens only
- **Key class:** `HardMaskProcessor` (extends `LogitsProcessor`)
- **Issues:** Legacy T5 code. Includes graceful fallback if `transformers` is not available.

### 8.3 `decoding/rerank.py`
- **Purpose:** Score combination for reranking
- **Issues:** Trivial 3-line function. Could be inlined.

### 8.4 `grounding/__init__.py`, `retrieval/__init__.py`
- **Purpose:** Empty placeholder packages
- **Issues:** These directories have no content beyond `__init__.py`. Likely placeholders for future work.

---

## 9. Tests

**17 test files, ~232 expected passing tests (per RUNBOOK.md):**

| Test File | Coverage Area | Approx Lines |
|-----------|--------------|------|
| `test_fever.py` | Label mapping, constraint extraction, loss, hard verification, eval metrics, split hash, BM25 | ~260 |
| `test_constraints_v2.py` | All 7 constraints individually + batch API shapes | ~130 |
| `test_calibration.py` | ECE, Brier, reliability diagram | ~100 |
| `test_config_validation.py` | YAML casting, range validation, NaN safety, CSR under NaN | ~170 |
| `test_fever_io.py` | fever.tsv schema validation (requires local data) | ~40 |
| `test_cegis.py` | CEGIS training loop | — |
| `test_eccg.py` | ECCG constraint gating | — |
| `test_infrastructure.py` | Import checks for all modules | — |
| `test_kinship.py` | Kinship dataset + model | — |
| `test_lagrangian.py` | Lagrangian dual variable | — |
| `test_logic.py` | Fuzzy logic primitives | — |
| `test_masks.py` | Hard mask logit processing | — |
| `test_multi_digit.py` | Multi-digit dataset + constraints | — |
| `test_rules.py` | Rule engine | — |
| `test_symbolic.py` | Symbolic solver | — |
| `test_wiki_cache.py` | Wiki cache build/query | — |
| `conftest.py` | Adds project root to sys.path | ~5 |

---

## 10. Scripts

| Script | Purpose |
|--------|---------|
| `scripts/verify_no_leakage.py` | 6-check data integrity verification |
| `scripts/benchmark_latency.py` | Benchmarks 4 inference modes (neural/soft/lagrangian/hard-Z3) |
| `scripts/export_tables.py` | Results table generation |
| `scripts/plot_alignment.py` | Lambda trajectory + constraint alignment plots |
| `scripts/get_fever.py` | FEVER data download helper |
| `scripts/get_cogs.py` | **Stub** — only `# TODO: download COGS dataset to data/cogs/` |
| `scripts/get_truthfulqa.py` | TruthfulQA download |
| `scripts/fever_sanity_samples.py` | Sample inspection |
| `scripts/synth_make.py` | Synthetic data generation |
| `scripts/normalize_line_endings.py` | Line ending normalization |

---

## 11. Documentation

### 11.1 `RESULTS.md`
- **Purpose:** Experiment tracking dashboard
- **Issue:** Most full-run experiments have **"—" (unfilled)** entries. Only the Neural baseline (0.8378 accuracy) and NST-VERI v1 single-seed result are documented.
- **Critical note at v1 result:** "Bug: constraints never fired" — the `_constraint_warmup()` returned 0.0 at Phase 3 start, so the result was purely neural backbone contribution.
- **Recommendation:** Consider using Weights & Biases or MLflow for automated experiment tracking rather than manual markdown.

### 11.2 `RUNBOOK.md`
- **Purpose:** Step-by-step reproduction instructions. 232 tests expected.
- **Issues:** None. Clear and comprehensive.

### 11.3 `README.md`
- **Purpose:** Project overview with `GroundedVerifier` quick-start, architecture diagram, Neural CEGIS explanation
- **Issues:** None. Well-written.

### 11.4 `PAPER.md`
- **Purpose:** Full academic paper draft
- **Sections covered:** Abstract, Introduction, Related Work, Method (3.1–3.6), ECCG, GroundedVerifier API
- **Issues:** None from code review perspective. Mathematical notation is correct.

---

## 12. GroundedVerifier

`grounded_verifier.py` is the **production-quality API surface** of this project.

**Architecture:**
1. NLI backbone (any HuggingFace model)
2. ConstraintEngineV2 (7 symbolic constraints)
3. ConstraintGate (ECCG per-sample gating)
4. Abstention logic (low confidence or evidence insufficiency)

**Strengths:**
- Drop-in: `from nst import GroundedVerifier`
- MPS/CUDA/CPU auto-detection
- `verify()` / `verify_batch()` API with structured `VerificationResult`
- `from_model()` factory for wrapping existing models
- Serialization: `save()` / `load()` with gate weights
- `benchmark_latency()` with overhead measurement

**Issues:**
- **No `load()` classmethod** — `save()` exists but there is no corresponding `load()` factory to reconstruct from disk. The user would have to manually reconstruct.
- `verify_batch()` calls `self._engine.evaluate_batch()` then loops over results with Python for-loops. For large batches this is slow — but the constraint engine is regex-based so vectorization is limited anyway.

---

## 13. Cross-Cutting Issues

### 13.1 Massive Training Code Duplication
`train_fever_veri.py`, `train_fever_nst.py`, and `train_fever_veri_v2.py` all contain nearly identical:
- Config parsing (~100-150 lines each)
- Dataloader construction
- Evaluation loops
- Optimizer setup
- Device auto-detection

**Recommendation:** Extract common infrastructure into a `training/fever_common.py` module.

### 13.2 Hardcoded Constants in Multiple Locations
| Constant | Location | Value | Risk |
|----------|----------|-------|------|
| HIGH_PRECISION indices | `nst_veri.py:predict()` | `{0, 3, 4}` | Breaks if constraints reordered |
| min_evidence_words | `constraints_v2.py` | 8 | Reasonable but not configurable |
| abstain_threshold | `grounded_verifier.py` | 0.4 | Configurable via constructor ✓ |
| Temperature init | `temperature_scaling.py` | 1.5 | Unusual; standard is 1.0 |
| FocalLoss gamma | `train_fever_veri.py` | 2.0 | Standard default, OK |
| cagrad c | `cagrad.py` | 0.3 | Hardcoded, not configurable |

### 13.3 _auto_device() Duplicated Everywhere
The `_auto_device()` function is copy-pasted in at least 6 files:
- `training/train_fever_veri.py`
- `training/train_fever_nst.py`
- `training/train_fever_veri_v2.py`
- `training/cegis.py`
- `training/baselines.py`
- `grounded_verifier.py`

Should be a single utility function (e.g., in `training/utils.py`).

### 13.4 sys.path Manipulation
Many files contain:
```python
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
```
This is a packaging smell. With `pyproject.toml` and proper `pip install -e .`, this should be unnecessary. It's present in ~15 files.

---

## 14. TODO/FIXME/HACK Audit

**Only 1 TODO found in the entire codebase:**

| File | Line | Content |
|------|------|---------|
| `scripts/get_cogs.py` | 1 | `# TODO: download COGS dataset to data/cogs/` |

This is impressively clean. No FIXME or HACK comments found anywhere.

---

## 15. Dead Code

| File | Status | Evidence |
|------|--------|----------|
| `eval/fliptest.py` | **100% dead** | Stub that only writes `{"status": "not-implemented"}` |
| `eval/cogs.py` | **Likely dead** | T5-only; no COGS training pipeline exists in current codebase |
| `eval/truthfulqa.py` | **Likely dead** | T5-only; weak metric; no integration with DeBERTa pipeline |
| `eval/rulecheck.py` | **Partially dead** | Only used for digit_add/kinship; no FEVER support |
| `decoding/rerank.py` | **Likely dead** | 3-line function; no imports found in codebase |
| `grounding/__init__.py` | **Dead** | Empty package with no content |
| `retrieval/__init__.py` | **Dead** | Empty package with no content |
| `scripts/get_cogs.py` | **Dead** | Only contains a TODO comment |
| `training/losses.py` | **Low value** | 18 lines; trivial wrappers that could be inlined |
| `training/cagrad.py` | **Near-dead** | 15-line illustrative implementation; only used if explicitly configured |

---

## 16. Import Cycle Analysis

**No circular imports detected.** The codebase follows a clean, acyclic dependency graph:

```
logic/ ← symbolic/ ← models/ ← training/
                   ← eval/
data/  ← models/  ← training/
                   ← eval/
```

The `grounded_verifier.py` uses lazy imports (`_init_constraints()`) to avoid heavy startup dependencies, which is good practice.

All `sys.path.insert()` calls are forward references (child importing parent), not cycles.

---

## 17. Test Coverage Assessment

### Well-Tested Areas
- ✅ FEVER label mapping and constraint extraction (`test_fever.py`)
- ✅ All 7 v2 constraints individually (`test_constraints_v2.py`)
- ✅ Calibration metrics (ECE, Brier) (`test_calibration.py`)
- ✅ Config YAML parsing and NaN safety (`test_config_validation.py`)
- ✅ Lagrangian dual variable updates (`test_lagrangian.py`)
- ✅ Fuzzy logic primitives (`test_logic.py`)
- ✅ CEGIS loop convergence (`test_cegis.py`)
- ✅ ECCG gating (`test_eccg.py`)

### Under-Tested Areas
- ❌ `GroundedVerifier` — no dedicated unit tests for the production API
- ❌ `NSTVeriModel.predict()` constraint fusion logic
- ❌ `NSTVeriModelV2` full forward/backward pass
- ❌ `AdaptiveLambdaModule` forward pass
- ❌ `RecalibrationNetwork` learned fusion
- ❌ `train_fever_veri.py` 3-phase schedule transitions
- ❌ `pretrain_mnli.py` label mapping correctness
- ❌ Temperature scaling convergence
- ❌ Multi-seed aggregation
- ❌ `benchmark_latency.py` runtime

### Tests Requiring External Data
- `test_fever_io.py` requires `data/fever.tsv` to exist locally
- `test_wiki_cache.py` may require network access for HF datasets

---

## 18. Prioritised Recommendations

### P0 — Fix Now (Correctness / Breakage Risk)

1. **Replace deprecated `binom_test`** in `eval/significance.py:155`:
   ```python
   # Before (deprecated, removed in SciPy 1.12):
   from scipy.stats import binom_test
   p_value = binom_test(b, b + c, 0.5)
   
   # After:
   from scipy.stats import binomtest
   p_value = binomtest(b, b + c, 0.5).pvalue
   ```

2. **Replace hardcoded constraint indices** in `models/nst_veri.py`:
   ```python
   # Before:
   HIGH_PRECISION = {0, 3, 4}
   
   # After:
   HIGH_PRECISION_NAMES = {"NumericalConstraint", "EvidenceSufficiencyConstraint", "TemporalConstraint"}
   high_prec_idx = {i for i, name in enumerate(engine.constraint_names) if name in HIGH_PRECISION_NAMES}
   ```

### P1 — Refactor (Maintainability)

3. **Extract shared training infrastructure** from `train_fever_veri.py`, `train_fever_nst.py`, `train_fever_veri_v2.py` into a common module. Target: eliminate ~300 lines of duplicated config parsing, dataloader construction, and eval loops.

4. **Consolidate `_auto_device()`** into a single utility function in `training/utils.py` (or a top-level `utils.py`).

5. **Remove `sys.path.insert()` hacks** (~15 files). The project has `pyproject.toml` — run `pip install -e .` and use proper package imports.

### P2 — Clean Up (Dead Code / Quality)

6. **Delete `eval/fliptest.py`** — 100% stub, no value.

7. **Delete `scripts/get_cogs.py`** — only a TODO comment.

8. **Delete empty packages** `grounding/` and `retrieval/` (or add content).

9. **Add `GroundedVerifier.load()` classmethod** to complement `save()`.

10. **Document temperature_scaling init=1.5** rationale, or change to standard 1.0.

### P3 — Enhance (Nice to Have)

11. **Add tests for `GroundedVerifier`** — the production API has zero test coverage.

12. **Fix `adaptive_lambda.py` docstring** — "near-zero" contradicts actual init of 0.5×lambda_max.

13. **Replace `truthfulqa.py` scoring** with a proper metric (BERTScore or entailment) if the eval is still used.

14. **Automate experiment tracking** — RESULTS.md with manual "—" entries will never be complete. Use W&B/MLflow.

---

*End of review. Total files analyzed: 80+ Python files, 4 documentation files, 50+ config files. No security vulnerabilities identified (no web endpoints, no user input deserialization, no credential handling).*

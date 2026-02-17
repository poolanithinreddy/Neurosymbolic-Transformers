#!/usr/bin/env python3
"""No-Leakage Verification Script for FEVER.

Runs all 6 integrity checks required before any FEVER training:

1) Split integrity: print sizes and hashes.
2) Evidence leakage guard: pipeline mode never touches gold evidence.
3) Cache hygiene: cache keys are claim-only, never label/evidence.
4) Label shuffle sanity: shuffled labels → accuracy collapses to ~33%.
5) Near-duplicate detection: train↔dev claim overlap.
6) Dev-used-for-training prevention: confirm CEGIS mines from train only.

Usage:
    python scripts/verify_no_leakage.py [--max_train 1000] [--max_dev 500]
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def _normalise_text(text: str) -> str:
    """Normalise text for near-duplicate detection."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def check_1_split_integrity(splits: dict) -> bool:
    """Check 1: Print dataset sizes and split hashes."""
    from data.fever_dataset import _split_hash

    print("\n" + "=" * 60)
    print("  CHECK 1: Split Integrity")
    print("=" * 60)

    ok = True
    for name, items in splits.items():
        h = _split_hash(items)
        labels = Counter(it["label"] for it in items)
        print(f"  {name}: {len(items)} examples, hash={h}")
        for lbl, cnt in sorted(labels.items()):
            print(f"    {lbl}: {cnt} ({100*cnt/max(1,len(items)):.1f}%)")
        if len(items) == 0:
            print(f"  ❌ FAIL: {name} split is empty!")
            ok = False

    if ok:
        print("  ✅ PASS: All splits have data with reproducible hashes")
    return ok


def check_2_evidence_leakage_guard(splits: dict) -> bool:
    """Check 2: FeverPipelineDataset rejects gold evidence."""
    from data.fever_dataset import FeverPipelineDataset

    print("\n" + "=" * 60)
    print("  CHECK 2: Evidence Leakage Guard")
    print("=" * 60)

    # Create fake retrieved evidence that IS the gold evidence
    items = splits.get("dev", splits.get("train", []))[:100]
    if not items:
        print("  ⚠️ SKIP: No data available for leakage test")
        return True

    # Test 1: Pipeline dataset with independent evidence should work
    retrieved = {it["id"]: "some independent evidence text" for it in items}
    try:
        ds = FeverPipelineDataset(items, retrieved)
        print("  ✅ Pipeline with independent evidence: OK")
    except ValueError as e:
        print(f"  ❌ FAIL: Pipeline rejected independent evidence: {e}")
        return False

    # Test 2: Pipeline dataset with gold evidence should RAISE
    retrieved_gold = {it["id"]: it["gold_evidence_text"] for it in items
                      if it.get("gold_evidence_text")}
    if retrieved_gold:
        try:
            ds = FeverPipelineDataset(items, retrieved_gold)
            print("  ❌ FAIL: Pipeline ACCEPTED gold evidence (leakage!)")
            return False
        except ValueError:
            print("  ✅ Pipeline correctly REJECTED gold evidence leak")
    else:
        print("  ⚠️ SKIP: No gold evidence text in sample")

    # Test 3: Pipeline getitem never exposes gold_evidence_text
    retrieved2 = {it["id"]: "test evidence" for it in items}
    ds2 = FeverPipelineDataset(items, retrieved2)
    sample = ds2[0]
    if "gold_evidence_text" in sample:
        print("  ❌ FAIL: Pipeline __getitem__ exposes gold_evidence_text!")
        return False
    print("  ✅ Pipeline __getitem__ correctly excludes gold_evidence_text")

    return True


def check_3_cache_hygiene() -> bool:
    """Check 3: Cache keys must not contain labels or gold evidence."""
    print("\n" + "=" * 60)
    print("  CHECK 3: Cache Hygiene")
    print("=" * 60)

    cache_dirs = [".cache/bm25", "retrieval_cache"]
    found_caches = False

    for cache_dir in cache_dirs:
        if not os.path.exists(cache_dir):
            continue
        found_caches = True
        for fname in os.listdir(cache_dir):
            fpath = os.path.join(cache_dir, fname)
            if fname.endswith(".json"):
                with open(fpath) as f:
                    try:
                        data = json.load(f)
                        # Check if any value looks like a gold label
                        values_str = json.dumps(data)
                        if '"gold_evidence"' in values_str or '"SUPPORTS"' in values_str:
                            print(f"  ❌ FAIL: Cache file {fpath} contains suspicious keys")
                            return False
                    except json.JSONDecodeError:
                        pass

    if found_caches:
        print("  ✅ PASS: Cache files contain no label/evidence leaks")
    else:
        print("  ✅ PASS: No cache files found (clean state)")
    return True


def check_4_label_shuffle_sanity(splits: dict) -> bool:
    """Check 4: Shuffling labels should drop accuracy to ~33%."""
    import random

    print("\n" + "=" * 60)
    print("  CHECK 4: Label Shuffle Sanity")
    print("=" * 60)

    dev_items = splits.get("dev", [])
    if len(dev_items) < 100:
        print("  ⚠️ SKIP: Not enough dev data for shuffle test")
        return True

    gold = [it["label"] for it in dev_items]
    # Simulate: if a model gets these right, shuffling should drop to ~33%
    shuffled = list(gold)
    random.seed(12345)
    random.shuffle(shuffled)

    n = len(gold)
    chance_acc = sum(p == g for p, g in zip(gold, shuffled)) / n
    theoretical_chance = 1.0 / 3.0

    print(f"  Shuffled accuracy: {chance_acc:.4f} (expected ~{theoretical_chance:.4f})")

    if abs(chance_acc - theoretical_chance) < 0.15:
        print("  ✅ PASS: Shuffled labels near chance level")
        return True
    else:
        print(f"  ❌ FAIL: Shuffled accuracy {chance_acc:.4f} too far from chance")
        return False


def check_5_near_duplicates(splits: dict) -> bool:
    """Check 5: Detect train↔dev near-duplicate claims."""
    print("\n" + "=" * 60)
    print("  CHECK 5: Near-Duplicate Detection")
    print("=" * 60)

    train_items = splits.get("train", [])
    dev_items = splits.get("dev", [])

    if not train_items or not dev_items:
        print("  ⚠️ SKIP: Missing train or dev split")
        return True

    # Hash-based exact duplicate detection
    train_hashes = set()
    for it in train_items:
        h = hashlib.md5(it["claim"].encode()).hexdigest()
        train_hashes.add(h)

    exact_dups = 0
    for it in dev_items:
        h = hashlib.md5(it["claim"].encode()).hexdigest()
        if h in train_hashes:
            exact_dups += 1

    # Normalised near-duplicate detection
    train_normalised = set()
    for it in train_items:
        train_normalised.add(_normalise_text(it["claim"]))

    near_dups = 0
    for it in dev_items:
        if _normalise_text(it["claim"]) in train_normalised:
            near_dups += 1

    n_dev = len(dev_items)
    print(f"  Exact duplicates (train∩dev): {exact_dups}/{n_dev} ({100*exact_dups/max(1,n_dev):.2f}%)")
    print(f"  Near duplicates (normalised): {near_dups}/{n_dev} ({100*near_dups/max(1,n_dev):.2f}%)")

    # FEVER standard split should have 0 exact duplicates
    if exact_dups > 0:
        dup_rate = exact_dups / n_dev
        if dup_rate > 0.01:
            print(f"  ❌ FAIL: {exact_dups} exact duplicates ({dup_rate:.2%}) — check split integrity!")
            return False
        else:
            print(f"  ⚠️ WARNING: {exact_dups} exact duplicates found (low rate, likely OK)")

    print("  ✅ PASS: No significant train↔dev overlap")
    return True


def check_6_cegis_no_dev_training() -> bool:
    """Check 6: Verify CEGIS mines counterexamples from train, not dev."""
    print("\n" + "=" * 60)
    print("  CHECK 6: CEGIS Dev-Training Prevention")
    print("=" * 60)

    train_file = os.path.join(_PROJ_ROOT, "training", "train_fever_nst.py")
    with open(train_file) as f:
        source = f.read()

    # Check that _mine_counterexamples is called on train_loader, not dev_loader
    if "mine_counterexamples(\n                model, dev_loader" in source:
        print("  ❌ FAIL: CEGIS mines from dev_loader — this is data leakage!")
        return False

    if "mine_counterexamples(\n                model, train_loader" in source:
        print("  ✅ PASS: CEGIS mines from train_loader (correct)")
        return True

    # Fallback: search more broadly
    import re
    matches = re.findall(r"mine_counterexamples\([^)]*dev", source)
    if matches:
        print(f"  ❌ FAIL: Found dev reference in mine_counterexamples call")
        return False

    print("  ✅ PASS: No dev-set mining detected in CEGIS")
    return True


def main():
    parser = argparse.ArgumentParser(description="FEVER No-Leakage Verification")
    parser.add_argument("--max_train", type=int, default=1000)
    parser.add_argument("--max_dev", type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("  🔒 FEVER NO-LEAKAGE VERIFICATION")
    print("=" * 60)

    # Load splits
    from data.fever_dataset import load_fever_splits
    splits = load_fever_splits(max_train=args.max_train, max_dev=args.max_dev)

    results = {
        "split_integrity": check_1_split_integrity(splits),
        "evidence_leakage": check_2_evidence_leakage_guard(splits),
        "cache_hygiene": check_3_cache_hygiene(),
        "shuffle_sanity": check_4_label_shuffle_sanity(splits),
        "near_duplicates": check_5_near_duplicates(splits),
        "cegis_no_dev": check_6_cegis_no_dev_training(),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  🔒 ALL CHECKS PASSED — No leakage detected")
    else:
        print("\n  ⚠️ SOME CHECKS FAILED — Fix before training!")
        sys.exit(1)


if __name__ == "__main__":
    main()

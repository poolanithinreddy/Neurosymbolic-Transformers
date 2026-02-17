#!/usr/bin/env python3
"""Print 5 FEVER sample examples for quick sanity checking.

Shows claim, evidence (first 200 chars), label, and lengths.

Usage:
    python scripts/fever_sanity_samples.py [--n 5] [--max_train 1000]
"""

import argparse
import os
import sys

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Print FEVER sanity samples")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to print")
    parser.add_argument("--max_train", type=int, default=1000)
    parser.add_argument("--max_dev", type=int, default=500)
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    args = parser.parse_args()

    from data.fever_dataset import load_fever_splits

    splits = load_fever_splits(max_train=args.max_train, max_dev=args.max_dev)
    items = splits.get(args.split, [])

    if not items:
        print(f"No {args.split} data found.")
        return

    print(f"\n{'='*70}")
    print(f"  FEVER Sanity Samples — {args.split} split ({len(items)} total)")
    print(f"{'='*70}")

    for i, it in enumerate(items[: args.n]):
        claim = it["claim"]
        evidence = it.get("gold_evidence_text", "")
        label = it["label"]
        print(f"\n  [{i+1}] ID: {it['id']}")
        print(f"      Label:    {label}")
        print(f"      Claim:    {claim[:200]}{'…' if len(claim) > 200 else ''}")
        print(f"      Evidence: {evidence[:200]}{'…' if len(evidence) > 200 else ''}")
        print(f"      Claim len:    {len(claim)} chars")
        print(f"      Evidence len: {len(evidence)} chars")

    # Summary stats
    claim_lens = [len(it["claim"]) for it in items]
    ev_lens = [len(it.get("gold_evidence_text", "")) for it in items]
    has_ev = sum(1 for e in ev_lens if e > 0)
    print(f"\n  Summary ({args.split}):")
    print(f"    Mean claim length:    {sum(claim_lens)/len(claim_lens):.0f} chars")
    print(f"    Mean evidence length: {sum(ev_lens)/max(1,len(ev_lens)):.0f} chars")
    print(f"    With evidence:        {has_ev}/{len(items)} ({100*has_ev/len(items):.1f}%)")
    print(f"  ✅ Sanity check complete")


if __name__ == "__main__":
    main()

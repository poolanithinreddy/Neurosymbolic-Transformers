import argparse
import csv
import json
import os


def convert_to_tsv(jsonl_in, tsv_out, split="dev"):
    os.makedirs(os.path.dirname(tsv_out), exist_ok=True)
    with open(jsonl_in) as fi, open(tsv_out, "w") as fo:
        w = csv.writer(fo, delimiter="\t")
        for line in fi:
            ex = json.loads(line)
            claim = ex.get("claim", "")
            label = ex.get("label", "NEI")
            evid = ex.get("evidence_text", "")
            w.writerow([split, claim, label, evid])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=False, help="Path to input FEVER jsonl")
    ap.add_argument("--out", dest="out", default="data/fever.tsv")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    if not args.inp:
        # Placeholder: user provides path; proper downloader can be added
        raise SystemExit(
            "Please provide --in path to FEVER jsonl with fields claim,label,evidence_text"
        )
    convert_to_tsv(args.inp, args.out, split=args.split)
    print(f"Wrote {args.out}")

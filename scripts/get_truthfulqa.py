import argparse
import csv
import os
import sys
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/master/TruthfulQA.csv"


def fetch_truthfulqa_csv(url: str) -> list[dict]:
    with urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    rdr = csv.DictReader(text.splitlines())
    for r in rdr:
        rows.append(r)
    return rows


def to_tsv(rows: list[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Columns: split\tquestion\tlabel\treference
    with open(out_path, "w") as f:
        f.write("split\tquestion\tlabel\treference\n")
        for r in rows:
            q = r.get("Question", "").strip()
            true_ans = r.get("Best Answer", "").strip()
            label = "True" if true_ans else "NEI"
            # No official splits; default to dev
            f.write(f"dev\t{q}\t{label}\t{true_ans}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/truthfulqa.tsv")
    args = ap.parse_args()
    try:
        rows = fetch_truthfulqa_csv(URL)
    except Exception as e:
        print(f"Failed to download TruthfulQA CSV: {e}", file=sys.stderr)
        sys.exit(1)
    to_tsv(rows, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

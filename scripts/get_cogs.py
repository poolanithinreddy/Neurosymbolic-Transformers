# TODO: download COGS dataset to data/cogs/
print("stub: get_cogs")
import argparse
import os

URL = "https://raw.githubusercontent.com/najoungkim/COGS/master/data/README.md"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_placeholder(outdir: str):
    ensure_dir(outdir)
    for split in ("train", "dev", "test"):
        path = os.path.join(outdir, f"{split}.tsv")
        with open(path, "w") as f:
            f.write("split\tinput\toutput\n")
            if split != "test":
                f.write(f"{split}\tJohn sees Mary.\tsee(John, Mary)\n")
            else:
                f.write(f"{split}\tMary sees John.\tsee(Mary, John)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/cogs")
    args = ap.parse_args()
    # Placeholder minimal dataset to avoid external dependency issues in offline runs.
    write_placeholder(args.outdir)
    print(f"Wrote placeholder COGS TSVs to {args.outdir}")


if __name__ == "__main__":
    main()

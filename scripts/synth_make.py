import argparse
import json
import random


def make_parity(n=500):
    data = []
    for _ in range(n):
        n0 = random.randint(1, 50)
        m = n0 + 1
        label = "odd" if n0 % 2 == 0 else "even"
        data.append({"n": n0, "m": m, "target": label})
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--family", action="store_true")
    args = ap.parse_args()
    data = make_parity(args.n)  # extend for --family
    with open(args.out, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

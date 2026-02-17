import argparse
import json
import os
import sys
from typing import List

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_cogs_tsv(dir_path: str, split: str) -> list[dict]:
    path = os.path.join(dir_path, f"{split}.tsv")
    rows = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            s, inp, out = line.rstrip("\n").split("\t")
            if s != split:
                continue
            rows.append({"split": s, "input": inp, "output": out})
    return rows


def batch_generate(model, tok, inputs: List[str], device: str = "cpu", max_new_tokens: int = 64):
    enc = tok(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    return tok.batch_decode(out, skip_special_tokens=True)


def f1(a: str, b: str) -> float:
    sa, sb = a.split(), b.split()
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    common = sum(min(sa.count(w), sb.count(w)) for w in set(sa) | set(sb))
    p = common / max(1, len(sa))
    r = common / max(1, len(sb))
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate(ckpt_dir: str, data_dir: str, split: str = "test", device: str = "cpu") -> dict:
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(device)
    rows = load_cogs_tsv(data_dir, split)
    prompts = [f"{r['input']}" for r in rows]
    preds = batch_generate(model, tok, prompts, device=device)
    em = sum(1 for p, r in zip(preds, rows) if p.strip() == r["output"].strip()) / max(1, len(rows))
    f1s = [f1(p, r["output"]) for p, r in zip(preds, rows)]
    return {"exact_match": em, "f1": sum(f1s) / max(1, len(f1s)), "count": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/cogs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rep = evaluate(args.ckpt, args.data, split=args.split, device=args.device)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in ("exact_match", "f1", "count")}, indent=2))


if __name__ == "__main__":
    main()

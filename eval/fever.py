"""FEVER evaluation — supports both DeBERTa NLI and legacy T5 models.

Primary usage (DeBERTa):
    python eval/fever.py --ckpt outputs_fever_gold_neural/ckpt \\
        --model_type deberta --device auto

Legacy usage (T5 seq2seq):
    python eval/fever.py --ckpt outputs_fever/ckpt --data data/fever.tsv \\
        --model_type t5 --device cpu
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.nn.functional as F

from data.fever_dataset import (
    LABEL2ID, ID2LABEL, NUM_LABELS, FEVER_LABELS,
    load_fever_splits, FeverGoldDataset, fever_collate_fn,
)
from eval.fever_metrics import label_accuracy, confusion_matrix, integrity_check
from eval.calibration_metrics import expected_calibration_error, brier_score


# ── DeBERTa evaluation (primary) ────────────────────────────

def evaluate_deberta(
    ckpt_dir: str,
    model_name: str = "microsoft/deberta-v3-base",
    max_dev: int | None = None,
    batch_size: int = 32,
    max_length: int = 384,
    device: str = "auto",
    model_type: str = "nli",
) -> dict:
    """Evaluate a DeBERTa-based FEVER checkpoint.

    Args:
        ckpt_dir: checkpoint directory with saved model + tokenizer.
        model_name: HF model name (for tokenizer fallback).
        max_dev: limit dev set size (None = full).
        batch_size: evaluation batch size.
        max_length: max token length.
        device: computation device.
        model_type: "nli" for FeverNLIWrapper, "veri" for NSTVeriModel.

    Returns:
        Report dict with accuracy, ECE, Brier, per-class metrics.
    """
    from functools import partial

    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )

    # Load model
    if model_type == "veri":
        from models.nst_veri import NSTVeriModel
        from models.fever_nli import build_fever_model
        from transformers import AutoConfig

        tokenizer, base_model = build_fever_model(model_name=ckpt_dir)
        hf_config = AutoConfig.from_pretrained(ckpt_dir)
        hidden_dim = hf_config.hidden_size

        model = NSTVeriModel(backbone=base_model, hidden_dim=hidden_dim)
        extra_state = os.path.join(ckpt_dir, "nst_veri_state.pt")
        if os.path.exists(extra_state):
            state = torch.load(extra_state, map_location="cpu", weights_only=True)
            model.verification.load_state_dict(state.get("verification_heads", {}))
            model.contrastive.load_state_dict(state.get("contrastive_head", {}))
        model.to(device).eval()
    else:
        from models.fever_nli import build_fever_model, FeverNLIWrapper
        tokenizer, base_model = build_fever_model(model_name=ckpt_dir)
        model = FeverNLIWrapper(base_model).to(device).eval()

    # Load data
    splits = load_fever_splits(max_dev=max_dev)
    dev_ds = FeverGoldDataset(splits["dev"])
    collate = partial(fever_collate_fn, tokenizer=tokenizer, max_length=max_length)
    loader = torch.utils.data.DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
    )

    all_preds, all_golds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            if hasattr(model, "predict"):
                out = model.predict(input_ids, attn_mask)
            else:
                out = model(input_ids, attn_mask)
            probs = out["probs"].cpu()
            preds = probs.argmax(dim=-1)
            all_preds.extend(preds.tolist())
            all_golds.extend(labels.tolist())
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)
    all_labels_t = torch.tensor(all_golds)

    pred_labels = [ID2LABEL[p] for p in all_preds]
    gold_labels = [ID2LABEL[g] for g in all_golds]

    # Metrics
    acc_report = label_accuracy(pred_labels, gold_labels)
    cm = confusion_matrix(pred_labels, gold_labels)
    ece, bin_data = expected_calibration_error(all_probs, all_labels_t)
    bs = brier_score(all_probs, all_labels_t)

    report = {
        "model_type": model_type,
        "checkpoint": ckpt_dir,
        "accuracy": acc_report["accuracy"],
        "ece": ece,
        "brier": bs,
        "per_class": acc_report["per_class"],
        "confusion_matrix": cm,
        "n_samples": len(all_golds),
    }

    return report


# ── Legacy T5 evaluation ────────────────────────────────────

LABELS_T5 = ["Supported", "Refuted", "NEI"]


def load_examples(path):
    data = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            split, claim, label, evid = parts[:4]
            data.append({"split": split, "claim": claim, "label": label, "evidence": evid})
    return data


def classify_batch_t5(model, tok, batch_inputs, device="cpu", task_name="fever", max_new_tokens=4):
    from decoding.hard_masks import build_mask_processor
    enc = tok(batch_inputs, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        mask_proc = build_mask_processor(tok, task_name)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            logits_processor=[mask_proc],
        )
    preds = tok.batch_decode(out, skip_special_tokens=True)
    mapped = []
    for p in preds:
        p = p.strip()
        match = None
        for lab in LABELS_T5:
            if p.lower().startswith(lab.lower()):
                match = lab
                break
        mapped.append(match if match is not None else "NEI")
    return mapped


def evaluate_t5(ckpt_dir, data_path, split="dev", device="cpu"):
    """Legacy T5 seq2seq evaluation."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(device)
    except Exception:
        from models.local_tiny import LocalTinySeq2Seq, build_tokenizer_from_texts
        seed_texts = [
            "Claim: test claim. Evidence: test evidence.",
        ]
        tok = build_tokenizer_from_texts(seed_texts, LABELS_T5)
        model = LocalTinySeq2Seq(
            vocab_size=len(tok.token_to_id), num_labels=len(LABELS_T5), d_model=128
        ).to(device)

    exs = [e for e in load_examples(data_path) if e["split"] == split]
    inputs = [f"{e['claim']} [SEP] {e['evidence']}" for e in exs]
    gold = [e["label"] for e in exs]
    preds = []
    bs = 8
    for i in range(0, len(inputs), bs):
        preds.extend(classify_batch_t5(model, tok, inputs[i : i + bs], device=device))
    acc = sum(1 for p, g in zip(preds, gold) if p == g) / max(1, len(gold))
    return {"accuracy": acc, "count": len(gold)}


# ── Unified entry point ─────────────────────────────────────

def evaluate(ckpt_dir, data_path=None, split="dev", device="cpu", model_type="deberta"):
    """Unified evaluate: dispatches to DeBERTa or T5 evaluation."""
    if model_type in ("deberta", "nli", "veri"):
        return evaluate_deberta(ckpt_dir, device=device, model_type=model_type if model_type == "veri" else "nli")
    else:
        return evaluate_t5(ckpt_dir, data_path or "data/fever.tsv", split=split, device=device)


def print_report(report: dict) -> None:
    """Pretty-print an evaluation report."""
    print(f"\n{'='*55}")
    print(f"  FEVER Evaluation Report")
    print(f"{'='*55}")
    print(f"  Model:     {report.get('checkpoint', 'N/A')}")
    print(f"  Type:      {report.get('model_type', 'N/A')}")
    print(f"  Samples:   {report.get('n_samples', report.get('count', 'N/A'))}")
    print(f"  Accuracy:  {report['accuracy']:.4f}")
    if "ece" in report:
        print(f"  ECE:       {report['ece']:.4f}")
    if "brier" in report:
        print(f"  Brier:     {report['brier']:.4f}")
    if "per_class" in report:
        print(f"  Per-class:")
        for lbl, stats in report["per_class"].items():
            acc = stats.get("accuracy", stats.get("acc", 0))
            cnt = stats.get("count", stats.get("n", 0))
            print(f"    {lbl:<20} {acc:.4f}  (n={cnt})")


def main():
    ap = argparse.ArgumentParser(description="FEVER evaluation (DeBERTa or T5)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint directory")
    ap.add_argument("--model_type", default="deberta",
                    choices=["deberta", "nli", "veri", "t5"],
                    help="Model type to evaluate")
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base",
                    help="HF model name (for tokenizer fallback)")
    ap.add_argument("--data", default="data/fever.tsv", help="TSV data path (T5 only)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_dev", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--report", default=None, help="Save JSON report to path")
    args = ap.parse_args()

    report = evaluate(args.ckpt, data_path=args.data, split=args.split,
                      device=args.device, model_type=args.model_type)
    print_report(report)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()

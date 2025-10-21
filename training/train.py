import math
import os
import random
import sys

import torch
import yaml

# Ensure project root is importable when running as a script
_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from logic.logic import horn_violation
from models.rcbm import RCBM
from peft import LoraConfig, get_peft_model
from training.cagrad import cagrad as cagrad_merge
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)


def build_model(cfg):
    model_name = cfg["model_name"]
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        # Fallback path: try t5-small; if still failing, re-raise
        print(f"[build_model] Failed to load '{model_name}' due to: {e}")
        print("[build_model] Trying fallback 't5-small'...")
        model_name = "t5-small"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if cfg["lora"]["use_lora"]:
        peft_conf = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            target_modules=cfg["lora"]["target_modules"],
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, peft_conf)
    return tok, model


LABELS = ["Supported", "Refuted", "NEI"]


def _read_fever_tsv(path):
    """Read FEVER TSV. Accepts either 4 cols (split,claim,label,evidence) or 5 cols (id,split,claim,label,evidence).
    Skips header if present."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and (parts[0].lower() in {"id", "split"}):
                continue
            if len(parts) >= 5:
                _, split, claim, label, evid = parts[:5]
            elif len(parts) >= 4:
                split, claim, label, evid = parts[:4]
            else:
                # skip malformed
                continue
            rows.append({"split": split, "claim": claim, "label": label, "evidence": evid})
    return rows


def _fever_batchify(tok, examples, device):
    texts = [f"Claim: {ex['claim']}\nEvidence: {ex['evidence']}" for ex in examples]
    labels = [ex["label"] if ex["label"] in LABELS else "NEI" for ex in examples]
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    lab_ids = tok(labels, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    return enc, lab_ids, labels


def _concept_targets_from_label(labels, device):
    # Supported -> TrueClaim=1, FalseClaim=0; Refuted -> 0,1; NEI -> 0.5,0.5 (uncertain)
    t = torch.tensor(
        [1.0 if l == "Supported" else (0.0 if l == "Refuted" else 0.5) for l in labels],
        device=device,
    )
    f = torch.tensor(
        [1.0 if l == "Refuted" else (0.0 if l == "Supported" else 0.5) for l in labels],
        device=device,
    )
    return t, f


def _lambda_warmup(step, warmup_steps, target_lambda):
    if warmup_steps <= 0:
        return target_lambda
    if step >= warmup_steps:
        return target_lambda
    return target_lambda * (step / warmup_steps)


def flatten_grads(params):
    grads = []
    shapes = []
    for p in params:
        if p.grad is None:
            shapes.append(None)
            grads.append(torch.zeros(0, device=p.device))
            continue
        g = p.grad.detach().reshape(-1)
        shapes.append(p.grad.shape)
        grads.append(g)
    if grads:
        return torch.cat([g for g in grads if g.numel() > 0], dim=0), shapes
    return torch.tensor([], device=params[0].device), shapes


def set_grads(params, flat_grad, shapes):
    idx = 0
    for p, shp in zip(params, shapes):
        if shp is None:
            continue
        n = math.prod(shp)
        g = flat_grad[idx : idx + n].reshape(shp)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        idx += n


def _auto_select_device(preferred: str | None = None) -> str:
    # Prefer CUDA, then MPS, then CPU
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    # If explicitly requested, validate and fallback
    dev = preferred
    if dev == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but not available; falling back to MPS/CPU.")
        return _auto_select_device("auto")
    if dev == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("[train] MPS requested but not available; falling back to CPU.")
        return _auto_select_device("auto")
    return dev


def train_one(cfg_path, task="parity", outdir_override: str | None = None):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(int(cfg.get("seed", 42)))
    device = _auto_select_device(str(cfg.get("device", "auto")))

    tok, model = build_model(cfg)
    model.to(device)
    hidden_dim = model.config.d_model if hasattr(model.config, "d_model") else 1024
    pred_info = [{"name": "TrueClaim", "arity": 1}, {"name": "FalseClaim", "arity": 1}]
    rcbm = RCBM(hidden_dim, pred_info).to(device)

    lr = float(cfg["train"].get("lr", 3e-4))
    optim = torch.optim.AdamW(list(model.parameters()) + list(rcbm.parameters()), lr=lr)

    use_fever = task.lower() == "fever"
    if use_fever:
        data = _read_fever_tsv(cfg["data"]["fever_path"])
        train_data = [d for d in data if d["split"].lower().startswith("train")]
        dev_data = [d for d in data if d["split"].lower().startswith("dev")]
        if len(train_data) == 0:
            # fallback to any
            train_data = data
        batch_size = max(1, int(cfg["train"].get("batch_size", 1)))
        epochs = int(cfg["train"].get("epochs", 1))
        max_steps = int(cfg["train"].get("max_steps", 0)) or (
            epochs * max(1, len(train_data) // batch_size)
        )
        warmup_steps = int(cfg["train"].get("warmup_steps", 500))
        target_lambda = float(cfg["logic"].get("lambda", 0.0))
        cagrad_c = float(cfg["logic"].get("cagrad_c", 0.0))

        model.train()
        step = 0
        while step < max_steps:
            random.shuffle(train_data)
            for i in range(0, len(train_data), batch_size):
                batch_ex = train_data[i : i + batch_size]
                enc, lab_ids, str_labels = _fever_batchify(tok, batch_ex, device)

                out = model(**enc, labels=lab_ids)
                loss_task = out.loss

                # encoder pooled reps
                with torch.no_grad():
                    enc_out = model.get_encoder()(**enc).last_hidden_state
                    pooled = enc_out.mean(dim=1)

                preds = rcbm(pooled, ent_reps={}, pairs={})
                tgt_t, tgt_f = _concept_targets_from_label(str_labels, device)
                # BCE for concept supervision
                bce = torch.nn.BCELoss()
                loss_concept = bce(preds["TrueClaim"], tgt_t) + bce(preds["FalseClaim"], tgt_f)
                # logic: TrueClaim => not FalseClaim
                v1 = horn_violation(
                    torch.stack([preds["TrueClaim"]], dim=-1), 1 - preds["FalseClaim"]
                )
                lam = _lambda_warmup(step, warmup_steps, target_lambda)
                loss_logic = v1.mean() * lam

                total_loss = loss_task + loss_concept + loss_logic

                if cagrad_c > 0:
                    # CAGrad merge between task and (concept+logic)
                    params = [
                        p
                        for p in list(model.parameters()) + list(rcbm.parameters())
                        if p.requires_grad
                    ]
                    optim.zero_grad()
                    loss_task.backward(retain_graph=True)
                    g_task, shapes = flatten_grads(params)

                    optim.zero_grad()
                    (loss_concept + loss_logic).backward(retain_graph=False)
                    g_logic, _ = flatten_grads(params)

                    g_comb = cagrad_merge([g_task, g_logic], c=cagrad_c)
                    # set combined grads
                    optim.zero_grad()
                    set_grads(params, g_comb, shapes)
                    optim.step()
                else:
                    optim.zero_grad()
                    total_loss.backward()
                    optim.step()

                if step % 20 == 0:
                    print(
                        f"step {step} task {loss_task.item():.3f} concept {loss_concept.item():.3f} logic {loss_logic.item():.3f} lam {lam:.3f}"
                    )
                step += 1
                if step >= max_steps:
                    break
        # save
        out_root = outdir_override or cfg["io"]["out_dir"]
        os.makedirs(out_root, exist_ok=True)
        ckpt_dir = os.path.join(out_root, "ckpt")
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)
        torch.save(
            {"state_dict": rcbm.state_dict(), "hidden_dim": hidden_dim},
            os.path.join(ckpt_dir, "rcbm.pt"),
        )
        return

    # Fallback: toy loop (parity demo)
    inputs = ["Barack Obama was born in Hawaii."]
    labels = ["Supported"]
    model.train()
    for step in range(min(cfg["train"].get("max_steps", 50), 50)):
        enc = tok(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc, labels=tok(labels, return_tensors="pt").input_ids.to(device))
        loss_task = out.loss
        with torch.no_grad():
            enc_out = model.get_encoder()(**enc).last_hidden_state
            pooled = enc_out.mean(dim=1)
        preds = rcbm(pooled, ent_reps={}, pairs={})
        v1 = horn_violation(torch.stack([preds["TrueClaim"]], dim=-1), 1 - preds["FalseClaim"])
        loss_logic = v1.mean() * cfg["logic"]["lambda"]
        loss = loss_task + loss_logic
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 10 == 0:
            print(f"step {step} task {loss_task.item():.3f} logic {loss_logic.item():.3f}")

    out_root = outdir_override or cfg["io"]["out_dir"]
    os.makedirs(out_root, exist_ok=True)
    ckpt_dir = os.path.join(out_root, "ckpt")
    model.save_pretrained(ckpt_dir)
    tok.save_pretrained(ckpt_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="parity")
    ap.add_argument(
        "--outdir", default=None, help="Override output directory (defaults to cfg io.out_dir)"
    )
    args = ap.parse_args()
    train_one(args.config, task=args.task, outdir_override=args.outdir)

"""MNLI pre-fine-tuning for FEVER.

Pre-fine-tunes DeBERTa on MultiNLI before FEVER training.
This initializes the NLI classification head with better weights.

MNLI → FEVER label mapping:
    entailment    → SUPPORTS
    contradiction → REFUTES
    neutral       → NOT ENOUGH INFO

Usage:
    python -m training.pretrain_mnli --config configs/fever_gold_nst_veri.yaml
    python -m training.pretrain_mnli --model microsoft/deberta-v3-large --epochs 1

The checkpoint can then be loaded via model.pretrained_path in the VERI config.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

logger = logging.getLogger("pretrain_mnli")

# MNLI → FEVER label mapping
MNLI_TO_FEVER = {
    "entailment": 0,      # SUPPORTS
    "contradiction": 1,   # REFUTES
    "neutral": 2,         # NOT ENOUGH INFO
}


class MNLIDataset(Dataset):
    """Minimal MNLI dataset for pre-fine-tuning."""

    def __init__(self, split: str = "train", max_samples: int | None = None):
        """Load MNLI from HuggingFace datasets.

        Args:
            split: "train" or "validation_matched"
            max_samples: limit number of samples (for debugging)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        logger.info(f"Loading MNLI split={split}...")
        ds = load_dataset("nli_tr", "multinli", split=split, trust_remote_code=True)

        # Fallback to standard MNLI if nli_tr doesn't work
        if ds is None:
            ds = load_dataset("glue", "mnli", split=split)

        self.examples = []
        for item in ds:
            label = item.get("label", -1)
            if label < 0 or label > 2:
                continue
            self.examples.append({
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": label,  # 0=entailment→SUPPORTS, 1=neutral→NEI, 2=contradiction→REFUTES
            })

        # MNLI uses 0=entailment, 1=neutral, 2=contradiction
        # FEVER uses 0=SUPPORTS, 1=REFUTES, 2=NEI
        # Remap: MNLI(0→0, 1→2, 2→1)
        for ex in self.examples:
            mnli_label = ex["label"]
            if mnli_label == 0:
                ex["label"] = 0  # entailment → SUPPORTS
            elif mnli_label == 1:
                ex["label"] = 2  # neutral → NOT ENOUGH INFO
            elif mnli_label == 2:
                ex["label"] = 1  # contradiction → REFUTES

        if max_samples and max_samples < len(self.examples):
            random.shuffle(self.examples)
            self.examples = self.examples[:max_samples]

        logger.info(f"MNLI {split}: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def mnli_collate_fn(batch, tokenizer, max_length=384):
    """Collate MNLI batch: premise [SEP] hypothesis."""
    premises = [item["premise"] for item in batch]
    hypotheses = [item["hypothesis"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    encoding = tokenizer(
        premises, hypotheses,
        max_length=max_length,
        padding="longest",
        truncation="only_first",  # Truncate premise, keep hypothesis intact
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def pretrain_mnli(
    model_name: str = "microsoft/deberta-v3-large",
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_train: int | None = None,
    max_length: int = 256,
    out_dir: str = "outputs_mnli_pretrain",
    seed: int = 42,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    device: str | None = None,
) -> str:
    """Pre-fine-tune DeBERTa on MNLI with FEVER-aligned labels.

    Returns path to saved checkpoint directory.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)

    # Build model
    from models.fever_nli import build_fever_model
    tokenizer, model = build_fever_model(
        model_name=model_name,
        label_smoothing=0.05,
        dropout=0.1,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gradient_checkpointing=True,
    )
    model = model.to(device)

    # Data
    train_ds = MNLIDataset("train", max_samples=max_train)
    val_ds = MNLIDataset("validation_matched", max_samples=5000)

    from functools import partial
    collate = partial(mnli_collate_fn, tokenizer=tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=4, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=(device == "cuda"),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.06)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # AMP
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"\n{'='*55}")
    print(f"  MNLI Pre-Fine-Tuning")
    print(f"  Model: {model_name}" + (f" + LoRA r={lora_rank}" if use_lora else ""))
    print(f"  Epochs: {epochs}, Batch: {batch_size}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"{'='*55}\n")

    t0 = time.time()
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs["logits"]
                loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)
            global_step += 1

            if global_step % 500 == 0:
                print(f"  Step {global_step}: loss={loss.item():.4f}, "
                      f"acc={n_correct/n_total:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs["logits"]

                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        train_acc = n_correct / max(1, n_total)
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"train_loss={epoch_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # Save checkpoint
    ckpt_dir = os.path.join(out_dir, "mnli_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    elapsed = time.time() - t0
    print(f"\n  MNLI pre-training done in {elapsed:.1f}s")
    print(f"  Checkpoint saved to: {ckpt_dir}")
    print(f"  Use this as model.pretrained_path in your VERI config")

    return ckpt_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="MNLI pre-fine-tuning for FEVER")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="outputs_mnli_pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    pretrain_mnli(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train=args.max_train,
        max_length=args.max_length,
        out_dir=args.out_dir,
        seed=args.seed,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        device=args.device,
    )

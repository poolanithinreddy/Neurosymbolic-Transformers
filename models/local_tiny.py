import json
from typing import Dict, List

import torch
import torch.nn as nn


class SimpleTokenizer:
    """
    A minimal whitespace tokenizer with a fixed vocab. Intended for offline/local quick runs.
    Provides a subset of the HF tokenizer API we use: __call__, batch_decode.
    Also adds encode_labels to map class labels to single IDs.
    """

    def __init__(self, vocab: Dict[str, int], labels: List[str]):
        self.token_to_id = dict(vocab)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.labels = list(labels)
        # Ensure specials exist
        for sp in [self.pad_token, self.eos_token, self.unk_token]:
            if sp not in self.token_to_id:
                self.token_to_id[sp] = len(self.token_to_id)
                self.id_to_token[self.token_to_id[sp]] = sp
        # Ensure labels exist as atomic tokens
        for lab in self.labels:
            if lab not in self.token_to_id:
                i = len(self.token_to_id)
                self.token_to_id[lab] = i
                self.id_to_token[i] = lab

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

    def __call__(
        self,
        texts: List[str],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        ids = []
        attn = []
        max_len = 0
        for t in texts:
            toks = t.strip().split()
            seq = [self.token_to_id.get(w, self.unk_token_id) for w in toks]
            seq = seq[:max_length]
            seq.append(self.eos_token_id)
            ids.append(seq)
            max_len = max(max_len, len(seq))
        if padding:
            ids = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in ids]
        attn = [[1 if tok != self.pad_token_id else 0 for tok in seq] for seq in ids]
        res = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
        return res

    def batch_decode(self, sequences: List[List[int]], skip_special_tokens=True) -> List[str]:
        outs = []
        for seq in sequences:
            toks = []
            for i in seq:
                tok = self.id_to_token.get(int(i), self.unk_token)
                if skip_special_tokens and tok in {self.pad_token, self.eos_token}:
                    continue
                toks.append(tok)
            outs.append(" ".join(toks).strip())
        return outs

    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        return torch.tensor(
            [[self.token_to_id.get(lab, self.unk_token_id)] for lab in labels], dtype=torch.long
        )

    def save_pretrained(self, path: str):
        with open(f"{path}/local_tiny_vocab.json", "w") as f:
            json.dump(self.token_to_id, f)

    @classmethod
    def from_pretrained(cls, path: str, labels: List[str]):
        with open(f"{path}/local_tiny_vocab.json") as f:
            vocab = json.load(f)
        return cls(vocab=vocab, labels=labels)


class LocalTinySeq2Seq(nn.Module):
    """
    A tiny classifier-like seq2seq that predicts one of N label tokens per input.
    Implements .generate and a .get_encoder shim returning last_hidden_state similar to HF models.
    """

    def __init__(self, vocab_size: int, num_labels: int, d_model: int = 128):
        super().__init__()
        self.is_local_tiny = True
        self.emb = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, num_labels)
        self.num_labels = num_labels
        self.config = type("Cfg", (), {"d_model": d_model})

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.emb(input_ids)  # [B, T, D]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            h = x.sum(dim=1) / denom
        else:
            h = x.mean(dim=1)
        logits = self.proj(h)  # [B, C]
        out = type("Out", (), {})()
        out.logits = logits
        if labels is not None:
            y = labels.squeeze(-1)
            loss = nn.functional.cross_entropy(logits, y)
            out.loss = loss
        return out

    class _EncoderShim:
        def __init__(self, parent):
            self.parent = parent

        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            x = self.parent.emb(input_ids)
            return type("EncOut", (), {"last_hidden_state": x})

    def get_encoder(self):
        return LocalTinySeq2Seq._EncoderShim(self)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=1,
        do_sample=False,
        num_beams=1,
        logits_processor=None,
        **kwargs,
    ):
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        preds = out.logits.argmax(dim=-1)  # [B]
        # Return sequences as single-token ids (treated as full sequence)
        return preds.unsqueeze(1)

    def save_pretrained(self, path: str):
        torch.save(self.state_dict(), f"{path}/local_tiny.pt")

    @classmethod
    def from_pretrained(cls, path: str, vocab_size: int, num_labels: int, d_model: int = 128):
        m = cls(vocab_size=vocab_size, num_labels=num_labels, d_model=d_model)
        m.load_state_dict(torch.load(f"{path}/local_tiny.pt", map_location="cpu"))
        return m


def build_tokenizer_from_texts(texts: List[str], labels: List[str]) -> SimpleTokenizer:
    vocab = {}
    # seed with labels and specials will be added by the tokenizer
    for line in texts:
        for w in line.strip().split():
            if w not in vocab:
                vocab[w] = len(vocab)
            if len(vocab) > 5000:
                break
        if len(vocab) > 5000:
            break
    return SimpleTokenizer(vocab=vocab, labels=labels)

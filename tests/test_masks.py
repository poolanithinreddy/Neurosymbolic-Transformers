import pytest
import torch


def _try_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("t5-small")
    except Exception:
        pytest.skip("HF tokenizer not available; skipping mask test")


def test_hard_mask_processor_limits_vocab():
    tok = _try_tokenizer()
    from decoding.hard_masks import LABEL_TOKENS, build_mask_processor

    proc = build_mask_processor(tok, "fever")
    # build allowed ids as in code (first subtoken of each label)
    allowed = []
    for lab in LABEL_TOKENS["fever"]:
        ids = tok.encode(lab, add_special_tokens=False)
        if ids:
            allowed.append(ids[0])
    assert allowed, "Expected non-empty allowed IDs"

    vocab = tok.vocab_size if hasattr(tok, "vocab_size") else 32128
    scores = torch.zeros(2, vocab)
    masked = proc(torch.zeros(2, 1, dtype=torch.long), scores)
    # Only allowed positions should remain 0, others -inf
    assert torch.isinf(masked).sum().item() == masked.numel() - len(allowed) * masked.shape[0]
    for a in allowed:
        assert torch.isinf(masked[:, a]).sum().item() == 0

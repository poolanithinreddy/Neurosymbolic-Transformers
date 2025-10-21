# For tasks with restricted vocab / labels
LABEL_TOKENS = {"fever": ["Supported", "Refuted", "NEI"]}


def constrain_logits(logits, tokenizer, task_name):
    """
    Constrain next-token logits to the first subtoken IDs of the label words.
    Works reasonably for SentencePiece/BPE tokenizers (e.g., T5).
    """
    if task_name not in LABEL_TOKENS:
        return logits
    allowed_labels = LABEL_TOKENS[task_name]
    allowed_ids = []
    for lab in allowed_labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        if len(ids) > 0:
            allowed_ids.append(ids[0])
    if len(allowed_ids) == 0:
        return logits
    mask = logits.new_full(logits.shape, float("-inf"))
    mask[..., allowed_ids] = 0.0
    return logits + mask


try:
    from transformers.generation.logits_process import LogitsProcessor
except Exception:

    class LogitsProcessor:  # fallback stub to avoid import error in non-transformers contexts
        def __call__(self, input_ids, scores):
            return scores


class HardMaskProcessor(LogitsProcessor):
    def __init__(self, tokenizer, task_name):
        self.allowed_ids = []
        if task_name in LABEL_TOKENS:
            for lab in LABEL_TOKENS[task_name]:
                ids = tokenizer.encode(lab, add_special_tokens=False)
                if len(ids) > 0:
                    self.allowed_ids.append(ids[0])

    def __call__(self, input_ids, scores):
        if not self.allowed_ids:
            return scores
        mask = scores.new_full(scores.shape, float("-inf"))
        mask[:, self.allowed_ids] = 0.0
        return scores + mask


def build_mask_processor(tokenizer, task_name):
    return HardMaskProcessor(tokenizer, task_name)

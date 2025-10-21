"""Reranking utilities."""


def combine_scores(lm_logprob: float, rule_score: float, alpha: float = 0.2) -> float:
    """
    Mix model LM score and rule/RCBM score.
    - lm_logprob: higher is better (e.g., negative loss or log-prob).
    - rule_score: normalized to [0,1] where higher is more consistent with rules.
    - alpha: weight on rule_score.
    """
    return (1.0 - alpha) * lm_logprob + alpha * rule_score

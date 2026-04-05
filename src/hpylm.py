from __future__ import annotations

from typing import Dict, List, Tuple

from data_pipeline import get_hpylm_data


class HPYLMModel:
    """Placeholder HPYLM model interface for Milestone 2+ implementation."""

    def __init__(self, vocab_size: int = 10_000) -> None:
        self.vocab_size = vocab_size
        self.trained = False

    def fit(self, corpus_ids: List[int]) -> None:
        """Train HPYLM parameters (to be implemented with Gibbs sampling)."""
        _ = corpus_ids
        self.trained = True

    def predict_next(self, context_ids: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """Return top-k next-word candidates as (token_id, probability)."""
        _ = context_ids
        _ = top_k
        raise NotImplementedError("HPYLM prediction is not implemented yet.")


def train_hpylm(vocab_size: int = 10_000) -> Tuple[HPYLMModel, Dict[str, int], Dict[int, str]]:
    """Load corpus IDs and initialize/train a placeholder HPYLM model."""
    corpus_ids, word_to_id, id_to_word = get_hpylm_data(vocab_size=vocab_size)
    model = HPYLMModel(vocab_size=vocab_size)
    model.fit(corpus_ids)
    return model, word_to_id, id_to_word

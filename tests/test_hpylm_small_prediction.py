from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure imports work when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hpylm import HPYLM


def run_hpylm_small_prediction_test() -> None:
    """
    Train HPYLM on a tiny synthetic corpus and validate top-k next-word prediction.

    The corpus is intentionally challenging: it mixes several medical templates
    so the model must rank competing next-word continuations.
    """
    word_to_id = {
        "<UNK>": 0,
        "the": 1,
        "patient": 2,
        "has": 3,
        "fever": 4,
        "pain": 5,
        "cough": 6,
        "reports": 7,
        "shows": 8,
        "mild": 9,
        "severe": 10,
        "improved": 11,
        "worsening": 12,
    }
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    w = word_to_id

    # Build a richer toy corpus with multiple competing patterns.
    corpus_ids: list[int] = []
    for _ in range(35):
        corpus_ids.extend([w["the"], w["patient"], w["has"], w["fever"]])
    for _ in range(18):
        corpus_ids.extend([w["the"], w["patient"], w["has"], w["pain"]])
    for _ in range(10):
        corpus_ids.extend([w["the"], w["patient"], w["reports"], w["cough"]])
    for _ in range(7):
        corpus_ids.extend([w["the"], w["patient"], w["shows"], w["improved"]])
    for _ in range(6):
        corpus_ids.extend([w["the"], w["patient"], w["shows"], w["worsening"]])

    # Add extra contextual noise around "has" to make this less trivial.
    for _ in range(8):
        corpus_ids.extend([w["the"], w["patient"], w["has"], w["mild"], w["pain"]])
    for _ in range(5):
        corpus_ids.extend([w["the"], w["patient"], w["has"], w["severe"], w["fever"]])

    random.seed(7)
    model = HPYLM(order=3, vocab_size=len(word_to_id), discount=0.75, concentration=1.0)
    model.fit(corpus_ids, num_gibbs_iterations=8)

    preds = model.predict_next_word(
        context=["the", "patient"],
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        top_k=3,
    )

    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"
    assert all(pred in word_to_id for pred in preds), "Prediction contains out-of-vocabulary token"
    assert "has" in preds, f"Expected 'has' in top-3 predictions, got {preds}"

    print("HPYLM small prediction test passed.")
    print(f"Top-3 predictions for context ['the', 'patient']: {preds}")


if __name__ == "__main__":
    run_hpylm_small_prediction_test()

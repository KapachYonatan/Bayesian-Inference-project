from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure imports work when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rnn import NeuralAutocompleter, NextWordRNN


def run_rnn_small_prediction_test() -> None:
    """
    Train a next-word RNN on a tiny synthetic corpus and validate predictions.

    The mapping is designed so context ['the', 'patient', 'has'] should strongly
    prefer symptom words, especially 'fever'.
    """
    random.seed(11)
    torch.manual_seed(11)

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
    }
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    w = word_to_id

    # Build supervised next-token examples for seq_len = 3.
    x_rows: list[list[int]] = []
    y_rows: list[int] = []

    for _ in range(80):
        x_rows.append([w["the"], w["patient"], w["has"]])
        y_rows.append(w["fever"])
    for _ in range(30):
        x_rows.append([w["the"], w["patient"], w["has"]])
        y_rows.append(w["pain"])
    for _ in range(25):
        x_rows.append([w["the"], w["patient"], w["reports"]])
        y_rows.append(w["cough"])
    for _ in range(20):
        x_rows.append([w["the"], w["patient"], w["shows"]])
        y_rows.append(w["pain"])

    inputs = torch.tensor(x_rows, dtype=torch.long)
    targets = torch.tensor(y_rows, dtype=torch.long)
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=16, shuffle=False)

    model = NextWordRNN(
        vocab_size=len(word_to_id),
        embed_dim=24,
        hidden_dim=48,
        num_layers=1,
        cell_type="gru",
    )
    completer = NeuralAutocompleter(
        model=model,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        seq_len=3,
    )

    losses = completer.fit(dataloader=dataloader, epochs=20, lr=0.01)

    preds = completer.predict_next_word(context=["the", "patient", "has"], top_k=3)

    assert len(losses) == 20, f"Expected 20 loss entries, got {len(losses)}"
    assert losses[-1] < losses[0], f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"
    assert preds[0] == "fever", f"Expected top-1 prediction 'fever', got {preds[0]}"
    assert "pain" in preds, f"Expected 'pain' in top-3 predictions, got {preds}"

    print("RNN small prediction test passed.")
    print(f"First/last training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"Top-3 predictions for context ['the', 'patient', 'has']: {preds}")


if __name__ == "__main__":
    run_rnn_small_prediction_test()

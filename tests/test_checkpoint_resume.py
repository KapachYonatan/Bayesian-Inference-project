from __future__ import annotations

import pickle
import random
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure imports work when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hpylm import HPYLM
from src.rnn import NeuralAutocompleter, NextWordRNN


def _build_toy_rnn_dataloader() -> tuple[DataLoader, dict[str, int], dict[int, str]]:
    word_to_id = {
        "<UNK>": 0,
        "the": 1,
        "patient": 2,
        "has": 3,
        "fever": 4,
        "pain": 5,
        "cough": 6,
    }
    id_to_word = {v: k for k, v in word_to_id.items()}
    w = word_to_id

    x_rows: list[list[int]] = []
    y_rows: list[int] = []

    for _ in range(30):
        x_rows.append([w["the"], w["patient"], w["has"]])
        y_rows.append(w["fever"])
    for _ in range(15):
        x_rows.append([w["the"], w["patient"], w["has"]])
        y_rows.append(w["pain"])
    for _ in range(10):
        x_rows.append([w["the"], w["patient"], w["patient"]])
        y_rows.append(w["cough"])

    x = torch.tensor(x_rows, dtype=torch.long)
    y = torch.tensor(y_rows, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    return loader, word_to_id, id_to_word


def test_rnn_save_and_resume() -> None:
    random.seed(5)
    torch.manual_seed(5)

    dataloader, word_to_id, id_to_word = _build_toy_rnn_dataloader()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "rnn"

        # First run: should save checkpoints and vocab metadata.
        model_1 = NextWordRNN(vocab_size=len(word_to_id), embed_dim=16, hidden_dim=32, cell_type="gru")
        trainer_1 = NeuralAutocompleter(model_1, word_to_id, id_to_word, device="cpu", seq_len=3)
        losses_1 = trainer_1.fit(
            dataloader=dataloader,
            epochs=4,
            lr=0.01,
            save_dir=str(save_dir),
            valid_dataloader=dataloader,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
        )

        assert 1 <= len(losses_1) <= 4, f"Unexpected epoch-loss count with early stopping: {len(losses_1)}"
        assert (save_dir / "vocab.json").exists(), "Expected vocab.json to be saved"
        assert any(save_dir.glob("rnn_epoch_*.pth")), "Expected at least one per-epoch checkpoint"
        assert (save_dir / "rnn_latest.pt").exists(), "Expected latest checkpoint"
        assert (save_dir / "rnn_best.pt").exists(), "Expected best-checkpoint file for early stopping"

        # Resume path: load latest checkpoint into a fresh model via fit(..., resume_checkpoint=...).
        checkpoint = torch.load(save_dir / "rnn_latest.pt", map_location="cpu")
        model_2 = NextWordRNN(vocab_size=len(word_to_id), embed_dim=16, hidden_dim=32, cell_type="gru")
        trainer_2 = NeuralAutocompleter(model_2, word_to_id, id_to_word, device="cpu", seq_len=3)

        # epochs=0 validates that checkpoint loading works without additional optimizer steps.
        losses_2 = trainer_2.fit(
            dataloader=dataloader,
            epochs=0,
            lr=0.01,
            save_dir=str(save_dir),
            resume_checkpoint=str(save_dir / "rnn_latest.pt"),
            valid_dataloader=dataloader,
            early_stopping_patience=1,
        )
        assert losses_2 == [], f"Expected no losses for 0 epochs, got {losses_2}"

        resumed_state = trainer_2.model.state_dict()
        for key, tensor in checkpoint["model_state_dict"].items():
            assert torch.equal(resumed_state[key], tensor), f"Model parameter mismatch after resume at key '{key}'"



def test_hpylm_save_and_warm_resume() -> None:
    random.seed(7)

    corpus = [1, 2, 3, 4] * 25 + [1, 2, 3, 5] * 10 + [1, 2, 6, 4] * 6

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "hpylm"

        model = HPYLM(order=3, vocab_size=10)
        model.fit(corpus, num_gibbs_iterations=1, save_dir=str(save_dir))

        ckpt_file = save_dir / "hpylm_checkpoint.pkl"
        assert ckpt_file.exists(), "Expected HPYLM checkpoint file to be saved"

        with ckpt_file.open("rb") as fp:
            loaded_model = pickle.load(fp)

        assert isinstance(loaded_model, HPYLM), "Loaded checkpoint is not an HPYLM instance"
        assert len(loaded_model.context_trie) > 1, "Expected non-trivial context trie in saved model"

        trie_size_before = len(loaded_model.context_trie)
        loaded_model.fit(corpus, num_gibbs_iterations=1, warm_start=True, save_dir=str(save_dir))
        assert len(loaded_model.context_trie) >= trie_size_before, "Warm-start should preserve learned context structure"

        with ckpt_file.open("rb") as fp:
            resumed_loaded_model = pickle.load(fp)
        assert isinstance(resumed_loaded_model, HPYLM), "Checkpoint should remain loadable after warm-start resume"


if __name__ == "__main__":
    test_rnn_save_and_resume()
    test_hpylm_save_and_warm_resume()
    print("Checkpoint/resume tests passed.")

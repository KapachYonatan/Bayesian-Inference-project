from __future__ import annotations

import torch
import torch.nn as nn

from data_pipeline import RnnDataBundle, get_rnn_dataloaders


class SimpleRNNLanguageModel(nn.Module):
    """Baseline RNN language model for next-word prediction."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        return self.output(last_hidden)


def build_rnn_training_bundle(
    seq_len: int = 20,
    vocab_size: int = 10_000,
    batch_size: int = 64,
) -> tuple[SimpleRNNLanguageModel, RnnDataBundle]:
    """Create model + dataloaders for training loop integration."""
    dataloaders = get_rnn_dataloaders(
        seq_len=seq_len,
        vocab_size=vocab_size,
        batch_size=batch_size,
    )
    model = SimpleRNNLanguageModel(vocab_size=len(dataloaders.word_to_id))
    return model, dataloaders

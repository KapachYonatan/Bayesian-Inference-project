from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from src.data_pipeline import RnnDataBundle, get_rnn_dataloaders


HiddenState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class NextWordRNN(nn.Module):
    """RNN-family model that predicts only the next token for each sequence."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        cell_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if self.cell_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.cell_type == "gru":
            self.recurrent = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.cell_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError("cell_type must be one of: 'rnn', 'gru', 'lstm'")

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[HiddenState] = None,
    ) -> Tuple[torch.Tensor, HiddenState]:
        """Return logits for the next token and recurrent hidden state."""
        embedded = self.embedding(x)
        out, next_hidden = self.recurrent(embedded, hidden)
        final_timestep = out[:, -1, :]
        logits = self.output(final_timestep)
        return logits, next_hidden


class NeuralAutocompleter:
    def __init__(
        self,
        model: NextWordRNN,
        word_to_id: Dict[str, int],
        id_to_word: Dict[int, str],
        device: str = "cpu",
        seq_len: Optional[int] = None,
    ):
        if device == "auto":
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            requested = torch.device(device)
            if requested.type == "cuda" and not torch.cuda.is_available():
                resolved_device = torch.device("cpu")
            else:
                resolved_device = requested

        self.model = model.to(resolved_device)
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.device = resolved_device
        self.seq_len = seq_len

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float,
        verbose: bool = False,
        save_dir: Optional[str] = None,
    ) -> List[float]:
        """Train with cross-entropy and Adam; return average loss per epoch."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        save_path: Optional[Path] = None
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            vocab_payload = {
                "word_to_id": self.word_to_id,
                "id_to_word": {str(k): v for k, v in self.id_to_word.items()},
                "model_config": {
                    "vocab_size": self.model.vocab_size,
                    "embed_dim": self.model.embed_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "num_layers": self.model.num_layers,
                    "cell_type": self.model.cell_type,
                    "seq_len": self.seq_len,
                },
            }
            with (save_path / "vocab.json").open("w", encoding="utf-8") as fp:
                json.dump(vocab_payload, fp, indent=2)
            if verbose:
                print(f"[RNN] wrote vocab metadata to {save_path / 'vocab.json'}")

        epoch_losses: List[float] = []
        self.model.train()
        if verbose:
            print(
                f"[RNN] fitting cell_type={self.model.cell_type}, embed_dim={self.model.embed_dim}, "
                f"hidden_dim={self.model.hidden_dim}, layers={self.model.num_layers}, epochs={epochs}, lr={lr}"
            )

        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = 0

            if verbose:
                print(f"[RNN] epoch {epoch + 1}/{epochs} started")

            for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += float(loss.item())
                num_batches += 1

                if verbose and batch_index == 1:
                    print(f"[RNN] epoch {epoch + 1} first-batch loss: {loss.item():.4f}")

            avg_loss = running_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            if save_path is not None:
                checkpoint_file = save_path / f"rnn_epoch_{epoch + 1}.pth"
                torch.save(self.model.state_dict(), checkpoint_file)
                if verbose:
                    print(f"[RNN] checkpoint saved: {checkpoint_file}")

            if verbose:
                print(f"[RNN] epoch {epoch + 1}/{epochs} average loss: {avg_loss:.4f}")

        if verbose:
            print("[RNN] training complete")

        return epoch_losses

    def predict_next_word(self, context: List[str], top_k: int = 3) -> List[str]:
        """Predict top-k next words for a context sequence."""
        unk_id = self.word_to_id.get("<UNK>", 0)
        target_len = self.seq_len if self.seq_len is not None else len(context)
        target_len = max(target_len, 1)

        context_ids = [self.word_to_id.get(word.lower(), unk_id) for word in context]
        if len(context_ids) > target_len:
            context_ids = context_ids[-target_len:]
        elif len(context_ids) < target_len:
            context_ids = [unk_id] * (target_len - len(context_ids)) + context_ids

        x = torch.tensor([context_ids], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            top_ids = torch.topk(probs, k=max(top_k, 1), dim=-1).indices[0].tolist()

        return [self.id_to_word.get(idx, "<UNK>") for idx in top_ids]


def build_rnn_training_bundle(
    seq_len: int = 20,
    vocab_size: int = 10_000,
    batch_size: int = 64,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 1,
    cell_type: str = "lstm",
) -> tuple[NextWordRNN, RnnDataBundle, NeuralAutocompleter]:
    """Create model + dataloaders + trainer wrapper for next-word experiments."""
    dataloaders = get_rnn_dataloaders(
        seq_len=seq_len,
        vocab_size=vocab_size,
        batch_size=batch_size,
    )
    model = NextWordRNN(
        vocab_size=len(dataloaders.word_to_id),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cell_type=cell_type,
    )
    completer = NeuralAutocompleter(
        model=model,
        word_to_id=dataloaders.word_to_id,
        id_to_word=dataloaders.id_to_word,
        seq_len=seq_len,
    )
    return model, dataloaders, completer

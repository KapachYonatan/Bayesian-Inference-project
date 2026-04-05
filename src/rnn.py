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
        dropout_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

        if self.cell_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_prob if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.cell_type == "gru":
            self.recurrent = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_prob if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.cell_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_prob if num_layers > 1 else 0.0,
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
        embedded = self.dropout(self.embedding(x))
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
        resume_checkpoint: Optional[str] = None,
        valid_dataloader: Optional[torch.utils.data.DataLoader] = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> List[float]:
        """Train with cross-entropy and Adam; return average loss per epoch."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        if early_stopping_patience > 0 and valid_dataloader is None:
            raise ValueError("valid_dataloader is required when early_stopping_patience > 0")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
        )

        start_epoch = 0
        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_epochs = 0
        best_model_state_dict: Optional[Dict[str, torch.Tensor]] = None

        if resume_checkpoint is not None:
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = int(checkpoint.get("epoch", 0))
                best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
                best_epoch = int(checkpoint.get("best_epoch", best_epoch))
                no_improve_epochs = int(checkpoint.get("no_improve_epochs", no_improve_epochs))
                saved_best_state = checkpoint.get("best_model_state_dict")
                if isinstance(saved_best_state, dict):
                    best_model_state_dict = saved_best_state
            else:
                # Backward compatibility for checkpoints saved as raw state_dict.
                self.model.load_state_dict(checkpoint)
            if verbose:
                print(f"[RNN] resumed from checkpoint: {resume_checkpoint}")

        def _evaluate_validation_loss() -> float:
            if valid_dataloader is None:
                return float("inf")

            self.model.eval()
            total_loss = 0.0
            total_items = 0
            with torch.no_grad():
                for inputs, targets in valid_dataloader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    logits, _ = self.model(inputs)
                    loss = criterion(logits, targets)
                    total_loss += float(loss.item()) * int(targets.numel())
                    total_items += int(targets.numel())
            self.model.train()
            if total_items == 0:
                return float("inf")
            return total_loss / total_items

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
                f"hidden_dim={self.model.hidden_dim}, layers={self.model.num_layers}, epochs={epochs}, lr={lr}, "
                f"early_stopping_patience={early_stopping_patience}, min_delta={early_stopping_min_delta}"
            )

        remaining_epochs = max(0, epochs - start_epoch)
        for local_epoch in range(remaining_epochs):
            current_epoch = start_epoch + local_epoch + 1
            running_loss = 0.0
            num_batches = 0

            if verbose:
                print(f"[RNN] epoch {current_epoch} started")

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
                    print(f"[RNN] epoch {current_epoch} first-batch loss: {loss.item():.4f}")

            avg_loss = running_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            val_loss = _evaluate_validation_loss() if valid_dataloader is not None else None
            improved = False
            if val_loss is not None:
                if val_loss < (best_val_loss - early_stopping_min_delta):
                    best_val_loss = val_loss
                    best_epoch = current_epoch
                    no_improve_epochs = 0
                    improved = True
                    best_model_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    if save_path is not None:
                        best_file = save_path / "rnn_best.pt"
                        torch.save(
                            {
                                "epoch": current_epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "best_val_loss": best_val_loss,
                            },
                            best_file,
                        )
                        if verbose:
                            print(f"[RNN] best checkpoint saved: {best_file}")
                else:
                    no_improve_epochs += 1

            if save_path is not None:
                checkpoint_file = save_path / f"rnn_epoch_{current_epoch}.pth"
                torch.save(self.model.state_dict(), checkpoint_file)
                latest_file = save_path / "rnn_latest.pt"
                torch.save(
                    {
                        "epoch": current_epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "no_improve_epochs": no_improve_epochs,
                        "best_model_state_dict": best_model_state_dict,
                    },
                    latest_file,
                )
                if verbose:
                    print(f"[RNN] checkpoint saved: {checkpoint_file}")

            if verbose:
                if val_loss is None:
                    print(f"[RNN] epoch {current_epoch} average train loss: {avg_loss:.4f}")
                else:
                    print(
                        f"[RNN] epoch {current_epoch} average train loss: {avg_loss:.4f}, "
                        f"val loss: {val_loss:.4f}, improved={improved}, "
                        f"no_improve_epochs={no_improve_epochs}"
                    )

            if val_loss is not None:
                previous_lr = float(optimizer.param_groups[0]["lr"])
                scheduler.step(val_loss)
                current_lr = float(optimizer.param_groups[0]["lr"])
                if verbose and current_lr != previous_lr:
                    print(
                        f"[RNN] lr reduced at epoch {current_epoch}: "
                        f"{previous_lr:.2e} -> {current_lr:.2e}"
                    )

            if early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
                if verbose:
                    print(
                        f"[RNN] early stopping triggered at epoch {current_epoch}; "
                        f"best epoch={best_epoch}, best val loss={best_val_loss:.4f}"
                    )
                break

        if restore_best_weights and best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f"[RNN] restored best model weights from epoch {best_epoch}")

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

        if not context_ids:
            context_ids = [unk_id]

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

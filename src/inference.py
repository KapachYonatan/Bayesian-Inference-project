from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import tokenize_text
from src.hpylm import HPYLM
from src.rnn import NeuralAutocompleter, NextWordRNN


def _load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, object]]:
    with Path(vocab_path).open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if "word_to_id" in payload and "id_to_word" in payload:
        word_to_id = {str(k): int(v) for k, v in payload["word_to_id"].items()}
        id_to_word = {int(k): str(v) for k, v in payload["id_to_word"].items()}
        model_config = payload.get("model_config", {})
        return word_to_id, id_to_word, model_config

    # Backward-compatible fallback for plain dict formats.
    word_to_id = {str(k): int(v) for k, v in payload.items()}
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word, {}


def _infer_rnn_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int | str]:
    embedding_weight = state_dict["embedding.weight"]
    output_weight = state_dict["output.weight"]
    recurrent_ih = state_dict["recurrent.weight_ih_l0"]

    vocab_size = int(embedding_weight.shape[0])
    embed_dim = int(embedding_weight.shape[1])
    hidden_dim = int(output_weight.shape[1])

    gate_ratio = int(recurrent_ih.shape[0] // max(hidden_dim, 1))
    if gate_ratio == 4:
        cell_type = "lstm"
    elif gate_ratio == 3:
        cell_type = "gru"
    else:
        cell_type = "rnn"

    layer_pattern = re.compile(r"recurrent\.weight_ih_l(\d+)")
    layers = []
    for key in state_dict.keys():
        match = layer_pattern.match(key)
        if match:
            layers.append(int(match.group(1)))
    num_layers = max(layers) + 1 if layers else 1

    return {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "cell_type": cell_type,
    }


def load_artifacts(model_type: str, checkpoint_path: str, vocab_path: str):
    word_to_id, id_to_word, model_config = _load_vocab(vocab_path)

    if model_type == "hpylm":
        with Path(checkpoint_path).open("rb") as fp:
            model = pickle.load(fp)
        if not isinstance(model, HPYLM):
            raise TypeError("Loaded checkpoint is not an HPYLM instance.")
        return model, word_to_id, id_to_word, model_config

    if model_type == "rnn":
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        inferred = _infer_rnn_config_from_state_dict(state_dict)
        cfg = {
            "vocab_size": int(model_config.get("vocab_size", inferred["vocab_size"])),
            "embed_dim": int(model_config.get("embed_dim", inferred["embed_dim"])),
            "hidden_dim": int(model_config.get("hidden_dim", inferred["hidden_dim"])),
            "num_layers": int(model_config.get("num_layers", inferred["num_layers"])),
            "cell_type": str(model_config.get("cell_type", inferred["cell_type"])),
            "seq_len": model_config.get("seq_len"),
        }

        model = NextWordRNN(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            cell_type=cfg["cell_type"],
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model, word_to_id, id_to_word, cfg

    raise ValueError("model_type must be one of: 'hpylm', 'rnn'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive next-word inference for HPYLM/RNN checkpoints.")
    parser.add_argument("--model-type", choices=["hpylm", "rnn"], required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[Inference] CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return str(requested)


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    model, word_to_id, id_to_word, model_config = load_artifacts(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        vocab_path=args.vocab_path,
    )

    if args.model_type == "rnn":
        seq_len = args.seq_len if args.seq_len is not None else model_config.get("seq_len")
        completer = NeuralAutocompleter(
            model=model,
            word_to_id=word_to_id,
            id_to_word=id_to_word,
            device=resolved_device,
            seq_len=seq_len,
        )
        print(f"Loaded RNN model from {args.checkpoint_path} (seq_len={seq_len}, device={resolved_device})")
    else:
        completer = None
        print(f"Loaded HPYLM model from {args.checkpoint_path}")

    print("Type 'quit' to exit.")
    while True:
        text = input("Enter text: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue

        tokens = tokenize_text(text)
        if not tokens:
            print("Top-3 predictions: []")
            continue

        if args.model_type == "rnn":
            preds = completer.predict_next_word(tokens, top_k=3)
        else:
            preds = model.predict_next_word(tokens, word_to_id, id_to_word, top_k=3)

        print(f"Top-3 predictions: {preds}")


if __name__ == "__main__":
    main()

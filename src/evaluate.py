from __future__ import annotations

import argparse

from data_pipeline import get_hpylm_data, get_rnn_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HPYLM and RNN data readiness.")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hpylm_ids, word_to_id, _ = get_hpylm_data(vocab_size=args.vocab_size)
    rnn_bundle = get_rnn_dataloaders(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
    )

    print("=== Evaluation Readiness Summary ===")
    print(f"HPYLM token count: {len(hpylm_ids)}")
    print(f"Vocabulary size: {len(word_to_id)}")
    print(f"RNN train windows: {len(rnn_bundle.train_loader.dataset)}")
    print(f"RNN validation windows: {len(rnn_bundle.valid_loader.dataset)}")
    print(f"RNN test windows: {len(rnn_bundle.test_loader.dataset)}")


if __name__ == "__main__":
    main()

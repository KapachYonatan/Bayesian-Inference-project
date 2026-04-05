from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

# Ensure imports work when this file is executed directly from the tests folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import UNK_TOKEN, get_hpylm_data, get_rnn_dataloaders


def count_unknown(token_ids: Iterable[int], unk_id: int) -> int:
    return sum(1 for token_id in token_ids if token_id == unk_id)


def decode_ids(ids: List[int], id_to_word: dict[int, str]) -> List[str]:
    return [id_to_word.get(i, UNK_TOKEN) for i in ids]


def main() -> None:
    vocab_size = 10_000
    seq_len = 20
    batch_size = 32

    print("=== HPYLM PIPELINE CHECK ===")
    corpus_ids, word_to_id, id_to_word = get_hpylm_data(vocab_size=vocab_size)
    unk_id = word_to_id[UNK_TOKEN]
    num_tokens = len(corpus_ids)
    num_unknown = count_unknown(corpus_ids, unk_id)

    print(f"tokens: {num_tokens}")
    print(f"unknown tokens: {num_unknown}")
    print(f"vocab size (incl. {UNK_TOKEN}): {len(word_to_id)}")

    hpylm_ids_snippet = corpus_ids[:30]
    hpylm_words_snippet = decode_ids(hpylm_ids_snippet, id_to_word)
    print("HPYLM final-product snippet (ids):")
    print(hpylm_ids_snippet)
    print("HPYLM final-product snippet (decoded words):")
    print(" ".join(hpylm_words_snippet))

    print("\n=== RNN PIPELINE CHECK ===")
    rnn = get_rnn_dataloaders(
        seq_len=seq_len,
        vocab_size=vocab_size,
        batch_size=batch_size,
    )

    train_dataset = rnn.train_loader.dataset
    valid_dataset = rnn.valid_loader.dataset
    test_dataset = rnn.test_loader.dataset

    print(f"train windows: {len(train_dataset)}")
    print(f"validation windows: {len(valid_dataset)}")
    print(f"test windows: {len(test_dataset)}")

    x_batch, y_batch = next(iter(rnn.train_loader))
    print(f"batch input shape: {tuple(x_batch.shape)}")
    print(f"batch target shape: {tuple(y_batch.shape)}")

    x0_ids = x_batch[0].tolist()
    y0_id = int(y_batch[0].item())
    x0_words = decode_ids(x0_ids, rnn.id_to_word)
    y0_word = rnn.id_to_word.get(y0_id, UNK_TOKEN)

    print("RNN final-product snippet (first sample in first batch):")
    print(f"input ids: {x0_ids}")
    print(f"input words: {' '.join(x0_words)}")
    print(f"next-word target: id={y0_id}, word={y0_word}")


if __name__ == "__main__":
    main()

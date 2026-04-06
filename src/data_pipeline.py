from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


UNK_TOKEN = "<UNK>"
DATASET_NAME = "japhba/pubmed_3k"
TEXT_FIELD = "abstract"


def tokenize_text(text: str) -> List[str]:
    """Clean and tokenize raw text into lowercase word tokens."""
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def load_pubmed_tokens(split: str = "train") -> List[str]:
    """Load PubMed abstracts and return a flat token list for the given split."""
    return list(_load_pubmed_tokens_cached(split))


@lru_cache(maxsize=3)
def _load_pubmed_tokens_cached(split: str) -> Tuple[str, ...]:
    """Cached helper for loading and tokenizing PubMed abstract splits."""
    split_datasets = _load_pubmed_split_datasets()
    if split not in split_datasets:
        raise ValueError(f"Unsupported split '{split}'. Use one of: {tuple(split_datasets.keys())}")

    dataset = split_datasets[split]

    tokens: List[str] = []
    for item in dataset:
        text = _extract_text(item)
        line_tokens = tokenize_text(text)
        if line_tokens:
            tokens.extend(line_tokens)

    return tuple(tokens)


@lru_cache(maxsize=1)
def _load_pubmed_split_datasets() -> Dict[str, object]:
    """
    Load PubMed data and create deterministic train/validation/test splits.

    The source dataset provides a train split only, so we partition it as:
    - train: 90%
    - validation: 5%
    - test: 5%
    """
    full_train = load_dataset(DATASET_NAME, split="train")
    first_split = full_train.train_test_split(test_size=0.10, seed=42)
    train_ds = first_split["train"]
    holdout_ds = first_split["test"]

    second_split = holdout_ds.train_test_split(test_size=0.50, seed=42)
    validation_ds = second_split["train"]
    test_ds = second_split["test"]

    return {
        "train": train_ds,
        "validation": validation_ds,
        "test": test_ds,
    }


def _extract_text(item: Dict[str, object]) -> str:
    """Extract the text string to tokenize from a dataset row."""
    text = item.get(TEXT_FIELD)
    if isinstance(text, str):
        return text
    raise ValueError(
        f"Expected string field '{TEXT_FIELD}' in dataset row, but got {type(text).__name__}."
    )


def build_vocabulary(
    tokens: Sequence[str],
    min_freq: int = 3,
    unk_token: str = UNK_TOKEN,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary mappings using a minimum frequency cutoff.

    Index 0 is always reserved for unk_token.
    """
    if min_freq < 1:
        raise ValueError("min_freq must be >= 1")

    counter = Counter(tokens)

    word_to_id: Dict[str, int] = {unk_token: 0}
    next_id = 1
    for word, count in counter.items():
        if count >= min_freq:
            word_to_id[word] = next_id
            next_id += 1

    id_to_word: Dict[int, str] = {idx: word for word, idx in word_to_id.items()}
    return word_to_id, id_to_word


def encode_tokens(
    tokens: Iterable[str],
    word_to_id: Dict[str, int],
    unk_token: str = UNK_TOKEN,
) -> List[int]:
    """Map tokens to integer IDs, using unk_token for out-of-vocabulary words."""
    unk_id = word_to_id[unk_token]
    return [word_to_id.get(token, unk_id) for token in tokens]


def get_hpylm_data(min_freq: int = 3) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
    """
    Return PubMed train corpus as integer IDs for HPYLM.

    Vocabulary is built on the train split using a minimum frequency cutoff,
    and all out-of-vocabulary words are mapped to <UNK>.
    """
    train_tokens = load_pubmed_tokens(split="train")
    word_to_id, id_to_word = build_vocabulary(train_tokens, min_freq=min_freq)
    corpus_ids = encode_tokens(train_tokens, word_to_id)
    return corpus_ids, word_to_id, id_to_word


class NextWordDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset yielding (input_sequence, next_word_target) pairs."""

    def __init__(self, token_ids: Sequence[int], seq_len: int, stride: int = 1) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if len(token_ids) <= seq_len:
            raise ValueError("token_ids must contain more elements than seq_len")

        self.sequences: List[List[int]] = []
        self.targets: List[int] = []
        self.seq_len = seq_len
        self.stride = stride

        for i in range(0, len(token_ids) - seq_len, stride):
            self.sequences.append(list(token_ids[i : i + seq_len]))
            self.targets.append(token_ids[i + seq_len])

        if not self.sequences:
            raise ValueError("No training windows produced. Reduce seq_len or stride.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq = self.sequences[idx]
        target = self.targets[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


@dataclass
class RnnDataBundle:
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    word_to_id: Dict[str, int]
    id_to_word: Dict[int, str]


def get_rnn_dataloaders(
    seq_len: int,
    min_freq: int = 3,
    stride: int = 5,
    batch_size: int = 64,
    num_workers: int = 0,
) -> RnnDataBundle:
    """
    Build PyTorch DataLoaders for next-word prediction.

    Returns train/valid/test loaders where each batch is:
    - input_sequence: LongTensor of shape (batch_size, seq_len)
    - next_word_target: LongTensor of shape (batch_size,)
    """
    train_tokens = load_pubmed_tokens(split="train")
    valid_tokens = load_pubmed_tokens(split="validation")
    test_tokens = load_pubmed_tokens(split="test")

    word_to_id, id_to_word = build_vocabulary(train_tokens, min_freq=min_freq)

    train_ids = encode_tokens(train_tokens, word_to_id)
    valid_ids = encode_tokens(valid_tokens, word_to_id)
    test_ids = encode_tokens(test_tokens, word_to_id)

    train_dataset = NextWordDataset(train_ids, seq_len=seq_len, stride=stride)
    valid_dataset = NextWordDataset(valid_ids, seq_len=seq_len, stride=stride)
    test_dataset = NextWordDataset(test_ids, seq_len=seq_len, stride=stride)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return RnnDataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
    )

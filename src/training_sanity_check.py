from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import build_vocabulary, encode_tokens, get_rnn_dataloaders, load_pubmed_tokens
from src.evaluate import (
    calculate_hpylm_perplexity,
    calculate_hpylm_topk_accuracy,
    calculate_rnn_perplexity,
    calculate_rnn_topk_accuracy,
    config_save_dir,
    latest_rnn_checkpoint,
    measure_latency_ms,
    random_contexts_from_tokens,
    resolve_device,
    set_seed,
)
from src.hpylm import HPYLM
from src.rnn import NeuralAutocompleter, NextWordRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HPYLM and RNN on real data and check if metrics look reasonable."
    )
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--latency-samples", type=int, default=100)

    # HPYLM params
    parser.add_argument("--hpylm-order", type=int, default=3)
    parser.add_argument("--hpylm-discount", type=float, default=0.75)
    parser.add_argument("--hpylm-concentration", type=float, default=1.0)

    # RNN params
    parser.add_argument("--rnn-cell-type", choices=["rnn", "gru", "lstm"], default="gru")
    parser.add_argument("--rnn-hidden-dim", type=int, default=128)
    parser.add_argument("--rnn-embed-dim", type=int, default=128)
    parser.add_argument("--rnn-epochs", type=int, default=3)
    parser.add_argument("--rnn-early-stopping-patience", type=int, default=0)
    parser.add_argument("--rnn-early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--rnn-no-restore-best", action="store_true")

    # Save/resume
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--resume-training", action="store_true")

    # Optional explicit thresholds
    parser.add_argument("--max-perplexity", type=float, default=None)
    parser.add_argument("--min-recall3", type=float, default=None)
    parser.add_argument("--min-recall5", type=float, default=None)
    parser.add_argument("--max-latency-ms", type=float, default=None)

    return parser.parse_args()


def _format_markdown_table(rows: List[Dict[str, str]]) -> str:
    headers = ["Model", "Parameters", "Perplexity", "Recall@3", "Recall@5", "Latency (ms)", "Status"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    def fmt(values: List[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[h]) for value, h in zip(values, headers)) + " |"

    lines = [
        fmt(headers),
        "| " + " | ".join("-" * widths[h] for h in headers) + " |",
    ]
    for row in rows:
        lines.append(fmt([row[h] for h in headers]))
    return "\n".join(lines)


def _is_reasonable(
    *,
    perplexity: float,
    recall3: float,
    recall5: float,
    latency_ms: float,
    vocab_size: int,
    max_perplexity: float | None,
    min_recall3: float | None,
    min_recall5: float | None,
    max_latency_ms: float | None,
) -> tuple[bool, str]:
    # Defaults anchored to random baseline + practical tolerance.
    rand_recall3 = 3.0 / max(vocab_size, 1)
    rand_recall5 = 5.0 / max(vocab_size, 1)

    ppl_limit = max_perplexity if max_perplexity is not None else float(vocab_size)
    recall3_floor = min_recall3 if min_recall3 is not None else max(0.01, 5.0 * rand_recall3)
    recall5_floor = min_recall5 if min_recall5 is not None else max(0.02, 5.0 * rand_recall5)
    latency_limit = max_latency_ms if max_latency_ms is not None else 1000.0

    conditions = [
        (perplexity <= ppl_limit, f"perplexity<={ppl_limit:.4f}"),
        (recall3 >= recall3_floor, f"recall@3>={recall3_floor:.4f}"),
        (recall5 >= recall5_floor, f"recall@5>={recall5_floor:.4f}"),
        (latency_ms <= latency_limit, f"latency<={latency_limit:.4f}ms"),
    ]

    failed = [rule for ok, rule in conditions if not ok]
    return (len(failed) == 0, "OK" if not failed else "FAIL: " + ", ".join(failed))


def _train_and_eval_hpylm(
    args: argparse.Namespace,
    train_ids: List[int],
    test_ids: List[int],
    test_words: List[str],
    word_to_id: Dict[str, int],
    id_to_word: Dict[int, str],
) -> Dict[str, str]:
    hpylm_save_dir = config_save_dir(
        args.save_dir,
        "hpylm",
        {
            "order": args.hpylm_order,
            "discount": args.hpylm_discount,
            "concentration": args.hpylm_concentration,
        },
    )

    checkpoint_file = Path(hpylm_save_dir) / "hpylm_checkpoint.pkl" if hpylm_save_dir is not None else None
    if args.resume_training and checkpoint_file is not None and checkpoint_file.exists():
        with checkpoint_file.open("rb") as fp:
            model = pickle.load(fp)
        warm_start = True
        print(f"[Sanity][HPYLM] resumed from {checkpoint_file}")
    else:
        model = HPYLM(
            order=args.hpylm_order,
            vocab_size=len(word_to_id),
            discount=args.hpylm_discount,
            concentration=args.hpylm_concentration,
        )
        warm_start = False

    print("[Sanity][HPYLM] training started")
    model.fit(
        train_ids,
        num_gibbs_iterations=50,
        verbose=True,
        save_dir=hpylm_save_dir,
        warm_start=warm_start,
    )

    perplexity = calculate_hpylm_perplexity(model, test_ids)
    recall3, recall5 = calculate_hpylm_topk_accuracy(model, test_ids, id_to_word)
    contexts = random_contexts_from_tokens(
        test_words,
        context_len=max(args.seq_len, 1),
        sample_count=args.latency_samples,
        seed=args.seed,
    )
    latency = measure_latency_ms(
        lambda context: model.predict_next_word(context, word_to_id, id_to_word, top_k=3),
        contexts,
        device=None,
    )

    ok, reason = _is_reasonable(
        perplexity=perplexity,
        recall3=recall3,
        recall5=recall5,
        latency_ms=latency,
        vocab_size=len(word_to_id),
        max_perplexity=args.max_perplexity,
        min_recall3=args.min_recall3,
        min_recall5=args.min_recall5,
        max_latency_ms=args.max_latency_ms,
    )

    return {
        "Model": "HPYLM",
        "Parameters": (
            f"order={args.hpylm_order}, discount={args.hpylm_discount}, "
            f"concentration={args.hpylm_concentration}, iterations=50"
        ),
        "Perplexity": f"{perplexity:.4f}",
        "Recall@3": f"{recall3:.4f}",
        "Recall@5": f"{recall5:.4f}",
        "Latency (ms)": f"{latency:.4f}",
        "Status": "PASS" if ok else reason,
    }


def _train_and_eval_rnn(
    args: argparse.Namespace,
    bundle,
    test_words: List[str],
    device,
) -> Dict[str, str]:
    model = NextWordRNN(
        vocab_size=len(bundle.word_to_id),
        embed_dim=args.rnn_embed_dim,
        hidden_dim=args.rnn_hidden_dim,
        cell_type=args.rnn_cell_type,
    )
    completer = NeuralAutocompleter(
        model=model,
        word_to_id=bundle.word_to_id,
        id_to_word=bundle.id_to_word,
        device=str(device),
        seq_len=args.seq_len,
    )

    rnn_save_dir = config_save_dir(
        args.save_dir,
        "rnn",
        {
            "cell_type": args.rnn_cell_type,
            "hidden_dim": args.rnn_hidden_dim,
            "embed_dim": args.rnn_embed_dim,
        },
    )
    resume_checkpoint = latest_rnn_checkpoint(rnn_save_dir) if args.resume_training else None
    if resume_checkpoint is not None:
        print(f"[Sanity][RNN] resumed from {resume_checkpoint}")

    print("[Sanity][RNN] training started")
    completer.fit(
        dataloader=bundle.train_loader,
        epochs=max(1, args.rnn_epochs),
        lr=1e-3,
        verbose=True,
        save_dir=rnn_save_dir,
        resume_checkpoint=resume_checkpoint,
        valid_dataloader=bundle.valid_loader,
        early_stopping_patience=args.rnn_early_stopping_patience,
        early_stopping_min_delta=args.rnn_early_stopping_min_delta,
        restore_best_weights=not args.rnn_no_restore_best,
    )

    perplexity = calculate_rnn_perplexity(completer, bundle.test_loader)
    recall3, recall5 = calculate_rnn_topk_accuracy(completer, bundle.test_loader)
    contexts = random_contexts_from_tokens(
        test_words,
        context_len=max(args.seq_len, 1),
        sample_count=args.latency_samples,
        seed=args.seed,
    )
    latency = measure_latency_ms(
        lambda context: completer.predict_next_word(context, top_k=3),
        contexts,
        device=device,
    )

    ok, reason = _is_reasonable(
        perplexity=perplexity,
        recall3=recall3,
        recall5=recall5,
        latency_ms=latency,
        vocab_size=len(bundle.word_to_id),
        max_perplexity=args.max_perplexity,
        min_recall3=args.min_recall3,
        min_recall5=args.min_recall5,
        max_latency_ms=args.max_latency_ms,
    )

    return {
        "Model": "RNN",
        "Parameters": (
            f"cell_type={args.rnn_cell_type}, hidden_dim={args.rnn_hidden_dim}, "
            f"embed_dim={args.rnn_embed_dim}, epochs={args.rnn_epochs}"
        ),
        "Perplexity": f"{perplexity:.4f}",
        "Recall@3": f"{recall3:.4f}",
        "Recall@5": f"{recall5:.4f}",
        "Latency (ms)": f"{latency:.4f}",
        "Status": "PASS" if ok else reason,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    device = resolve_device(args.device)
    print(f"[Sanity] using device: {device}")

    print("[Sanity] loading real dataset splits")
    train_words = load_pubmed_tokens(split="train")
    test_words = load_pubmed_tokens(split="test")

    word_to_id, id_to_word = build_vocabulary(train_words, min_freq=args.min_freq)
    train_ids = encode_tokens(train_words, word_to_id)
    test_ids = encode_tokens(test_words, word_to_id)

    bundle = get_rnn_dataloaders(
        seq_len=args.seq_len,
        min_freq=args.min_freq,
        batch_size=args.batch_size,
    )

    hpylm_row = _train_and_eval_hpylm(
        args,
        train_ids=train_ids,
        test_ids=test_ids,
        test_words=test_words,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
    )

    rnn_row = _train_and_eval_rnn(
        args,
        bundle=bundle,
        test_words=test_words,
        device=device,
    )

    print("# Training Sanity Check Results")
    print(_format_markdown_table([hpylm_row, rnn_row]))


if __name__ == "__main__":
    main()

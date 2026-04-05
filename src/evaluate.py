from __future__ import annotations

import argparse
import itertools
import math
import pickle
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import (
    UNK_TOKEN,
    build_vocabulary,
    encode_tokens,
    get_hpylm_data,
    get_rnn_dataloaders,
    load_pubmed_tokens,
)
from src.hpylm import HPYLM
from src.rnn import NeuralAutocompleter, NextWordRNN


HPYLM_ORDER_GRID = [2, 3, 5]
HPYLM_DISCOUNT_GRID = [0.5, 0.75, 0.9]
HPYLM_CONCENTRATION_GRID = [1.0, 5.0]
RNN_CELL_TYPE_GRID = ["lstm", "gru"]
RNN_HIDDEN_DIM_GRID = [128, 256, 512]
RNN_EMBED_DIM_GRID = [128, 300]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HPYLM and RNN on perplexity and latency.")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hpylm-iterations", type=int, default=5)
    parser.add_argument("--rnn-epochs", type=int, default=3)
    parser.add_argument("--max-train-tokens", type=int, default=10_000)
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for RNN training/eval: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--quick-sweep",
        action="store_true",
        help="Run a reduced hyperparameter sweep for faster smoke testing.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional root directory for saving model checkpoints during sweeps.",
    )
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume from latest checkpoints in --save-dir when available.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[Eval] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return requested


def config_save_dir(base_dir: str | None, model_name: str, params: Dict[str, object]) -> str | None:
    if base_dir is None:
        return None

    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(f"{key}-{value}" for key, value in params.items())
    run_dir = root / model_name / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def latest_rnn_checkpoint(run_dir: str | None) -> str | None:
    if run_dir is None:
        return None
    path = Path(run_dir)
    if not path.exists():
        return None

    latest_file = path / "rnn_latest.pt"
    if latest_file.exists():
        return str(latest_file)

    epoch_files = sorted(path.glob("rnn_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    if not epoch_files:
        return None
    return str(epoch_files[-1])


def selected_hpylm_grid(args: argparse.Namespace) -> Tuple[List[int], List[float], List[float]]:
    if args.quick_sweep:
        return [2], [0.75], [1.0]
    return HPYLM_ORDER_GRID, HPYLM_DISCOUNT_GRID, HPYLM_CONCENTRATION_GRID


def selected_rnn_grid(args: argparse.Namespace) -> Tuple[List[str], List[int], List[int]]:
    if args.quick_sweep:
        return ["gru"], [128], [128]
    return RNN_CELL_TYPE_GRID, RNN_HIDDEN_DIM_GRID, RNN_EMBED_DIM_GRID


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def calculate_hpylm_perplexity(model: HPYLM, token_ids: Sequence[int]) -> float:
    """Calculate perplexity over a token sequence using HPYLM next-token probabilities."""
    if len(token_ids) < 2:
        return float("inf")

    log_prob_sum = 0.0
    token_count = 0

    for index in range(1, len(token_ids)):
        context_start = max(0, index - (model.order - 1))
        context = tuple(token_ids[context_start:index])
        dish = token_ids[index]
        restaurant = model._find_existing_restaurant(context)
        prob = restaurant.predictive_prob(dish, model.discount, model.concentration)
        log_prob_sum += math.log(max(prob, 1e-12))
        token_count += 1

    return math.exp(-log_prob_sum / max(token_count, 1))


def calculate_hpylm_topk_accuracy(
    model: HPYLM,
    token_ids: Sequence[int],
    id_to_word: Dict[int, str],
) -> Tuple[float, float]:
    """Return Top-1 and Top-3 next-token accuracy for HPYLM."""
    if len(token_ids) < 2:
        return 0.0, 0.0

    top1_hits = 0
    top3_hits = 0
    total = 0
    for index in range(1, len(token_ids)):
        context_start = max(0, index - (model.order - 1))
        context = tuple(token_ids[context_start:index])
        restaurant = model._find_existing_restaurant(context)
        probs, unseen_prob = model._sparse_predictive_distribution(restaurant)
        support = set(probs.keys())

        scored = [(prob, token_id) for token_id, prob in probs.items()]
        scored.extend(
            (unseen_prob, token_id)
            for token_id in model._candidate_unseen_ids(support, id_to_word, 3)
        )
        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = [token_id for _, token_id in scored[:3]]

        target = token_ids[index]
        if top_ids and target == top_ids[0]:
            top1_hits += 1
        if target in top_ids:
            top3_hits += 1
        total += 1

    if total == 0:
        return 0.0, 0.0
    return top1_hits / total, top3_hits / total


def calculate_rnn_perplexity(completer: NeuralAutocompleter, dataloader) -> float:
    """Calculate perplexity for an RNN model over a dataloader."""
    import torch

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_targets = 0

    completer.model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(completer.device)
            targets = targets.to(completer.device)
            logits, _ = completer.model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item())
            total_targets += int(targets.numel())

    if total_targets == 0:
        return float("inf")
    return math.exp(total_loss / total_targets)


def calculate_rnn_topk_accuracy(completer: NeuralAutocompleter, dataloader) -> Tuple[float, float]:
    """Return Top-1 and Top-3 next-token accuracy for RNN."""
    import torch

    top1_hits = 0
    top3_hits = 0
    total = 0

    completer.model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(completer.device)
            targets = targets.to(completer.device)
            logits, _ = completer.model(inputs)
            top1 = torch.argmax(logits, dim=-1)
            top3 = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1).indices

            top1_hits += int((top1 == targets).sum().item())
            top3_hits += int((top3 == targets.unsqueeze(1)).any(dim=1).sum().item())
            total += int(targets.numel())

    if total == 0:
        return 0.0, 0.0
    return top1_hits / total, top3_hits / total


def measure_latency_ms(predict_fn, contexts: Iterable[Sequence[str]], device=None) -> float:
    """Measure average latency in milliseconds over a stream of contexts."""
    import torch

    samples: List[float] = []
    for context in contexts:
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        predict_fn(list(context))
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples.append(elapsed_ms)
    return statistics.mean(samples) if samples else float("inf")


def random_contexts_from_tokens(tokens: Sequence[str], context_len: int, sample_count: int, seed: int) -> List[List[str]]:
    if len(tokens) <= context_len:
        return []
    rng = random.Random(seed)
    contexts: List[List[str]] = []
    upper_bound = len(tokens) - context_len - 1
    for _ in range(sample_count):
        start = rng.randint(0, max(upper_bound, 0))
        contexts.append(list(tokens[start : start + context_len]))
    return contexts


def format_markdown_table(rows: List[Dict[str, str]]) -> str:
    headers = ["Model", "Parameters", "Perplexity", "Top-1 Accuracy", "Top-3 Accuracy", "Latency (ms)"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    def fmt_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[header]) for value, header in zip(values, headers)) + " |"

    separator = "| " + " | ".join("-" * widths[header] for header in headers) + " |"
    lines = [fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row([row[header] for header in headers]))
    return "\n".join(lines)


def evaluate_hpylm_sweep(
    args: argparse.Namespace,
    test_tokens: Sequence[int],
    test_words: Sequence[str],
    word_to_id: Dict[str, int],
    id_to_word: Dict[int, str],
) -> List[Dict[str, str]]:
    corpus_ids, _, _ = get_hpylm_data(vocab_size=args.vocab_size)
    trained_corpus = corpus_ids[: min(len(corpus_ids), args.max_train_tokens)]
    order_grid, discount_grid, concentration_grid = selected_hpylm_grid(args)
    contexts = random_contexts_from_tokens(
        test_words,
        context_len=max(args.seq_len, 1),
        sample_count=args.latency_samples if not args.quick_sweep else min(args.latency_samples, 10),
        seed=args.seed,
    )

    rows: List[Dict[str, str]] = []
    total_runs = len(order_grid) * len(discount_grid) * len(concentration_grid)
    print(f"[Eval][HPYLM] starting sweep with {total_runs} configuration(s)")
    print(f"[Eval][HPYLM] training corpus tokens: {len(trained_corpus)}")
    for order, discount, concentration in itertools.product(order_grid, discount_grid, concentration_grid):
        print(
            f"[Eval][HPYLM] training config order={order}, discount={discount}, "
            f"concentration={concentration}"
        )
        hpylm_save_dir = config_save_dir(
            args.save_dir,
            "hpylm",
            {"order": order, "discount": discount, "concentration": concentration},
        )
        if hpylm_save_dir is not None:
            print(f"[Eval][HPYLM] checkpoints -> {hpylm_save_dir}")

        checkpoint_file = Path(hpylm_save_dir) / "hpylm_checkpoint.pkl" if hpylm_save_dir is not None else None
        if args.resume_training and checkpoint_file is not None and checkpoint_file.exists():
            with checkpoint_file.open("rb") as fp:
                model = pickle.load(fp)
            if not isinstance(model, HPYLM):
                raise TypeError(f"Invalid HPYLM checkpoint at {checkpoint_file}")
            warm_start = True
            print(f"[Eval][HPYLM] resumed from checkpoint: {checkpoint_file}")
        else:
            model = HPYLM(
                order=order,
                vocab_size=args.vocab_size,
                discount=discount,
                concentration=concentration,
            )
            warm_start = False

        model.fit(
            trained_corpus,
            num_gibbs_iterations=max(1, args.hpylm_iterations if not args.quick_sweep else 1),
            verbose=args.quick_sweep,
            save_dir=hpylm_save_dir,
            warm_start=warm_start,
        )
        perplexity = calculate_hpylm_perplexity(model, test_tokens)
        top1_acc, top3_acc = calculate_hpylm_topk_accuracy(model, test_tokens, id_to_word)
        latency = measure_latency_ms(
            lambda context: model.predict_next_word(context, word_to_id, id_to_word, top_k=3),
            contexts,
            device=None,
        )
        params = f"order={order}, discount={discount}, concentration={concentration}"
        rows.append(
            {
                "Model": "HPYLM",
                "Parameters": params,
                "Perplexity": f"{perplexity:.4f}",
                "Top-1 Accuracy": f"{top1_acc:.4f}",
                "Top-3 Accuracy": f"{top3_acc:.4f}",
                "Latency (ms)": f"{latency:.4f}",
            }
        )
        print(
            f"[Eval][HPYLM] finished config order={order}, discount={discount}, "
            f"concentration={concentration} -> perplexity={perplexity:.4f}, "
            f"top1={top1_acc:.4f}, top3={top3_acc:.4f}, latency={latency:.4f} ms"
        )
    print("[Eval][HPYLM] sweep complete")
    return rows


def evaluate_rnn_sweep(
    args: argparse.Namespace,
    bundle,
    test_words: Sequence[str],
    device,
) -> List[Dict[str, str]]:
    import torch

    rows: List[Dict[str, str]] = []
    cell_grid, hidden_grid, embed_grid = selected_rnn_grid(args)
    total_runs = len(cell_grid) * len(hidden_grid) * len(embed_grid)
    print(f"[Eval][RNN] starting sweep with {total_runs} configuration(s)")
    for cell_type, hidden_dim, embed_dim in itertools.product(cell_grid, hidden_grid, embed_grid):
        print(
            f"[Eval][RNN] creating model cell_type={cell_type}, hidden_dim={hidden_dim}, "
            f"embed_dim={embed_dim}"
        )
        model = NextWordRNN(
            vocab_size=len(bundle.word_to_id),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            cell_type=cell_type,
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
            {"cell_type": cell_type, "hidden_dim": hidden_dim, "embed_dim": embed_dim},
        )
        if rnn_save_dir is not None:
            print(f"[Eval][RNN] checkpoints -> {rnn_save_dir}")

        resume_checkpoint = latest_rnn_checkpoint(rnn_save_dir) if args.resume_training else None
        if resume_checkpoint is not None:
            print(f"[Eval][RNN] resumed from checkpoint: {resume_checkpoint}")

        completer.fit(
            dataloader=bundle.train_loader,
            epochs=max(1, args.rnn_epochs if not args.quick_sweep else 1),
            lr=1e-3,
            verbose=args.quick_sweep,
            save_dir=rnn_save_dir,
            resume_checkpoint=resume_checkpoint,
        )
        perplexity = calculate_rnn_perplexity(completer, bundle.test_loader)
        top1_acc, top3_acc = calculate_rnn_topk_accuracy(completer, bundle.test_loader)

        test_contexts = random_contexts_from_tokens(
            test_words,
            context_len=max(args.seq_len, 1),
            sample_count=args.latency_samples if not args.quick_sweep else min(args.latency_samples, 10),
            seed=args.seed,
        )

        latency = measure_latency_ms(
            lambda context: completer.predict_next_word(context, top_k=3),
            test_contexts,
            device=device,
        )
        params = f"cell_type={cell_type}, hidden_dim={hidden_dim}, embed_dim={embed_dim}"
        rows.append(
            {
                "Model": "RNN",
                "Parameters": params,
                "Perplexity": f"{perplexity:.4f}",
                "Top-1 Accuracy": f"{top1_acc:.4f}",
                "Top-3 Accuracy": f"{top3_acc:.4f}",
                "Latency (ms)": f"{latency:.4f}",
            }
        )
        print(
            f"[Eval][RNN] finished config cell_type={cell_type}, hidden_dim={hidden_dim}, "
            f"embed_dim={embed_dim} -> perplexity={perplexity:.4f}, top1={top1_acc:.4f}, "
            f"top3={top3_acc:.4f}, latency={latency:.4f} ms"
        )
        del model
        del completer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("[Eval][RNN] sweep complete")
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[Eval] using device: {device}")

    # Load data once for test-set token sequences and contexts.
    train_tokens = load_pubmed_tokens(split="train")
    test_words = load_pubmed_tokens(split="test")
    word_to_id, id_to_word = build_vocabulary(train_tokens, vocab_size=args.vocab_size)
    test_ids = encode_tokens(test_words, word_to_id)
    bundle = get_rnn_dataloaders(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
    )

    hpylm_rows = evaluate_hpylm_sweep(args, test_ids, test_words, word_to_id, id_to_word)
    rnn_rows = evaluate_rnn_sweep(args, bundle, test_words, device)
    rows = hpylm_rows + rnn_rows

    print("# Evaluation Results")
    if args.quick_sweep:
        print("Quick sweep enabled: reduced grids and fewer training/latency samples.")
    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()

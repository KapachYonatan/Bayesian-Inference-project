from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data_pipeline import UNK_TOKEN, get_hpylm_data


class Table:
    def __init__(self, dish: int):
        self.dish: int = dish
        self.customers: int = 0


class Restaurant:
    # Set by HPYLM so root can compute a uniform base distribution.
    base_vocab_size: int = 1

    def __init__(self, context: Tuple[int, ...], parent: Optional["Restaurant"] = None):
        self.context: Tuple[int, ...] = context
        self.parent: Optional[Restaurant] = parent
        self.tables: List[Table] = []
        self.total_customers: int = 0
        self.total_tables: int = 0
        self.dish_customer_counts: Dict[int, int] = {}
        self.dish_table_counts: Dict[int, int] = {}

    def _dish_customers(self, dish: int) -> int:
        return self.dish_customer_counts.get(dish, 0)

    def _dish_tables(self, dish: int) -> int:
        return self.dish_table_counts.get(dish, 0)

    def predictive_prob(self, dish: int, discount: float, concentration: float) -> float:
        """Recursive HPYLM predictive probability for a dish in this context."""
        if self.parent is None:
            parent_prob = 1.0 / float(max(self.base_vocab_size, 1))
        else:
            parent_prob = self.parent.predictive_prob(dish, discount, concentration)

        if self.total_customers == 0:
            return parent_prob

        n_w = self._dish_customers(dish)
        t_w = self._dish_tables(dish)
        n = self.total_customers
        t = self.total_tables
        observed_mass = max(n_w - discount * t_w, 0.0)
        backoff_mass = (concentration + discount * t) * parent_prob
        return (observed_mass + backoff_mass) / (n + concentration)

    def add_customer(self, dish: int, discount: float, concentration: float) -> bool:
        """Seat a customer under CRP seating dynamics. Return True if a new table is created."""
        candidates: List[Tuple[float, Optional[Table]]] = []
        total_mass = 0.0

        for table in self.tables:
            if table.dish != dish:
                continue
            mass = max(table.customers - discount, 0.0)
            if mass > 0:
                candidates.append((mass, table))
                total_mass += mass

        new_table_mass = concentration + discount * self.total_tables
        candidates.append((new_table_mass, None))
        total_mass += new_table_mass

        draw = random.random() * total_mass
        accum = 0.0
        chosen_table: Optional[Table] = None
        for mass, table in candidates:
            accum += mass
            if draw <= accum:
                chosen_table = table
                break

        self.total_customers += 1
        if chosen_table is not None:
            chosen_table.customers += 1
            self.dish_customer_counts[dish] = self.dish_customer_counts.get(dish, 0) + 1
            return False

        new_table = Table(dish=dish)
        new_table.customers = 1
        self.tables.append(new_table)
        self.total_tables += 1
        self.dish_customer_counts[dish] = self.dish_customer_counts.get(dish, 0) + 1
        self.dish_table_counts[dish] = self.dish_table_counts.get(dish, 0) + 1
        if self.parent is not None:
            self.parent.add_customer(dish, discount, concentration)
        return True

    def remove_customer(self, dish: int) -> bool:
        """Remove a customer assignment. Return True if a table is destroyed."""
        matching_tables = [table for table in self.tables if table.dish == dish and table.customers > 0]
        if not matching_tables:
            return False

        total_dish_customers = sum(table.customers for table in matching_tables)
        draw = random.randint(1, total_dish_customers)
        accum = 0
        chosen: Optional[Table] = None
        for table in matching_tables:
            accum += table.customers
            if draw <= accum:
                chosen = table
                break

        if chosen is None:
            return False

        chosen.customers -= 1
        self.total_customers -= 1
        self.dish_customer_counts[dish] = self.dish_customer_counts.get(dish, 0) - 1
        if self.dish_customer_counts[dish] <= 0:
            del self.dish_customer_counts[dish]
        if chosen.customers > 0:
            return False

        self.tables.remove(chosen)
        self.total_tables -= 1
        self.dish_table_counts[dish] = self.dish_table_counts.get(dish, 0) - 1
        if self.dish_table_counts[dish] <= 0:
            del self.dish_table_counts[dish]
        if self.parent is not None:
            self.parent.remove_customer(dish)
        return True


class HPYLM:
    def __init__(self, order: int, vocab_size: int, discount: float = 0.75, concentration: float = 1.0):
        if order < 1:
            raise ValueError("order must be >= 1")
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")

        self.order: int = order
        self.vocab_size: int = vocab_size
        self.discount: float = discount
        self.concentration: float = concentration
        Restaurant.base_vocab_size = vocab_size

        self.root: Restaurant = Restaurant(context=())
        self.context_trie: Dict[Tuple[int, ...], Restaurant] = {(): self.root}

    def _context_for_index(self, tokenized_corpus: List[int], index: int) -> Tuple[int, ...]:
        start = max(0, index - (self.order - 1))
        return tuple(tokenized_corpus[start:index])

    def _get_or_create_restaurant(self, context: Tuple[int, ...]) -> Restaurant:
        if context in self.context_trie:
            return self.context_trie[context]

        if not context:
            return self.root

        parent_context = context[1:]
        parent = self._get_or_create_restaurant(parent_context)
        restaurant = Restaurant(context=context, parent=parent)
        self.context_trie[context] = restaurant
        return restaurant

    def _find_existing_restaurant(self, context: Tuple[int, ...]) -> Restaurant:
        """Back off to shorter contexts until a known restaurant is found."""
        probe = context
        while probe:
            restaurant = self.context_trie.get(probe)
            if restaurant is not None:
                return restaurant
            probe = probe[1:]
        return self.root

    def _context_path(self, restaurant: Restaurant) -> List[Restaurant]:
        """Return path from root to the target restaurant."""
        path: List[Restaurant] = []
        current: Optional[Restaurant] = restaurant
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def _sparse_predictive_distribution(
        self,
        restaurant: Restaurant,
    ) -> Tuple[Dict[int, float], float]:
        """
        Compute predictive probabilities only for dishes observed on the context path.

        Returns:
        - probs: P(w|context) for observed dishes along context->root backoff path
        - unseen_prob: P(w|context) for any dish unseen on that full path
        """
        path = self._context_path(restaurant)
        support: set[int] = set()
        for node in path:
            support.update(node.dish_customer_counts.keys())

        base_prob = 1.0 / float(max(self.vocab_size, 1))
        probs: Dict[int, float] = {dish: base_prob for dish in support}
        unseen_prob = base_prob

        for node in path:
            if node.total_customers == 0:
                continue

            n = node.total_customers
            t = node.total_tables
            denom = n + self.concentration
            backoff_coeff = (self.concentration + self.discount * t) / denom

            next_probs: Dict[int, float] = {}
            for dish in support:
                n_w = node.dish_customer_counts.get(dish, 0)
                t_w = node.dish_table_counts.get(dish, 0)
                observed_mass = max(n_w - self.discount * t_w, 0.0)
                next_probs[dish] = (observed_mass + backoff_coeff * probs[dish])

            probs = next_probs
            unseen_prob = backoff_coeff * unseen_prob

        return probs, unseen_prob

    def _candidate_unseen_ids(
        self,
        support: set[int],
        id_to_word: Dict[int, str],
        max_count: int,
    ) -> List[int]:
        """Return a small set of unseen token IDs without scanning range(vocab_size)."""
        candidates: List[int] = []
        for token_id in id_to_word.keys():
            if token_id in support:
                continue
            candidates.append(token_id)
            if len(candidates) >= max_count:
                break
        return candidates

    def _sequence_perplexity(self, tokenized_corpus: List[int]) -> float:
        if len(tokenized_corpus) < 2:
            return float("inf")

        import math

        log_prob_sum = 0.0
        token_count = 0
        for index in range(1, len(tokenized_corpus)):
            context_start = max(0, index - (self.order - 1))
            context = tuple(tokenized_corpus[context_start:index])
            dish = tokenized_corpus[index]
            restaurant = self._find_existing_restaurant(context)
            prob = restaurant.predictive_prob(dish, self.discount, self.concentration)
            log_prob_sum += math.log(max(prob, 1e-12))
            token_count += 1

        return math.exp(-log_prob_sum / max(token_count, 1))

    def fit(
        self,
        tokenized_corpus: List[int],
        num_gibbs_iterations: int = 30,
        verbose: bool = False,
        save_dir: Optional[str] = None,
        warm_start: bool = False,
        valid_tokenized_corpus: Optional[List[int]] = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        eval_interval: int = 1,
        restore_best_state: bool = True,
    ) -> None:
        if early_stopping_patience > 0 and not valid_tokenized_corpus:
            raise ValueError("valid_tokenized_corpus is required when early_stopping_patience > 0")
        if eval_interval < 1:
            raise ValueError("eval_interval must be >= 1")

        # Initialize seating arrangement.
        if verbose:
            print(
                f"[HPYLM] fitting order={self.order}, vocab_size={self.vocab_size}, "
                f"discount={self.discount}, concentration={self.concentration}, "
                f"tokens={len(tokenized_corpus)}, gibbs_iterations={num_gibbs_iterations}, "
                f"warm_start={warm_start}, early_stopping_patience={early_stopping_patience}, "
                f"min_delta={early_stopping_min_delta}, eval_interval={eval_interval}"
            )
        if not warm_start:
            for idx, dish in enumerate(tokenized_corpus):
                context = self._context_for_index(tokenized_corpus, idx)
                restaurant = self._get_or_create_restaurant(context)
                restaurant.add_customer(dish, self.discount, self.concentration)
        elif verbose:
            print("[HPYLM] warm-start enabled: skipping seating initialization")

        if verbose:
            print(f"[HPYLM] initialization complete with {len(self.context_trie)} restaurants")

        checkpoint_file: Optional[Path] = None
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            checkpoint_file = save_path / "hpylm_checkpoint.pkl"

        best_val_perplexity = float("inf")
        best_iteration = 0
        no_improve_iterations = 0
        best_snapshot: Optional[bytes] = None

        # Gibbs resampling over all token positions.
        for iteration in range(num_gibbs_iterations):
            if verbose:
                print(f"[HPYLM] gibbs iteration {iteration + 1}/{num_gibbs_iterations}")
            for idx, dish in enumerate(tokenized_corpus):
                context = self._context_for_index(tokenized_corpus, idx)
                restaurant = self._get_or_create_restaurant(context)
                restaurant.remove_customer(dish)
                restaurant.add_customer(dish, self.discount, self.concentration)

            if checkpoint_file is not None:
                with checkpoint_file.open("wb") as fp:
                    pickle.dump(self, fp)
                if verbose:
                    print(f"[HPYLM] checkpoint saved: {checkpoint_file}")

            should_eval = valid_tokenized_corpus is not None and ((iteration + 1) % eval_interval == 0)
            if should_eval:
                val_perplexity = self._sequence_perplexity(valid_tokenized_corpus)
                improved = val_perplexity < (best_val_perplexity - early_stopping_min_delta)
                if improved:
                    best_val_perplexity = val_perplexity
                    best_iteration = iteration + 1
                    no_improve_iterations = 0
                    if restore_best_state:
                        best_snapshot = pickle.dumps(self)
                else:
                    no_improve_iterations += 1

                if verbose:
                    print(
                        f"[HPYLM] validation perplexity at iter {iteration + 1}: {val_perplexity:.4f}, "
                        f"improved={improved}, no_improve={no_improve_iterations}"
                    )

                if early_stopping_patience > 0 and no_improve_iterations >= early_stopping_patience:
                    if verbose:
                        print(
                            f"[HPYLM] early stopping triggered at iteration {iteration + 1}; "
                            f"best iteration={best_iteration}, best val perplexity={best_val_perplexity:.4f}"
                        )
                    break

        if restore_best_state and best_snapshot is not None:
            best_model: "HPYLM" = pickle.loads(best_snapshot)
            self.__dict__ = best_model.__dict__
            if verbose:
                print(f"[HPYLM] restored best state from iteration {best_iteration}")

        if checkpoint_file is not None:
            with checkpoint_file.open("wb") as fp:
                pickle.dump(self, fp)
            if verbose:
                print(f"[HPYLM] final checkpoint saved: {checkpoint_file}")

        if verbose:
            print("[HPYLM] training complete")

    def predict_next_word(
        self,
        context: List[str],
        word_to_id: Dict[str, int],
        id_to_word: Dict[int, str],
        top_k: int = 3,
    ) -> List[str]:
        # Convert context words to IDs and use last order-1 tokens.
        unk_id = word_to_id.get(UNK_TOKEN, 0)
        context_ids = [word_to_id.get(token, unk_id) for token in context]
        truncated_context = tuple(context_ids[-(self.order - 1) :]) if self.order > 1 else ()
        restaurant = self._find_existing_restaurant(truncated_context)

        probs, unseen_prob = self._sparse_predictive_distribution(restaurant)
        support = set(probs.keys())

        observed_candidates = [(prob, token_id) for token_id, prob in probs.items()]
        unseen_candidates = [
            (unseen_prob, token_id)
            for token_id in self._candidate_unseen_ids(support, id_to_word, max(top_k, 1))
        ]

        scored = observed_candidates + unseen_candidates
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_ids = [token_id for _, token_id in scored[: max(top_k, 1)]]
        return [id_to_word.get(token_id, UNK_TOKEN) for token_id in top_ids]

def train_hpylm(
    order: int = 3,
    vocab_size: int = 10_000,
    discount: float = 0.75,
    concentration: float = 1.0,
    num_gibbs_iterations: int = 30,
) -> Tuple[HPYLM, Dict[str, int], Dict[int, str]]:
    """Load corpus IDs, fit HPYLM, and return model with vocabulary mappings."""
    corpus_ids, word_to_id, id_to_word = get_hpylm_data(vocab_size=vocab_size)
    model = HPYLM(
        order=order,
        vocab_size=vocab_size,
        discount=discount,
        concentration=concentration,
    )
    model.fit(corpus_ids, num_gibbs_iterations=num_gibbs_iterations)
    return model, word_to_id, id_to_word

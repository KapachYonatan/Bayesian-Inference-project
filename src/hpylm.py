from __future__ import annotations

import random
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

    def _dish_customers(self, dish: int) -> int:
        return sum(table.customers for table in self.tables if table.dish == dish)

    def _dish_tables(self, dish: int) -> int:
        return sum(1 for table in self.tables if table.dish == dish)

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
            return False

        new_table = Table(dish=dish)
        new_table.customers = 1
        self.tables.append(new_table)
        self.total_tables += 1
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
        if chosen.customers > 0:
            return False

        self.tables.remove(chosen)
        self.total_tables -= 1
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

    def fit(self, tokenized_corpus: List[int], num_gibbs_iterations: int = 30) -> None:
        # Initialize seating arrangement.
        for idx, dish in enumerate(tokenized_corpus):
            context = self._context_for_index(tokenized_corpus, idx)
            restaurant = self._get_or_create_restaurant(context)
            restaurant.add_customer(dish, self.discount, self.concentration)

        # Gibbs resampling over all token positions.
        for _ in range(num_gibbs_iterations):
            for idx, dish in enumerate(tokenized_corpus):
                context = self._context_for_index(tokenized_corpus, idx)
                restaurant = self._get_or_create_restaurant(context)
                restaurant.remove_customer(dish)
                restaurant.add_customer(dish, self.discount, self.concentration)

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

        scored: List[Tuple[float, int]] = []
        for token_id in range(self.vocab_size):
            prob = restaurant.predictive_prob(token_id, self.discount, self.concentration)
            scored.append((prob, token_id))

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

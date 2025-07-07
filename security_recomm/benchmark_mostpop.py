from collections import Counter
from typing import Dict, List, Set
import torch

"""
MostPop baseline — Section 4.3
==============================
Selects top-K most interacted items *globally* over entire training history.
Used as a non-personalised popularity-based benchmark.

Inputs:
  interactions_list : list of dictionaries, each with keys {"users", "items"},
                      where items is a Tensor of interacted item IDs (positive)
  k                 : number of top items to recommend

Returns:
  predictions : dict of user_id → List[item_id] (same list for all users)
"""

def most_pop_predictor(interactions_list: List[Dict], k: int, user_ids: List[int]) -> Dict[int, List[int]]:
    """Return top-K most popular items for each user."""
    item_counter = Counter()
    for daily_batch in interactions_list:
        items = daily_batch["items"].tolist()
        item_counter.update(items)

    top_k_items = [item for item, _ in item_counter.most_common(k)]
    return {u: top_k_items for u in user_ids}

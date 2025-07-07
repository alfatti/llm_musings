from collections import Counter
from typing import Dict, List, Set
import torch

"""
RecentPop baseline — Section 4.3
===============================
Selects top-K most interacted items within a **fixed-size trailing window**
of the most recent interaction snapshots.

Inputs:
  interactions_list : list of daily interaction dicts with {"users", "items"}
  k                 : number of top items to recommend
  user_ids          : list of user IDs to generate predictions for
  window_size       : number of most recent days to use (e.g. 2 or 5)

Returns:
  predictions : dict of user_id → List[item_id] (same list for all users)
"""

def recent_pop_predictor(
    interactions_list: List[Dict],
    k: int,
    user_ids: List[int],
    window_size: int = 2,
) -> Dict[int, List[int]]:
    """Return top-K recently popular items over last `window_size` snapshots."""
    item_counter = Counter()
    for daily_batch in interactions_list[-window_size:]:
        items = daily_batch["items"].tolist()
        item_counter.update(items)

    top_k_items = [item for item, _ in item_counter.most_common(k)]
    return {u: top_k_items for u in user_ids}

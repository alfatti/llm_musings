import numpy as np
from typing import Dict, List, Set, Sequence, Tuple

"""
================================================================================
Evaluation Utilities  —  Paper § 4.3
================================================================================
This module gathers **all offline‑ranking metrics** required by § 4.3 of
*Rolling Forward: Enhancing LightGCN with Causal Graph Convolution for Credit
Bond Recommendation*.

The paper reports:
  • **Precision@K**
  • **Recall@K**
  • **NDCG@K** (Normalised Discounted Cumulative Gain)
  • **MRR@K** (Mean Reciprocal Rank)

Below, each metric is implemented as a *pure function*.  They accept per‑user
ranked recommendation lists and the corresponding ground‑truth item sets.

Typical usage (after you have generated top‑K predictions for every user):

```python
from lightgcn_eval import evaluate_all
metrics = evaluate_all(preds, ground_truth, k_values=[25, 50])
print(metrics["Recall@25"], metrics["NDCG@50"], ...)
```

*Glossary*
----------
`preds`         Mapping `user_id → List[item_id]` sorted by descending score.
`ground_truth`  Mapping `user_id → Set[item_id]` of relevant (positive) items.
`k_values`      List of cutoff values *K* (e.g. `[25, 50]`).

All functions are **NumPy‑based**  (no additional ML deps) and are written to be
vectorisable if you wish to extend them; but clarity is prioritised here.
================================================================================
"""

# -----------------------------------------------------------------------------
# Low‑level single‑user helpers
# -----------------------------------------------------------------------------

def _precision_at_k(ranked: Sequence[int], relevant: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def _recall_at_k(ranked: Sequence[int], relevant: Set[int], k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def _dcg_at_k(ranked: Sequence[int], relevant: Set[int], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(idx + 1)
    return dcg


def _ndcg_at_k(ranked: Sequence[int], relevant: Set[int], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return _dcg_at_k(ranked, relevant, k) / idcg


def _reciprocal_rank(ranked: Sequence[int], relevant: Set[int], k: int) -> float:
    for idx, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


# -----------------------------------------------------------------------------
# Batch‑level public API
# -----------------------------------------------------------------------------

def evaluate_all(
    preds: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k_values: Sequence[int] = (25, 50),
    aggregate: bool = True,
) -> Dict[str, float]:
    """Compute all § 4.3 metrics for each K in `k_values`.

    Parameters
    ----------
    preds         : user → ranked list of recommendations.
    ground_truth  : user → set of relevant items.
    k_values      : list/tuple of cutoff thresholds.
    aggregate     : if True, returns global mean metrics; otherwise returns
                    per‑user dict {user: {metric: value}}.
    """

    users = set(preds.keys()) & set(ground_truth.keys())
    per_user: Dict[int, Dict[str, float]] = {}

    for u in users:
        ranked = preds[u]
        relevant = ground_truth[u]
        stats = {}
        for k in k_values:
            stats[f"Precision@{k}"] = _precision_at_k(ranked, relevant, k)
            stats[f"Recall@{k}"] = _recall_at_k(ranked, relevant, k)
            stats[f"NDCG@{k}"] = _ndcg_at_k(ranked, relevant, k)
            stats[f"MRR@{k}"] = _reciprocal_rank(ranked, relevant, k)
        per_user[u] = stats

    if not aggregate:
        return per_user  # type: ignore[return-value]

    # ----------------- Aggregate by arithmetic mean --------------------
    agg: Dict[str, float] = {metric: 0.0 for metric in next(iter(per_user.values())).keys()}
    for stats in per_user.values():
        for m, v in stats.items():
            agg[m] += v
    num_users = len(per_user)
    for m in agg:
        agg[m] /= num_users
    return agg


# -----------------------------------------------------------------------------
# Convenience: print nicely
# -----------------------------------------------------------------------------

def format_metrics(metrics: Dict[str, float]) -> str:
    """Return a pretty, aligned string for console logging."""
    return "  ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items()))


# -----------------------------------------------------------------------------
# Quick self‑test (sanity) ------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    preds = {}
    ground = {}
    for user in range(5):
        relevant = set(rng.choice(100, size=10, replace=False))
        ranked = rng.permutation(100).tolist()
        preds[user] = ranked
        ground[user] = relevant
    out = evaluate_all(preds, ground, k_values=[5, 10])
    print("Synthetic‑test:", format_metrics(out))

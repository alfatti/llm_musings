import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import LGConv
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

"""
================================================================================
LightGCN‑W  —  PyTorch / PyG Reference Implementation
================================================================================
*Paper*: **Rolling Forward: Enhancing LightGCN with Causal Graph Convolution for
Credit Bond Recommendation** (Ghiye et al., ICAIF 2024, arXiv:2503.14213)

This file is **NOT** a drop‑in copy of the original LightGCN code.  Instead it
starts from the official LightGCN implementation (He et al., SIGIR 2020) and
**adds the minimal changes required** to reproduce the *temporal, causal* model
described in the paper.  Every section that deviates from vanilla LightGCN is
marked with ▶▶  **CHANGED**  ◀◀ comment blocks.

--------------------------------------------------------------------------------
Key Additions / Differences vs Original LightGCN
--------------------------------------------------------------------------------
1. **Sliding‑Window Temporal Encoding**
   • Embeddings are computed from a *causal* window of historical graphs
     `G_{t−w+1} … G_t` (no future leakage).
   • Implemented in `encode_window()`.

2. **Union‑of‑Edges Aggregation**
   • All edge sets inside the window are concatenated before message passing.
     (Paper uses union; concatenation is equivalent when duplicate edges are
     allowed.)

3. **Day‑by‑Day Training Loop Stub**
   • `TemporalInteractionDataset` and `train_epoch()` iterate snapshot‑wise.
   • Negative sampling uses Dynamic Negative Sampling (DNS) as in the paper.

4. **Window‑Size Hyper‑parameter**
   • Exposed via constructor; default `w = 2` as recommended by authors.

5. **Code‑Level Modernisations**
   • PyG `LGConv` supersedes custom sparse ops from original repo.
   • Type annotations & extensive docstrings for clarity.

--------------------------------------------------------------------------------
Acronyms
--------------------------------------------------------------------------------
BPR  —  Bayesian Personalised Ranking loss
DNS  —  Dynamic Negative Sampling
LGConv —  LightGCN convolution layer in PyTorch Geometric
--------------------------------------------------------------------------------
"""

# ==============================================================================
# 1️⃣  Model Definition
# ==============================================================================
class LightGCN_W(nn.Module):
    """LightGCN‑W: Causal, windowed LightGCN for temporal bond recommendation.

    Parameters
    ----------
    num_users      : Total number of unique user nodes.
    num_items      : Total number of unique item (bond) nodes.
    embedding_dim  : Size of latent vectors (*d* in the paper).
    num_layers     : Number of LightGCN propagation layers (*K*).
    window_size    : Temporal window length (*w*).  **CHANGED**
    """

    # ------------------------------------------------------------------
    # ▶▶  **CHANGED**  ◀◀
    # Added `window_size` so we can slide over historical graphs.
    # ------------------------------------------------------------------
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        window_size: int = 2,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.window_size = window_size  # **new** hyper‑parameter

        # Static ID embeddings — same as LightGCN
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # LightGCN message‑passing layers (no transformations / activations)
        self.convs = nn.ModuleList(LGConv() for _ in range(num_layers))
        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self):  # identical to original
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    # ------------------------------------------------------------------
    # 2️⃣  Embedding Computation with Causal Window
    # ------------------------------------------------------------------
    def encode_window(self, window_graphs: List[Data]) -> Tuple[Tensor, Tensor]:
        """Compute user / item embeddings for the *current* day (*t*).

        This replaces the single‑graph `encode()` in vanilla LightGCN.
        Only the graphs within the causal window (past→present) are used.
        """
        # ▶▶  **CHANGED**  ◀◀  Multiple snapshots ⇒ merge edge indices
        if len(window_graphs) == 0:
            raise ValueError("Window is empty. Provide ≥1 snapshots.")

        # (a) Collect edges from all graphs in window (union/concat)
        #     Users are indexed [0, num_users);  items: [num_users, …)
        edge_index = torch.cat([g.edge_index for g in window_graphs], dim=1)

        # (b) Layer‑0 initial features = ID embeddings (LightGCN assumption)
        x0 = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )  # shape: (N_nodes, d)
        all_layers = [x0]
        x = x0

        # (c) Propagate through K LightGCN layers
        for conv in self.convs:
            x = conv(x, edge_index)  # simple mean aggregation, no bias, no act
            all_layers.append(x)

        # (d) Final embedding = mean of embeddings from all (K+1) layers
        out = torch.stack(all_layers, dim=0).mean(0)
        return out[: self.num_users], out[self.num_users :]

    # ------------------------------------------------------------------
    # 3️⃣  Scoring  &  Loss
    # ------------------------------------------------------------------
    @staticmethod
    def dot_score(u_vec: Tensor, i_vec: Tensor) -> Tensor:
        """Dot‑product recommendation score (eq. 3 in original LightGCN)."""
        return (u_vec * i_vec).sum(dim=-1)

    def score(self, users: Tensor, items: Tensor, u_emb: Tensor, i_emb: Tensor) -> Tensor:
        """Convenience wrapper around `dot_score`."""
        return self.dot_score(u_emb[users], i_emb[items])

    def bpr_loss(
        self,
        u_emb: Tensor,
        i_emb: Tensor,
        users: Tensor,
        pos_items: Tensor,
        neg_items: Tensor,
        reg_lambda: float = 1e-4,
    ) -> Tensor:
        """Bayesian Personalised Ranking loss (same as original)."""
        pos_score = self.score(users, pos_items, u_emb, i_emb)
        neg_score = self.score(users, neg_items, u_emb, i_emb)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

        # L2 regularisation on static embeddings — unchanged
        reg = (
            self.user_embedding.weight.norm(2).pow(2)
            + self.item_embedding.weight.norm(2).pow(2)
        ) / (self.num_users + self.num_items)
        return loss + reg_lambda * reg


# ==============================================================================
# 2️⃣  Temporal Dataset Helper  (*new*)
# ==============================================================================
class TemporalInteractionDataset(Dataset):
    """Pairs daily PyG graphs with positive (u, i) interactions for that day."""

    # ▶▶  **CHANGED**  ◀◀  Original LightGCN had a single static interaction set.
    def __init__(self, graphs: List[Data], interactions: List[Dict]):
        assert len(graphs) == len(interactions)
        self.graphs = graphs
        self.interactions = interactions  # list[ {"users": Tensor, "items": Tensor} ]

    def __len__(self) -> int:
        return len(self.graphs)  # number of snapshots / days

    def __getitem__(self, idx):
        return self.graphs[idx], self.interactions[idx]


# ==============================================================================
# 3️⃣  Training Utilities
# ==============================================================================

def dns_negative_sampling(users: Tensor, num_items: int, k: int = 10) -> Tensor:
    """Dynamic Negative Sampling: draw `k` random negatives per user.
    Later, the caller re‑scores and picks the hardest.  Simpler + faster than
    full supervised negatives, yet empirically strong for implicit feedback.
    """
    return torch.randint(0, num_items, (users.size(0), k), device=users.device)


def train_epoch(
    model: LightGCN_W,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
):
    """One epoch over **all** snapshots (deterministic chronological order).

    Vanilla LightGCN uses uniform random sampling of (u, i, j) triples; here we
    respect temporal ordering so that snapshot *t* does not see the future.
    """
    model.train()
    total_loss = 0.0

    for t, (graph_t, interactions_t) in enumerate(data_loader):
        # ------- 1. Build causal window  --------------------------------
        start_day = max(0, t - model.window_size + 1)
        window_graphs = [data_loader.dataset.graphs[d].to(device) for d in range(start_day, t + 1)]

        # ------- 2. Forward pass ----------------------------------------
        u_emb, i_emb = model.encode_window(window_graphs)

        users = interactions_t["users"].to(device)
        pos_items = interactions_t["items"].to(device)

        # ------- 3. Dynamic Negative Sampling (DNS) ---------------------
        neg_cands = dns_negative_sampling(users, model.num_items, k=10)
        with torch.no_grad():
            scores = model.score(
                users.unsqueeze(1).expand_as(neg_cands), neg_cands, u_emb, i_emb
            )
        hardest_idx = scores.argmax(dim=-1)
        neg_items = neg_cands[torch.arange(len(hardest_idx)), hardest_idx]

        # ------- 4. Back‑prop ------------------------------------------
        optimizer.zero_grad()
        loss = model.bpr_loss(u_emb, i_emb, users, pos_items, neg_items)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# ==============================================================================
# 4️⃣  Example Usage (Pseudo‑code)
# ==============================================================================
if __name__ == "__main__":
    # ────────────────────────────────────────────────────────────────────
    # Replace the placeholders below with real data loading logic.
    # The BNP dataset used in the paper is not public, so we only show
    # structure / API expectations here.
    # ────────────────────────────────────────────────────────────────────

    NUM_USERS = 10_000   # <-- replace with actual count
    NUM_ITEMS = 50_000   # <-- replace with actual count

    # `daily_graphs`  : list[ PyG.Data ]  — edges for each day (or week etc.)
    # `daily_pos`     : list[ {"users": Tensor, "items": Tensor} ]  positives
    # daily_graphs, daily_pos = load_your_snapshots(...)

    # dataset  = TemporalInteractionDataset(daily_graphs, daily_pos)
    # loader   = DataLoader(dataset, batch_size=1, shuffle=False)  # chronological

    model = LightGCN_W(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=64,
        num_layers=3,
        window_size=2,         # **empirically best per paper**
    ).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # for epoch in range(40):
    #     avg_loss = train_epoch(model, loader, torch.device("cuda"), optimizer)
    #     print(f"Epoch {epoch:02d} — Loss: {avg_loss:.4f}")

    print("Template ready ✔ — plug in your data & start training!")

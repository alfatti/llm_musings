from lightgcn_w import LightGCN_W

def make_ablation_model(variant: str, num_users: int, num_items: int) -> LightGCN_W:
    """
    Construct a LightGCN_W model with ablation toggles based on variant name.

    Supported variants:
    - "full":          Full LightGCN-W model
    - "no_window":     Ablation A: No sliding window (just current snapshot)
    - "no_causal":     Ablation B: Uses future context (not strictly causal)
    - "no_temporal":   Ablation D: Disables layer aggregation
    - "no_dns":        Ablation E: Uses static random negative sampling
    """
    # Default: full model (causal, with window, with aggregation, with DNS)
    window_size = 2 if variant != "no_window" else 1

    model = LightGCN_W(num_users, num_items, window_size=window_size)
    model.ablation_mode = variant  # used in encode_window and training loop
    return model

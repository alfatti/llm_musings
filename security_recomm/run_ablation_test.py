from lightgcn_w import LightGCN_W, train_epoch
from lightgcn_eval import evaluate_all, format_metrics
from ablations import make_ablation_model
from torch.utils.data import DataLoader
import torch

# Load data
graphs, interactions = load_snapshots()  # user-defined
dataset = TemporalInteractionDataset(graphs, interactions)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
user_ids = list(range(NUM_USERS))

# Config
ablation_variants = ["full", "no_window", "no_causal", "no_temporal", "no_dns"]

results = {}

for variant in ablation_variants:
    print(f"Training variant: {variant}")
    model = make_ablation_model(variant, NUM_USERS, NUM_ITEMS).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(NUM_EPOCHS):
        loss = train_epoch(model, loader, torch.device("cuda"), optimizer, variant=variant)
    
    # Evaluation
    recs = model.recommend_all_topk(user_ids, k=50)  # add this method if not done
    truth = get_test_positives()  # {user_id: [item_ids]}
    metrics = evaluate_all(recs, truth, k_vals=[20, 50])
    results[variant] = metrics
    print(format_metrics(metrics))

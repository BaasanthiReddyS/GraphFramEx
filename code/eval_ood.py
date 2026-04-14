
import os, sys, json
sys.path.insert(0, "code")
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from gnn.model import GCN
from gnn.eval import gnn_accuracy
from utils.io_utils import load_ckpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Cora
dataset = Planetoid(root="/kaggle/working/data", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0].to(device)
data.edge_weight = torch.ones(data.edge_index.shape[1]).to(device)

results = {}

for seed in [42, 43, 44]:
    model = GCN(num_node_features=1433, hidden_dim=16, num_classes=7,
                dropout=0.5, num_layers=2, device=device)
    ckpt_path = f"/kaggle/working/models/cora/cora_nc_h16_o16_gcn_2_epch200_lr0.01_wd0.0005_drop0.5_{seed}.pth.tar"
    ckpt = load_ckpt(ckpt_path, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    # Clean accuracy
    with torch.no_grad():
        out = model(data.x, data.edge_index, edge_weight=data.edge_weight)
    clean_acc = (out.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    ood_results = {}
    for rate in [0.10, 0.15, 0.20]:
        # perturb edges
        num_edges = data.edge_index.shape[1]
        num_drop = int(num_edges * rate)
        keep = torch.ones(num_edges, dtype=torch.bool)
        keep[torch.randperm(num_edges)[:num_drop]] = False

        ood_edge_index = data.edge_index[:, keep].to(device)
        ood_edge_weight = data.edge_weight[keep].to(device)

        with torch.no_grad():
            out_ood = model(data.x, ood_edge_index, edge_weight=ood_edge_weight)
        ood_acc = (out_ood.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        drop = clean_acc - ood_acc
        ood_results[f"perturb_{int(rate*100)}pct"] = {
            "clean_acc": round(clean_acc, 4),
            "ood_acc": round(ood_acc, 4),
            "acc_drop": round(drop, 4)
        }
        print(f"Seed {seed} | Perturb {int(rate*100)}% | Clean: {clean_acc:.4f} | OOD: {ood_acc:.4f} | Drop: {drop:.4f}")

    results[f"seed_{seed}"] = ood_results

# Save results
with open("/kaggle/working/ood_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to /kaggle/working/ood_results.json")

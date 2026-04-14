
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

    with torch.no_grad():
        out = model(data.x, data.edge_index, edge_weight=data.edge_weight)
    clean_acc = (out.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    ood_results = {}
    for rate in [0.10, 0.15, 0.20]:
        # --- Edge perturbation (drop edges) ---
        num_edges = data.edge_index.shape[1]
        keep = torch.ones(num_edges, dtype=torch.bool)
        keep[torch.randperm(num_edges)[:int(num_edges * rate)]] = False
        with torch.no_grad():
            out_e = model(data.x, data.edge_index[:, keep], edge_weight=data.edge_weight[keep])
        acc_e = (out_e.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        # --- Node feature perturbation (add Gaussian noise) ---
        x_noisy = data.x + rate * torch.randn_like(data.x)
        with torch.no_grad():
            out_n = model(x_noisy, data.edge_index, edge_weight=data.edge_weight)
        acc_n = (out_n.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        key = f"perturb_{int(rate*100)}pct"
        ood_results[key] = {
            "clean_acc": round(clean_acc, 4),
            "edge_ood_acc": round(acc_e, 4),
            "edge_acc_drop": round(clean_acc - acc_e, 4),
            "node_ood_acc": round(acc_n, 4),
            "node_acc_drop": round(clean_acc - acc_n, 4),
        }
        print(f"Seed {seed} | {int(rate*100)}% | Edge drop: {clean_acc - acc_e:.4f} | Node noise: {clean_acc - acc_n:.4f}")

    results[f"seed_{seed}"] = ood_results

with open("/kaggle/working/ood_results_full.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to /kaggle/working/ood_results_full.json")

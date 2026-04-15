import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from math import sqrt

EPS = 1e-15

class GNNExplainer(torch.nn.Module):
    coeffs = {"edge_size": 0.005, "edge_reduction": "sum",
              "node_feat_size": 1.0, "node_feat_reduction": "mean",
              "edge_ent": 1.0, "node_feat_ent": 0.1}

    def __init__(self, model, epochs=100, lr=0.01, num_hops=None,
                 allow_edge_mask=True, allow_node_mask=True, device=None, **kwargs):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.allow_edge_mask = allow_edge_mask
        self.allow_node_mask = allow_node_mask
        self.device = device or torch.device("cpu")
        self.coeffs.update({k: v for k, v in kwargs.items() if k in self.coeffs})

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__
        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        x_sub = x[subset]
        kwargs_sub = {}
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                kwargs_sub[key] = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                kwargs_sub[key] = item[edge_mask]
            else:
                kwargs_sub[key] = item
        return x_sub, edge_index_sub, mapping, edge_mask, subset, kwargs_sub

    def __init_masks__(self, x, edge_index):
        num_nodes, num_features = x.size()
        num_edges = edge_index.size(1)
        self.node_feat_mask = torch.nn.Parameter(torch.randn(num_features, device=self.device) * 0.1)
        std_edge = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * num_nodes))
        self.edge_mask = torch.nn.Parameter(torch.randn(num_edges, device=self.device) * std_edge)

    def __loss__(self, node_idx, log_logits, pred_label, target=None):
        t = target if target is not None else pred_label[node_idx]
        loss = -log_logits[node_idx, t]
        if self.allow_edge_mask:
            m = self.edge_mask.sigmoid()
            loss += self.coeffs["edge_size"] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss += self.coeffs["edge_ent"] * ent.mean()
        if self.allow_node_mask:
            m = self.node_feat_mask.sigmoid()
            loss += self.coeffs["node_feat_size"] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss += self.coeffs["node_feat_ent"] * ent.mean()
        return loss

    def explain_node(self, node_idx, x, edge_index, target=None, **kwargs):
        self.model.eval()
        num_edges = edge_index.size(1)
        with torch.no_grad():
            out = self.model(x, edge_index, **kwargs)
            log_logits = F.log_softmax(out, dim=-1)
            pred_label = log_logits.argmax(dim=-1)
        x_sub, edge_index_sub, mapping, edge_mask_bool, subset, kwargs_sub = \
            self.__subgraph__(node_idx, x, edge_index, **kwargs)
        self.__init_masks__(x_sub, edge_index_sub)
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            h = x_sub * self.node_feat_mask.sigmoid()
            out = self.model(h, edge_index_sub, **kwargs_sub)
            log_logits = F.log_softmax(out, dim=-1)
            loss = self.__loss__(mapping.item(), log_logits, pred_label[subset], target=target)
            loss.backward()
            optimizer.step()
        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        sub_edge_mask = self.edge_mask.detach().sigmoid()
        full_edge_mask = torch.zeros(num_edges, device=self.device)
        full_edge_mask[edge_mask_bool] = sub_edge_mask
        return node_feat_mask, full_edge_mask

class TargetedGNNExplainer(GNNExplainer):
    def explain_node_with_target(self, node_idx, x, edge_index, target=None, **kwargs):
        return self.explain_node(node_idx, x, edge_index, target=target, **kwargs)

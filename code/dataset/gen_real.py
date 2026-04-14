
from utils.gen_utils import (
    from_adj_to_edge_index,
    from_edge_index_to_adj,
    from_edge_index_to_sparse_adj,
    from_sparse_adj_to_edge_index,
    init_weights,
)

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def load_data_real(data_filename=None):
    """Load Cora dataset using PyTorch Geometric - skip double preprocessing."""
    dataset = Planetoid(root='/kaggle/working/data', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]

    # just add edge_weight, skip re-normalizing since PyG already did it
    data.edge_weight = torch.ones(data.edge_index.shape[1])
    return data


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

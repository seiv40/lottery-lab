"""
Module 10: GNN Embeddings
Graph neural network embeddings of kNN graph structure.

Simple 2-layer GCN trained with unsupervised loss to minimize
distance between connected nodes in embedding space.

"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from config import (
    LotteryName,
    RANDOM_STATE_DEFAULT,
    get_module10_output_dir,
)
from data import load_module9_graphs


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return x


def graph_to_pyg_data(G) -> Data:
    import torch

    # Node ordering is important, so create a deterministic mapping
    nodes = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    edges = []
    for u, v in G.edges():
        edges.append((node_idx[u], node_idx[v]))
        edges.append((node_idx[v], node_idx[u]))
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Use simple node features: degree as 1D feature
    degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    x = torch.tensor(degrees[:, None], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index)
    return data, nodes


def train_gcn_embeddings(
    G,
    hidden_dim: int = 16,
    out_dim: int = 8,
    epochs: int = 100,
    lr: float = 1e-2,
    device: str = "cpu",
) -> np.ndarray:
    data, nodes = graph_to_pyg_data(G)
    data = data.to(device)

    model = SimpleGCN(in_dim=data.num_node_features, hidden_dim=hidden_dim, out_dim=out_dim)
    model = model.to(device)

    # Simple unsupervised objective: minimize difference between
    # embeddings of adjacent nodes (L2 loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        # For each edge, penalize distance between embeddings
        row, col = data.edge_index
        if row.numel() == 0:
            break
        loss = ((z[row] - z[col]) ** 2).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index).cpu().numpy()

    return z


def run_gnn_embeddings(
    lottery: LotteryName,
    hidden_dim: int = 16,
    out_dim: int = 8,
    epochs: int = 100,
) -> None:
    graphs = load_module9_graphs(lottery)
    knn_graph = graphs["knn_graph"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = train_gcn_embeddings(
        knn_graph,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        epochs=epochs,
        device=device,
    )

    output_dir = get_module10_output_dir(lottery)
    out_npy = os.path.join(
        output_dir,
        f"{lottery}_gnn_embeddings.npy",
    )
    np.save(out_npy, embeddings)

    # Also save as CSV for quick inspection
    out_csv = os.path.join(
        output_dir,
        f"{lottery}_gnn_embeddings.csv",
    )
    pd.DataFrame(embeddings).to_csv(out_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 10 - GNN embeddings (simple GCN on kNN graph)"
    )
    parser.add_argument(
        "--lottery",
        type=str,
        choices=["powerball", "megamillions"],
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gnn_embeddings(
        lottery=args.lottery,  # type: ignore[arg-type]
        epochs=args.epochs,
    )

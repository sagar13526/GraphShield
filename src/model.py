"""A small GraphSAGE-style model implemented with plain PyTorch."""
import torch
from torch import nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.25):
        super().__init__()
        # Each layer combines the node's own features with its neighbors' mean.
        self.layer1 = nn.Linear(input_dim * 2, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, adjacency):
        neighbor_mean = adjacency @ x
        h = torch.cat([x, neighbor_mean], dim=1)
        h = F.relu(self.layer1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        neighbor_h = adjacency @ h
        h = torch.cat([h, neighbor_h], dim=1)
        h = F.relu(self.layer2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h).squeeze(1)

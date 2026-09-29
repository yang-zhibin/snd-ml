"""Event classifier built around the maintained PyG GravNetConv operator."""

from __future__ import annotations

import torch
from torch import nn


class GravNetBlock(nn.Module):
    def __init__(self, channels: int, space_dimensions: int,
                 propagate_dimensions: int, k: int, dropout: float = 0.0):
        super().__init__()
        try:
            from torch_geometric.nn import GravNetConv
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "PyGGravNetClassifier requires torch-geometric and a working "
                "torch-cluster/pyg-lib kNN backend"
            ) from exc
        self.conv = GravNetConv(
            channels, channels, space_dimensions, propagate_dimensions, k
        )
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        update = self.conv(x, batch=batch)
        return x + self.dropout(self.activation(self.norm(update)))


class PyGGravNetClassifier(nn.Module):
    """Three-block fixed-k GravNet baseline with mean/max event pooling."""

    def __init__(self, in_channels: int = 5, hidden_channels: int = 64,
                 num_classes: int = 7, blocks: int = 3,
                 space_dimensions: int = 4, propagate_dimensions: int = 64,
                 k: int = 16, dropout: float = 0.1):
        super().__init__()
        if int(blocks) < 1 or int(k) < 1:
            raise ValueError("GravNet blocks and k must be positive")
        self.num_classes = int(num_classes)
        self.input_mlp = nn.Sequential(
            nn.Linear(int(in_channels), int(hidden_channels)),
            nn.LayerNorm(int(hidden_channels)),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            GravNetBlock(
                int(hidden_channels), int(space_dimensions),
                int(propagate_dimensions), int(k), float(dropout),
            )
            for _ in range(int(blocks))
        ])
        self.head = nn.Sequential(
            nn.Linear(2 * int(hidden_channels), int(hidden_channels)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_channels), self.num_classes),
        )

    def forward(self, x, batch):
        if x.ndim != 2:
            raise ValueError("GravNet x must have shape [nodes,features]")
        if batch.ndim != 1 or len(batch) != len(x):
            raise ValueError("GravNet batch must identify every node")
        if not len(x):
            raise ValueError("GravNet cannot classify an empty node batch")
        from torch_geometric.nn import global_max_pool, global_mean_pool

        hidden = self.input_mlp(x)
        for block in self.blocks:
            hidden = block(hidden, batch)
        pooled = torch.cat([
            global_mean_pool(hidden, batch),
            global_max_pool(hidden, batch),
        ], dim=1)
        return self.head(pooled)

"""Memory-bounded, Cluster-GCN-inspired GravNet event classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from train_3d.models.pyg_gravnet_classifier import GravNetBlock


class ClusteredPyGGravNetClassifier(nn.Module):
    """Encode bounded spatial clusters and aggregate exactly once per event."""

    def __init__(self, in_channels: int = 5, hidden_channels: int = 64,
                 num_classes: int = 7, blocks: int = 3,
                 space_dimensions: int = 4, propagate_dimensions: int = 64,
                 k: int = 16, dropout: float = 0.1,
                 max_cluster_nodes: int = 8192,
                 max_nodes_per_forward: int = 8192,
                 activation_checkpointing: bool = True):
        super().__init__()
        if int(blocks) < 1 or int(k) < 1:
            raise ValueError("GravNet blocks and k must be positive")
        if int(max_cluster_nodes) < 1:
            raise ValueError("max_cluster_nodes must be positive")
        if int(max_nodes_per_forward) < int(max_cluster_nodes):
            raise ValueError("max_nodes_per_forward must cover at least one cluster")
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.max_cluster_nodes = int(max_cluster_nodes)
        self.max_nodes_per_forward = int(max_nodes_per_forward)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.input_mlp = nn.Sequential(
            nn.Linear(int(in_channels), self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            GravNetBlock(
                self.hidden_channels, int(space_dimensions),
                int(propagate_dimensions), int(k), float(dropout),
            )
            for _ in range(int(blocks))
        ])
        self.head = nn.Sequential(
            nn.Linear(2 * self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_channels, self.num_classes),
        )

    def _encode(self, x, cluster_batch):
        hidden = self.input_mlp(x)
        for block in self.blocks:
            hidden = block(hidden, cluster_batch)
        return hidden

    @staticmethod
    def _validate(x, cluster_ptr, cluster_event, num_events: int,
                  max_cluster_nodes: int):
        if x.ndim != 2 or not len(x):
            raise ValueError("Clustered GravNet x must be non-empty [nodes,features]")
        if cluster_ptr.ndim != 1 or len(cluster_ptr) < 2:
            raise ValueError("cluster_ptr must identify at least one cluster")
        if cluster_event.ndim != 1 or len(cluster_event) != len(cluster_ptr) - 1:
            raise ValueError("cluster_event must identify every cluster")
        pointers = cluster_ptr.detach().cpu().tolist()
        if pointers[0] != 0 or pointers[-1] != len(x):
            raise ValueError("cluster_ptr must cover x exactly")
        lengths = [stop - start for start, stop in zip(pointers[:-1], pointers[1:])]
        if any(length <= 0 for length in lengths):
            raise ValueError("cluster_ptr must be strictly increasing")
        if max(lengths) > int(max_cluster_nodes):
            raise ValueError("A cluster exceeds max_cluster_nodes")
        if int(num_events) < 1:
            raise ValueError("num_events must be positive")
        events = cluster_event.detach().cpu()
        if int(events.min()) < 0 or int(events.max()) >= int(num_events):
            raise ValueError("cluster_event contains an invalid event index")
        counts = torch.bincount(events, minlength=int(num_events))
        if torch.any(counts == 0):
            raise ValueError("Every event must contain at least one cluster")
        return pointers, lengths

    def forward(self, x, cluster_ptr, cluster_event, num_events: int):
        pointers, lengths = self._validate(
            x, cluster_ptr, cluster_event, int(num_events), self.max_cluster_nodes
        )
        from torch_geometric.nn import global_add_pool, global_max_pool

        cluster_sums = []
        cluster_maxima = []
        first_cluster = 0
        while first_cluster < len(lengths):
            last_cluster = first_cluster
            nodes = 0
            while last_cluster < len(lengths):
                proposed = nodes + lengths[last_cluster]
                if last_cluster > first_cluster and proposed > self.max_nodes_per_forward:
                    break
                if proposed > self.max_nodes_per_forward:
                    raise ValueError("A cluster exceeds max_nodes_per_forward")
                nodes = proposed
                last_cluster += 1
            start, stop = pointers[first_cluster], pointers[last_cluster]
            local_lengths = torch.tensor(
                lengths[first_cluster:last_cluster], device=x.device, dtype=torch.long
            )
            local_batch = torch.repeat_interleave(
                torch.arange(len(local_lengths), device=x.device), local_lengths
            )
            local_x = x[start:stop]
            if self.activation_checkpointing and self.training:
                hidden = checkpoint(
                    self._encode, local_x, local_batch, use_reentrant=False
                )
            else:
                hidden = self._encode(local_x, local_batch)
            cluster_sums.append(
                global_add_pool(hidden, local_batch, size=len(local_lengths))
            )
            cluster_maxima.append(
                global_max_pool(hidden, local_batch, size=len(local_lengths))
            )
            first_cluster = last_cluster

        sums = torch.cat(cluster_sums, dim=0)
        maxima = torch.cat(cluster_maxima, dim=0)
        event_index = cluster_event.to(device=x.device, dtype=torch.long)
        length_tensor = torch.tensor(lengths, device=x.device, dtype=x.dtype)
        event_sums = global_add_pool(sums, event_index, size=int(num_events))
        event_counts = global_add_pool(
            length_tensor[:, None], event_index, size=int(num_events)
        )
        event_maxima = global_max_pool(maxima, event_index, size=int(num_events))
        pooled = torch.cat([event_sums / event_counts, event_maxima], dim=1)
        return self.head(pooled)

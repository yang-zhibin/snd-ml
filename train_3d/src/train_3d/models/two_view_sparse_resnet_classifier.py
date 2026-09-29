"""Two-branch 2D sparse ResNet over native SciFi detector coordinates."""

from __future__ import annotations

import torch
from torch import nn

from train_3d.models.sparse_resnet_classifier import (
    SafeMinkowskiBatchNorm,
    SparseResidualBlock,
    _minkowski_engine,
)


class SparseViewEncoder(nn.Module):
    """Encode one orientation without downsampling the five-station axis."""

    def __init__(self, in_channels: int, channels):
        super().__init__()
        ME = _minkowski_engine()
        channels = tuple(int(value) for value in channels)
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("Two-view sparse ResNet requires four positive widths")
        self.me = ME
        self.output_channels = channels[-1]
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(
                int(in_channels), channels[0], kernel_size=3, stride=1, dimension=2
            ),
            SafeMinkowskiBatchNorm(channels[0]),
            ME.MinkowskiReLU(inplace=True),
        )
        # Keep station resolution; downsample only along the channel axis.
        self.stages = nn.Sequential(
            SparseResidualBlock(channels[0], channels[0], 1, 2),
            SparseResidualBlock(channels[0], channels[1], (1, 2), 2),
            SparseResidualBlock(channels[1], channels[2], (1, 2), 2),
            SparseResidualBlock(channels[2], channels[3], (1, 2), 2),
        )

    def forward(self, coords, features, expected_events: int):
        if coords.ndim != 2 or coords.size(1) != 3:
            raise ValueError("Two-view sparse coordinates must have shape [sites,3]")
        if features.ndim != 2 or features.size(0) != coords.size(0):
            raise ValueError("Two-view sparse features must align with coordinates")
        output = features.new_zeros((expected_events, 2 * self.output_channels))
        if coords.size(0) == 0:
            return output
        input_batch = coords[:, 0].detach().cpu().long()
        if (input_batch < 0).any() or (input_batch >= expected_events).any():
            raise ValueError("Two-view sparse batch IDs are outside the event batch")
        sparse = self.me.SparseTensor(
            features=features, coordinates=coords, device=features.device
        )
        encoded = self.stages(self.stem(sparse))
        batch_ids = encoded.C[:, 0].long().to(encoded.F.device)
        for event_index in batch_ids.unique(sorted=True).tolist():
            selected = encoded.F[batch_ids == int(event_index)]
            if selected.numel():
                output[int(event_index)] = torch.cat([
                    selected.mean(dim=0), selected.max(dim=0).values
                ])
        return output


class TwoViewSparseResNetClassifier(nn.Module):
    """Independent vertical/horizontal encoders with event-level late fusion."""

    def __init__(self, in_channels: int = 2, num_classes: int = 7,
                 channels=(32, 64, 128, 256), dropout: float = 0.1,
                 dimension: int = 2, independent_view_encoders: bool = True,
                 pooling: str = "mean_max"):
        super().__init__()
        if int(dimension) != 2:
            raise ValueError("Two-view sparse ResNet requires dimension=2")
        if not independent_view_encoders:
            raise ValueError("Version 1 requires independent view encoders")
        if pooling != "mean_max":
            raise ValueError("Version 1 requires mean_max pooling")
        self.vertical_encoder = SparseViewEncoder(in_channels, channels)
        self.horizontal_encoder = SparseViewEncoder(in_channels, channels)
        fused_channels = 4 * int(tuple(channels)[-1]) + 2
        hidden = 2 * int(tuple(channels)[-1])
        self.head = nn.Sequential(
            nn.Linear(fused_channels, hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(num_classes)),
        )

    def forward(self, vertical_coords, vertical_features, horizontal_coords,
                horizontal_features, vertical_present, horizontal_present):
        if vertical_present.ndim != 1 or horizontal_present.ndim != 1:
            raise ValueError("View-presence flags must have shape [events]")
        if len(vertical_present) != len(horizontal_present):
            raise ValueError("View-presence flags must identify the same events")
        expected_events = len(vertical_present)
        if expected_events == 0:
            raise ValueError("Two-view sparse classifier cannot process an empty batch")
        vertical = self.vertical_encoder(
            vertical_coords, vertical_features, expected_events
        )
        horizontal = self.horizontal_encoder(
            horizontal_coords, horizontal_features, expected_events
        )
        flags = torch.stack([
            vertical_present.to(vertical.device, dtype=vertical.dtype),
            horizontal_present.to(vertical.device, dtype=vertical.dtype),
        ], dim=1)
        return self.head(torch.cat([vertical, horizontal, flags], dim=1))

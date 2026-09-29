"""MinkowskiEngine sparse residual event classifier.

The basic residual-block layout is adapted from the MIT-licensed official
NVIDIA/MinkowskiEngine examples; the event head and SND input contract are
repository-specific.
"""

from __future__ import annotations

import torch
from torch import nn


def _minkowski_engine():
    try:
        import MinkowskiEngine as ME
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "SparseResNetClassifier requires a pinned MinkowskiEngine build"
        ) from exc
    return ME


class SafeMinkowskiBatchNorm(nn.Module):
    """Avoid BatchNorm's undefined training update for a one-voxel tensor."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _minkowski_engine().MinkowskiBatchNorm(channels)

    def forward(self, value):
        if self.training and value.F.size(0) < 2:
            return value
        return self.norm(value)


class SparseResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dimension: int = 3):
        super().__init__()
        ME = _minkowski_engine()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=stride,
            dimension=dimension,
        )
        self.norm1 = SafeMinkowskiBatchNorm(out_channels)
        self.activation = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, stride=1,
            dimension=dimension,
        )
        self.norm2 = SafeMinkowskiBatchNorm(out_channels)
        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels, out_channels, kernel_size=1, stride=stride,
                    dimension=dimension,
                ),
                SafeMinkowskiBatchNorm(out_channels),
            )

    def forward(self, value):
        residual = value if self.projection is None else self.projection(value)
        output = self.activation(self.norm1(self.conv1(value)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


class SparseResNetClassifier(nn.Module):
    """Four-stage sparse ResNet with robust event-wise mean pooling."""

    def __init__(self, in_channels: int = 2, num_classes: int = 7,
                 channels=(32, 64, 128, 256), dropout: float = 0.1,
                 dimension: int = 3):
        super().__init__()
        ME = _minkowski_engine()
        channels = tuple(int(value) for value in channels)
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("Sparse ResNet requires four positive stage widths")
        self.me = ME
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(
                int(in_channels), channels[0], kernel_size=3, stride=1,
                dimension=dimension,
            ),
            SafeMinkowskiBatchNorm(channels[0]),
            ME.MinkowskiReLU(inplace=True),
        )
        self.stages = nn.Sequential(
            SparseResidualBlock(channels[0], channels[0], 1, dimension),
            SparseResidualBlock(channels[0], channels[1], 2, dimension),
            SparseResidualBlock(channels[1], channels[2], 2, dimension),
            SparseResidualBlock(channels[2], channels[3], 2, dimension),
        )
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(channels[-1], int(num_classes)),
        )

    @staticmethod
    def _eventwise_mean(encoded, expected_events: int):
        """Average active features per event without creating a coordinate map.

        MinkowskiGlobalAvgPooling in the pinned MinkowskiEngine release rejects
        an already-global sparse tensor (one active coordinate per event). Its
        ``decomposed_features`` accessor reaches the same failing origin-map
        operation, so group the feature matrix directly using the batch column
        in the sparse coordinates instead.
        """
        encoded_batch_ids = encoded.C[:, 0].long().to(encoded.F.device)
        if encoded_batch_ids.numel() == 0:
            raise RuntimeError("Sparse encoder produced no active coordinates")
        if (encoded_batch_ids < 0).any() or (encoded_batch_ids >= expected_events).any():
            raise RuntimeError("Sparse encoder produced an invalid event batch ID")
        counts = torch.bincount(encoded_batch_ids, minlength=expected_events)
        if (counts == 0).any():
            missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Sparse encoder produced no coordinates for events {missing}"
            )
        sums = encoded.F.new_zeros((expected_events, encoded.F.size(1))).index_add(
            0, encoded_batch_ids, encoded.F
        )
        return sums / counts.to(encoded.F.dtype).unsqueeze(1)

    def forward(self, coords, features):
        if coords.ndim != 2 or coords.size(1) != 4:
            raise ValueError("Sparse coordinates must have shape [voxels,4]")
        if features.ndim != 2 or features.size(0) != coords.size(0):
            raise ValueError("Sparse features must align with coordinates")
        if coords.size(0) == 0:
            raise ValueError("Sparse classifier cannot process an empty batch")
        batch_ids = coords[:, 0].detach().cpu().long().unique(sorted=True)
        expected_ids = torch.arange(len(batch_ids))
        if not torch.equal(batch_ids, expected_ids):
            raise ValueError("Sparse batch IDs must be contiguous and start at zero")
        # The collator deliberately keeps integer coordinates on the CPU while
        # Lightning moves floating-point features with the model.  Explicitly
        # select the feature device so MinkowskiEngine moves both components
        # to one backend before constructing its coordinate map.
        sparse = self.me.SparseTensor(
            features=features,
            coordinates=coords,
            device=features.device,
        )
        encoded = self.stages(self.stem(sparse))
        pooled = self._eventwise_mean(encoded, expected_events=len(batch_ids))
        return self.head(pooled)

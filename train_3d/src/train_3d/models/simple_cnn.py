from torch import nn


class SparseConvRegressor(nn.Module):
    def __init__(
        self,
        in_channels,
        output_dim=1,
        base_channels=16,
        dropout=0.1,
        backend="minkowski",
        dimension=3,
    ):
        super().__init__()
        self.backend = backend
        self.output_dim = output_dim

        if backend != "minkowski":
            raise ValueError(f"Unsupported sparse backend: {backend}")

        try:
            import MinkowskiEngine as ME
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SparseConvRegressor requires MinkowskiEngine. "
                "Install MinkowskiEngine or switch to a different backend."
            ) from exc

        self.me = ME
        self.network = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, base_channels, kernel_size=3, stride=1, dimension=dimension),
            ME.MinkowskiBatchNorm(base_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(base_channels, base_channels * 2, kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(base_channels * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(base_channels * 4),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiGlobalAvgPooling(),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4, output_dim),
        )

    def forward(self, coords, features):
        sparse_tensor = self.me.SparseTensor(
            features=features,
            coordinates=coords,
        )
        pooled = self.network(sparse_tensor)
        return self.head(pooled.F)

import torch.nn as nn
from torch_geometric.graphgym.models.layer import LayerConfig, MLP
from torch_geometric.graphgym.register import register_head

def new_layer_config(
    dim_in: int,
    dim_out: int,
    num_layers: int,
    has_act: bool,
    has_bias: bool,
    cfg,
) -> LayerConfig:
    r"""Create a layer configuration for a GNN layer.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
        num_layers (int): The number of hidden layers
        has_act (bool): Whether to apply an activation function after the
            layer.
        has_bias (bool): Whether to apply a bias term in the layer.
        cfg (ConfigNode): The underlying configuration.
    """
    return LayerConfig(
        has_batchnorm=getattr(cfg, 'gnn_batchnorm', True),
        bn_eps=getattr(cfg, 'bn_eps', 1e-5),
        bn_mom=getattr(cfg, 'bn_mom', 0.1),
        mem_inplace=getattr(cfg, 'mem_inplace', False),
        dim_in=dim_in,
        dim_out=dim_out,
        edge_dim=getattr(cfg, 'dataset_edge_dim', 128),
        has_l2norm=getattr(cfg, 'gnn_l2norm', True),
        dropout=getattr(cfg, 'gnn_dropout', 0.0),
        has_act=has_act,
        final_act=True,
        act='relu_plain',# getattr(cfg, 'gnn_act', 'relu'),
        has_bias=has_bias,
        keep_edge=getattr(cfg, 'gnn_keep_edge', 0.5),
        dim_inner=getattr(cfg, 'gnn_dim_inner', 16),
        num_layers=num_layers,
    )

@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out, cfg):
        super(GNNInductiveNodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg['gnn_layers_post_mp'],
                             has_act=False, has_bias=True, cfg=cfg))

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        print('in inductive node, batch', batch)
        return pred, label
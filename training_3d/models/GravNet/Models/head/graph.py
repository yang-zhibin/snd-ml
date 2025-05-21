import torch

import torch.nn as nn
from torch_geometric.graphgym.models.layer import MLP


from torch_scatter import scatter

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_head
from .inductive_node import new_layer_config


# Pooling options (pool nodes into graph representations)
# pooling function takes in node embedding [num_nodes x emb_dim] and
# batch (indices) and outputs graph embedding [num_graphs x emb_dim].
def global_add_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def global_mean_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def global_max_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')


pooling_dict = {
    'add': global_add_pool,
    'mean': global_mean_pool,
    'max': global_max_pool
}

pooling_dict = {**register.pooling_dict, **pooling_dict}

@register_head('mygraph')
class GNNGraphHead(nn.Module):
    '''Head of GNN, graph prediction

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''
    def __init__(self, dim_in, dim_out,  cfg):
        super(GNNGraphHead, self).__init__()
        # todo: PostMP before or after global pooling
        self.layer_post_mp = MLP(new_layer_config(dim_in, dim_out, cfg['gnn_layers_post_mp'],
                             has_act=False, has_bias=True, cfg=cfg))
        self.pooling_fun = pooling_dict[cfg['model_graph_pooling']]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.graph_label

    def forward(self, batch):

        graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        # pred, label = self._apply_index(batch)
        return graph_emb # pred, label

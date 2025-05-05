import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)

import torch_geometric.graphgym.models.act

from .layer.multi_model_layer import MultiLayer, SingleLayer
from .encoder.er_edge_encoder import EREdgeEncoder
from .encoder.exp_edge_fixer import ExpanderEdgeFixer
from .encoder.linear_node_encoder import LinearNodeEncoder
from .encoder.dummy_edge_encoder import DummyEdgeEncoder
from .head.graph import GNNGraphHead

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, cfg):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg['dataset_node_encoder']:
            # Encode integer node features via nn.Embeddings
            print(register.node_encoder_dict)
            NodeEncoder = register.node_encoder_dict[
                cfg['dataset_node_encoder_name']]
            self.node_encoder = NodeEncoder(cfg['gnn_dim_inner'], dim_in)
            if cfg['dataset_node_encoder_bn']:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg['gnn_dim_inner'], -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg['gnn_dim_inner']
        if cfg['dataset_edge_encoder']:
            if  not getattr(cfg, 'gt_dim_edge', None):
                cfg['gt_dim_edge'] = cfg['gt_dim_hidden']

            if cfg['dataset_edge_encoder_name'] == 'ER':
                self.edge_encoder = EREdgeEncoder(cfg['gt_dim_edge'])
            elif cfg['dataset_edge_encoder_name'].endswith('+ER'):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg['dataset_edge_encoder_name'][:-3]]
                self.edge_encoder = EdgeEncoder(cfg['gt_dim_edge'] - cfg['posenc_ERE_dim_pe'])
                self.edge_encoder_er = EREdgeEncoder(cfg['posenc_ERE_dim_pe'], use_edge_attr=True)
            else:
                EdgeEncoder = register.edge_encoder_dict[
                    cfg['dataset_edge_encoder_name']]
                self.edge_encoder = EdgeEncoder(cfg['gt_dim_edge'])

            if cfg['dataset_edge_encoder_bn']:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg['gt_dim_edge'], -1, -1, has_act=False,
                                    has_bias=False, cfg=cfg))

        if 'Exphormer' in cfg['gt_layer_type']:
            self.exp_edge_fixer = ExpanderEdgeFixer(cfg, add_edge_index=cfg['prep_add_edge_index'], 
                                                    num_virt_node=cfg['prep_num_virt_node'])

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

# Core basic layers
# Input: batch; Output: batch
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch
    
layer_dict = {}
layer_dict['linear'] = Linear

mem_inplace = False

act_dict = {
    'relu': nn.ReLU, #(inplace=mem_inplace),
    'relu_plain': nn.ReLU(), # for compatibility with pyg graphgym...
    'selu': nn.SELU(inplace=mem_inplace),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(inplace=mem_inplace),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1, inplace=mem_inplace),
    'lrelu_025': nn.LeakyReLU(negative_slope=0.25, inplace=mem_inplace),
    'lrelu_05': nn.LeakyReLU(negative_slope=0.5, inplace=mem_inplace),
}

register.act_dict = act_dict

class GeneralLayer(nn.Module):
    '''General wrapper for layers'''
    def __init__(self,
                 name,
                 dim_in,
                 dim_out,
                 cfg,
                 has_act=True,
                 has_bn=True,
                 has_l2norm=False,
                 **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn and cfg['gnn_batchnorm']
        self.layer = layer_dict[name](dim_in,
                                      dim_out,
                                      bias=not has_bn,
                                      **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(dim_out, eps=cfg['bn_eps'], momentum=cfg['bn_mom']))
        if cfg['gnn_dropout'] > 0:
            layer_wrapper.append(
                nn.Dropout(p=cfg['gnn_dropout'], inplace=cfg['mem_inplace']))
        if has_act:
            layer_wrapper.append(act_dict[cfg['gnn_act']])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x,
                                                 p=2,
                                                 dim=1)
        return batch

class GeneralMultiLayer(nn.Module):
    '''General wrapper for stack of layers'''
    def __init__(self,
                 name,
                 num_layers,
                 dim_in,
                 dim_out,
                 cfg,
                 dim_inner=None,
                 final_act=True,
                 **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, cfg, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch

def GNNPreMP(dim_in, dim_out, cfg):
    """
    Wrapper for NN layer before GNN message passing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    return GeneralMultiLayer('linear',
                             cfg['gnn_layers_pre_mp'],
                             dim_in,
                             dim_out,
                             cfg,
                             dim_inner=dim_out,
                             final_act=True)

class MultiModel(torch.nn.Module):
    """Multiple layer types can be combined here.
    """

    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in, cfg)
        dim_in = self.encoder.dim_in

        if cfg['gnn_layers_pre_mp'] > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg['gnn_dim_inner'], cfg)
            dim_in = cfg['gnn_dim_inner']

        assert cfg['gt_dim_hidden'] == cfg['gnn_dim_inner'] == dim_in, \
            "The inner and hidden dims must match."

        try:
            model_types = cfg['gt_layer_type'].split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg['gt_layer_type']}")
        layers = []
        for _ in range(cfg['gt_layers']):
            layers.append(MultiLayer(
                dim_h=cfg['gt_dim_hidden'],
                model_types=model_types,
                num_heads=cfg['gt_n_heads'],
                pna_degrees=getattr(cfg, 'gt_pna_degrees', None),
                equivstable_pe=getattr(cfg, 'posenc_EquivStableLapPE_enable', None),
                dropout=cfg['gt_dropout'],
                attn_dropout=cfg['gt_attn_dropout'],
                layer_norm=cfg['gt_layer_norm'],
                batch_norm=cfg['gt_batch_norm'],
                # bigbird_cfg=cfg['gt_bigbird'],
                exp_edges_cfg=cfg 
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg['gnn_head']]
        self.post_mp = GNNHead(dim_in=cfg['gnn_dim_inner'], dim_out=dim_out, cfg=cfg)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

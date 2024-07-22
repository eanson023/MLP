import torch
import torch.nn as nn

from einops import rearrange, reduce, repeat
from mlp.model.blocks import GCN, WeightedPool
from mlp.model.encoder.t_enc_only import TEncoder
from mlp.model.utils import Graph

class FactorisedSTEncoder(nn.Module):
    def __init__(self, in_dim: int,
                    dim: int, 
                    max_snippet_len: int,
                    pooling: str,
                    dataset: str,
                    adaptive: str,
                    num_heads = 4, 
                    num_layers = 4,
                    gcn_layers = 5, 
                    drop_rate = 0.1,
                    shared = False, 
                    **kwargs):
        """
        S-Enc (Graph Convolutional Network & Attentive Pooling) + T-Enc (Self-Guided Parallel Attention Module)

        Args:
            in_dim (int): Model input dimensions (denoted as $d_{p}$ in our paper) 
            dim (int): Model hidden dimensions (denoted as $d$ in our paper) 
            max_snippet_len (int): max interval snippet number (denoted as $S$ in our paper) 
            pooling (str): Ways of pooling spatial features, including 'cls_token','weighted_pool','mean'
            dataset (str): _description_
            adaptive (str): Graph Convolutional Adjacency Matrix Initialization Approach
            num_heads (int, optional): The attention head of SGPA. Defaults to 4.
            num_layers (int, optional): The block numbers of SGPA. Defaults to 4.
            gcn_layers (int, optional): Defaults to 5.
            drop_rate (float, optional): Defaults to 0.1.
            shared (bool, optional): Represents whether T-Enc parameters are shared. Defaults to False.
        """
        super(FactorisedSTEncoder, self).__init__()

        self.shared = shared
        self.spatial_blocks = DividedGraphConvolutionBlock(in_dim=in_dim, 
                                                           embed_dims=dim,
                                                           gcn_layers=gcn_layers, 
                                                           num_heads=num_heads, 
                                                           drop_rate=drop_rate, 
                                                           dataset=dataset, 
                                                           pooling=pooling,
                                                           adaptive=adaptive)
        self.temporal_feat_enc = TEncoder(dim=dim,
                                             max_snippet_len=max_snippet_len,
                                             num_heads=num_heads,
                                             num_layers=num_layers,
                                             drop_rate=drop_rate,
                                             shared=shared)


    def forward(self, mfeats, qfeats, m_mask, q_mask, h_mask=None, *args):
        assert len(mfeats.shape) == 4
        # S-Enc+T-Enc = Factorized space-time encoder
        mfeats = self.spatial_blocks(mfeats, m_mask).unsqueeze(2)
        mfeats, qfeats = self.temporal_feat_enc(mfeats, qfeats, m_mask, q_mask, h_mask)
        
        return mfeats, qfeats



class DividedGraphConvolutionBlock(nn.Module):
    """Spatial Graph Convolution in Divided Space Time Attention.
        A warp for GCB+Attentive Pooling.
        
    Args:
        in_dim (int): The dimension of motion input
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        drop_rate (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        pooling (str): The way to aggregate the spatial dimensions at the end.
        adaptive (str): How to apply adjacency matrix
    """

    def __init__(self, in_dim, embed_dims, gcn_layers=0, drop_rate=0, pooling='mean', dataset="babel", adaptive="none",
                 **kwargs):
        super().__init__()
        assert pooling in ['cls_token','weighted_pool','mean','max','sum']
        assert gcn_layers>0 and gcn_layers % 2 == 1
        self.pooling = pooling
        num_stage = (gcn_layers-1)//2
        
        # Generate adjacency matrix
        graph_cfg=dict(layout=dataset, mode='spatial', num_filter=8, init_off=.04, init_std=.02)
        graph = Graph(**graph_cfg)
        graph_adj = graph.A.max(0) != 0
        if pooling == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(in_dim))
            # add cls (collects infos from all the other joints, so first and last column are set to all ones)
            graph_adj_cls = torch.ones(graph_adj.shape[0] + 1, graph_adj.shape[1] + 1).bool()
            graph_adj_cls[1:, 1:] = torch.BoolTensor(graph_adj)
            graph_adj = graph_adj_cls
        else:
            # Convert graph_adj to a PyTorch tensor
            graph_adj = torch.BoolTensor(graph_adj)

        A = graph_adj.float()

        self.gcns = GCN(in_dim, embed_dims, A, num_stage=num_stage, p_dropout=drop_rate, adaptive=adaptive, is_bn=True)
        if self.pooling == "weighted_pool":
            self.pool_layer = WeightedPool(embed_dims)


    def forward(self, x, temporal_mask, **kwargs):
        bs, t = x.shape[:2]
        if self.pooling == "cls_token":
            cls_token = repeat(self.cls_token, 'd -> b t 1 d', b=bs, t=t)
            # adding the embedding token for all sequences
            x = torch.cat((cls_token, x), dim=2)
        x = rearrange(x, 'b t p d -> (b t) p d')
        
        # Perform interactions between joint points
        x = self.gcns(x)
        
        if self.pooling == "weighted_pool":
            x = self.pool_layer(x, rearrange(temporal_mask, 'b t -> (b t) 1'))
            x = rearrange(x, '(b t) d -> b t d', b=bs)
            return x
        
        x = rearrange(x, '(b t) p d -> b t p d', b=bs)
        if self.pooling == "cls_token":
            x = x[:, :, 0, ...]
        else:
            x = reduce(x, 'b t p d -> b t d', self.pooling)
        # Uncommenting will cause nan tensor to appear in the subsequent LayerNorm layer.
        # x = mask_logits(x, temporal_mask.unsqueeze(-1))
        return x

        

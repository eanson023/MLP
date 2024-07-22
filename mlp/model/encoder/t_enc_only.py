import copy
import torch.nn as nn

from mlp.model.blocks import DualMultiHeadAttentionConvBlock
from mlp.model.utils import PositionalEncoding


class TEncoder(nn.Module):
    def __init__(self, dim, 
                    max_snippet_len, 
                    num_heads = 4, 
                    num_layers = 4, 
                    drop_rate = 0.1,
                    shared = False, 
                    **kwargs):
        """
        Self-Guided Parallel Attention Module
        """
        super(TEncoder, self).__init__()
        self.shared = shared
        self._create_sgpa_layers(dim, num_heads, num_layers, drop_rate)
        
        self.pos_embed = PositionalEncoding(dim, drop_rate, max_len=max_snippet_len, batch_first=True)
    

    def _create_sgpa_layers(self, dim, num_heads, num_layers, drop_rate):
        sgpa = DualMultiHeadAttentionConvBlock(dim, num_heads, drop_rate)
        if self.shared:
            self.sgpa_layers=nn.ModuleList([copy.deepcopy(sgpa) for i in range(num_layers)])
        else:
            self.sgpa_m_layers=nn.ModuleList([copy.deepcopy(sgpa) for i in range(num_layers)])
            self.sgpa_q_layers=nn.ModuleList([copy.deepcopy(sgpa) for i in range(num_layers)])


    def forward(self, mfeats, qfeats, m_mask, q_mask, h_mask=None, *args):
        assert mfeats.shape[2] == 1, 'SGPA Encoder does not support spatial interaction!'
        bs, seq_len = mfeats.shape[:2]
        mfeats = mfeats.view(bs, seq_len, -1)

        # w/o sometimes better
        # mfeats = self.pos_embed(mfeats)
        # qfeats = self.pos_embed(qfeats)

        if self.shared:
            sgpa_layers = self.sgpa_layers
        else:
            sgpa_layers = zip(self.sgpa_m_layers, self.sgpa_q_layers)

        for sgpa_layer in sgpa_layers:
            if self.shared:
                # Two modalities share one encoder to obtain more robust features
                sgpa_m = sgpa_layer
                mfeats_ = sgpa_m(from_tensor=mfeats, to_tensor=qfeats, from_mask=m_mask, to_mask=q_mask)
                qfeats_ = sgpa_m(from_tensor=qfeats, to_tensor=mfeats, from_mask=q_mask, to_mask=m_mask)
            else:
                sgpa_m, sgpa_q = sgpa_layer
                mfeats_ = sgpa_m(from_tensor=mfeats, to_tensor=qfeats, from_mask=m_mask,to_mask=q_mask)
                qfeats_ = sgpa_q(from_tensor=qfeats, to_tensor=mfeats, from_mask=q_mask, to_mask=m_mask)\

            mfeats, qfeats = mfeats_, qfeats_
            
        return mfeats, qfeats

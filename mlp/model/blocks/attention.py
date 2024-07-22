import math
import torch
import torch.nn as nn

from mlp.model.blocks import Conv1D
from mlp.model.utils import mask_logits


def create_attention_mask(from_mask, to_mask, broadcast_ones=False):
    batch_size, from_seq_len = from_mask.size()
    _, to_seq_len = to_mask.size()
    to_mask = to_mask.unsqueeze(1).float()

    if broadcast_ones:
        mask = torch.ones(batch_size, from_seq_len, 1).float()
    else:
        mask = from_mask.unsqueeze(2).float()

    mask = torch.matmul(mask, to_mask)  # (batch_size, from_seq_len, to_seq_len)
    return mask


class BiLinear(nn.Module):
    def __init__(self, dim):
        super(BiLinear, self).__init__()
        self.dense_1 = Conv1D(in_dim=dim, out_dim=dim,
                              kernel_size=1, stride=1, padding=0, bias=True)
        self.dense_2 = Conv1D(in_dim=dim, out_dim=dim,
                              kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input1, input2):
        output = self.dense_1(input1) + self.dense_2(input2)
        return output


class DualMultiHeadAttentionConvBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        """
        Self-Guided Parallel Attention Module Pytorch Implementation. adapted from : https://github.com/26hzhang/SeqPAN/blob/0512ac358c65856a80e3f3d9bd6aa2e084ba4479/models/modules.py#L73
        """
        super(DualMultiHeadAttentionConvBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)

        self.query = Conv1D(in_dim=dim, out_dim=dim)
        self.f_key = Conv1D(in_dim=dim, out_dim=dim)
        self.f_value = Conv1D(in_dim=dim, out_dim=dim)
        self.s_proj = Conv1D(in_dim=dim, out_dim=dim)
        self.t_key = Conv1D(in_dim=dim, out_dim=dim)
        self.t_value = Conv1D(in_dim=dim, out_dim=dim)
        self.x_proj = Conv1D(in_dim=dim, out_dim=dim)
        self.s_gate = Conv1D(in_dim=dim, out_dim=dim)
        self.x_gate = Conv1D(in_dim=dim, out_dim=dim)
        self.bilinear_1 = BiLinear(dim)
        self.bilinear_2 = BiLinear(dim)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_normt = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        self.guided_dense = Conv1D(in_dim=dim, out_dim=dim)
        self.output_dense = Conv1D(in_dim=dim, out_dim=dim)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        # (batch_size, num_heads, w_seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)
    
    def compute_attention(self, query, key, value, mask):
        attn_value = torch.matmul(query, key.transpose(-1, -2))
        attn_value = attn_value / math.sqrt(self.head_size)
        attn_value += (1.0 - mask) * -1e30
        attn_score = nn.Softmax(dim=-1)(attn_value)
        attn_score = self.dropout(attn_score)
        out = torch.matmul(attn_score, value)
        return out

    def forward(self, from_tensor, to_tensor, from_mask, to_mask):
        x = from_tensor
        from_tensor = self.layer_norm1(from_tensor)  # (batch_size, from_seq_len, dim)
        to_tensor = self.layer_normt(to_tensor)  # (batch_size, to_seq_len, dim)
        # dual multi-head attention layer
        # self-attn projection (batch_size, num_heads, from_seq_len, head_size)
        query = self.transpose_for_scores(self.query(from_tensor))
        f_key = self.transpose_for_scores(self.f_key(from_tensor))
        f_value = self.transpose_for_scores(self.f_value(from_tensor))
        # cross-attn projection (batch_size, num_heads, to_seq_len, head_size)
        t_key = self.transpose_for_scores(self.t_key(to_tensor))
        t_value = self.transpose_for_scores(self.t_value(to_tensor))
        # create attention mask
        s_attn_mask = create_attention_mask(from_mask, from_mask, broadcast_ones=False).unsqueeze(1)
        x_attn_mask = create_attention_mask(from_mask, to_mask, broadcast_ones=False).unsqueeze(1)
        # compute self-attention
        s_value = self.compute_attention(query, f_key, f_value, s_attn_mask)
        s_value = self.combine_last_two_dim(s_value.permute(0, 2, 1, 3))  # (batch_size, from_seq_len, dim)
        s_value = self.s_proj(s_value)
        # compute cross-attention
        x_value = self.compute_attention(query, t_key, t_value, x_attn_mask)
        x_value = self.combine_last_two_dim(x_value.permute(0, 2, 1, 3))  # (batch_size, from_seq_len, dim)
        x_value = self.x_proj(x_value)
        # cross gating strategy
        s_score = nn.Sigmoid()(self.s_gate(s_value))
        x_score = nn.Sigmoid()(self.x_gate(x_value))
        outputs = s_score * x_value + x_score * s_value
        outputs = self.guided_dense(outputs)
        # self-guided
        scores = self.bilinear_1(from_tensor, outputs)
        values = self.bilinear_2(from_tensor, outputs)
        output = nn.Sigmoid()(mask_logits(scores, from_mask.unsqueeze(2))) * values
        outputs = self.output_dense(output)
        # intermediate layer
        output = self.dropout(output)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output


class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0, **kwargs):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(
            in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        # (batch_size, c_seq_len, q_seq_len)
        score = self.trilinear_attention(context, query)
        # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))
        # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)
        output = torch.cat([context, c2q, torch.mul(
            context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand(
            [-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(
            1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res

class CQConcatenate(nn.Module):
    def __init__(self, dim, **kwargs):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim,
                             kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(
            1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        # (batch_size, c_seq_len, 2*dim)
        output = torch.cat([context, pooled_query], dim=2)
        output = self.conv1d(output)
        return output



class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        # shape = (batch_size, seq_length, 1)
        alpha = torch.tensordot(x, self.weight, dims=1)
        if mask is not None:
            alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(
            1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x

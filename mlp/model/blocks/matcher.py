import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp.model.blocks import Conv1D
from mlp.model.utils import mask_logits, PositionalEncoding


class CommonSequenceMatcher(nn.Module):
    def __init__(self, dim, **kwargs):
        super(CommonSequenceMatcher, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1,
                             kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask, *args):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores


class LabelPriorSequenceMatcher(nn.Module):
    def __init__(self, dim, max_snippet_len=400,
                 label_size=3,
                 drop_rate=0.1,
                 kernel_size=1,
                 sampling_type="scheduled",
                 scheduled_sampling_limit=1.0,
                 mask_ratio=0.8,
                 mask_type="structured_frame",
                 **kwargs):
        """
        Label-Prior Sequence Matcher Implementation
        """
        super(LabelPriorSequenceMatcher, self).__init__()
        # PAD LABEL ID=2
        self.PAD_LABEL_ID = 2
        self.conv1d = Conv1D(in_dim=dim, out_dim=1,
                             kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=True)
        self.tok_emb = nn.Embedding(label_size, dim)
        self.pos_embed = PositionalEncoding(
            dim, drop_rate, max_len=max_snippet_len + 1, batch_first=True)
        self.sampling_type = sampling_type
        self.scheduled_sampling_limit = scheduled_sampling_limit
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def forward(self, x, m_mask, cur_step, total_step, idxs, mode="Train"):
        B = x.size(0)
        nfeats = m_mask.sum(1).to(torch.int)
        mask_idxs = idxs.clone() if idxs is not None else torch.zeros_like(m_mask, dtype=torch.int)
        if mode != "Train":
            # Can't see real labels when validating and testing
            mask_idxs[m_mask] = self.PAD_LABEL_ID
            return self._infer(mask_idxs, x, m_mask)
        # calc tf_ratio
        tf_ratio = self.calc_teacher_forcing_ratio(cur_step, total_step)
        if tf_ratio <= 0.0:
            return self._infer(idxs, x, m_mask)
        else:
            # Calculate the number of features to keep
            mask_n = (nfeats.float() * tf_ratio).floor().long()

            if self.mask_type == 'structured_frame':
                # Structured Frame Keeping (Continuous)
                for i in range(B):
                    rand_start = torch.randint(0, nfeats[i] - mask_n[i], (1,)) if nfeats[i]!=mask_n[i] else [0]                   
                    mask_idxs[i, rand_start[0]:rand_start[0] + mask_n[i]] = self.PAD_LABEL_ID
            elif self.mask_type == "random_frame":
                # Random Frame Keeping
                for i in range(B):
                    rand = torch.randperm(nfeats[i])[:mask_n[i]]
                    mask_idxs[i, rand] = self.PAD_LABEL_ID
            else:
                raise NotImplementedError("Not supported mask type ({})".format(self.mask_type))
        return self._infer(mask_idxs, x, m_mask)

    def _infer(self, idxs, x, mask):
        token_embeddings = self.tok_emb(idxs)
        x = x + self.pos_embed(token_embeddings)
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    def calc_teacher_forcing_ratio(self, cur_it, total_step):
        if self.sampling_type == "teacher_forcing":
            return 0.0
        elif self.sampling_type == "fixed":
            return self.mask_ratio
        elif self.sampling_type == "scheduled":  # scheduled sampling
            # linear 
            mask_ratio = self.mask_ratio + (1 - self.mask_ratio) * (cur_it / total_step)
            mask_ratio = min(self.scheduled_sampling_limit, mask_ratio)
            return mask_ratio
        else:  # always sample from the model predictions
            return 1.0

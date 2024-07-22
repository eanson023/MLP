import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 name,
                 pred_lambda: float,
                 smooth: bool = False,
                 extend: float = 0.1,
                 alpha: float = 0.4,
                 **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.name = name
        self.smooth = smooth
        self.extend = extend
        self.alpha = alpha
        self.pred_lambda = pred_lambda
        print("Cross Entropy Loss - ", name)


    def forward(self, net_outs, gts):
        start_logits = net_outs["grounding_start_loc"]
        end_logits = net_outs["grounding_end_loc"]
        start_labels = gts["grounding_s_labels"]
        end_labels = gts["grounding_e_labels"]
        
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return self.pred_lambda*(start_loss + end_loss)

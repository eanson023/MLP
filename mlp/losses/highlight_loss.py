import torch
import torch.nn as nn
import torch.nn.functional as F

class HighLightLoss(nn.Module):
    def __init__(self,
                 name,
                 epsilon=1e-12,
                 highlight_lambda: float = 5.0,
                 **kwargs):
        super(HighLightLoss, self).__init__()
        self.name = name
        self.epsilon = epsilon
        self.w = highlight_lambda
        print("High Light Loss - ", name)

    def forward(self, net_outs, gts):
        scores = net_outs["h_score"]
        labels = gts["grounding_h_labels"]
        mask = gts["motion_masks"]
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels.float())
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + self.epsilon)
        return self.w * loss

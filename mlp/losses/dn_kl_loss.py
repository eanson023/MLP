import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    def __init__(self,
                 name,
                 kl_lambda: float,
                 t: float,
                 epsilon = 1e-8,
                 **kwargs):
        super(KLLoss, self).__init__()
        self.name = name
        self.epsilon = epsilon
        # self.t = nn.Parameter(t * torch.ones([]))
        self.t = t
        self.kl_lambda = kl_lambda
        print("KL Loss - ", name)


    def forward(self, net_outs, gts):
        # with torch.no_grad():
        #     self.t.clamp_(0.7, 2.0)
        mask = gts["motion_masks"]
        start_logits = net_outs["grounding_start_loc"]
        end_logits = net_outs["grounding_end_loc"]
        dn_s_logits = net_outs["dn_start_kl_loc"]
        dn_e_logits = net_outs["dn_end_kl_loc"]
        masked_start_logits = start_logits * mask
        masked_end_logits = end_logits * mask
        masked_dn_s_logits = dn_s_logits * mask
        masked_dn_e_logits = dn_e_logits * mask

        s_prob = F.log_softmax(masked_start_logits / self.t, dim=-1)
        dn_s_prob = F.softmax(masked_dn_s_logits / self.t, dim=-1)
        start_loss = F.kl_div(s_prob, dn_s_prob, reduction='batchmean')
        

        e_prob = F.log_softmax(masked_end_logits / self.t, dim=-1)
        dn_e_prob = F.softmax(masked_dn_e_logits / self.t, dim=-1)
        end_loss = F.kl_div(e_prob, dn_e_prob, reduction='batchmean')
        
        return self.kl_lambda*(start_loss + end_loss)

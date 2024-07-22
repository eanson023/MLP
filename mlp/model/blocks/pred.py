import torch
import torch.nn as nn

from mlp.model.blocks import Conv1D
from mlp.model.utils import mask_logits


class DynamicRNN(nn.Module):
    def __init__(self, dim):
        super(DynamicRNN, self).__init__()
        self.rnn = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, bias=True, batch_first=True,
                            bidirectional=False)

    def forward(self, x, mask):
        out, _ = self.rnn(x)  # (bsz, seq_len, dim)
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(2)
        out = out * mask
        return out


class CommonPredictor(nn.Module):
    def __init__(self, dim, max_pos_len, drop_rate=0.0):
        super(CommonPredictor, self).__init__()
        self.start_encoder = DynamicRNN(dim=dim)
        self.end_encoder = DynamicRNN(dim=dim)
        self.start_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.end_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.start_block = nn.Sequential(
            Conv1D(in_dim=dim, out_dim=dim, kernel_size=1,
                   stride=1, padding=0, bias=True),
            nn.LayerNorm(dim, eps=1e-6),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1,
                   stride=1, padding=0, bias=True)
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=dim, out_dim=dim, kernel_size=1,
                   stride=1, padding=0, bias=True),
            nn.LayerNorm(dim, eps=1e-6),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1,
                   stride=1, padding=0, bias=True)
        )

    def forward(self, x, mask, **kwargs):
        start_features = self.start_encoder(x, mask)  # (batch_size, seq_len, dim)
        end_features = self.end_encoder(start_features, mask)

        start_features = self.start_block(start_features)  # (batch_size, seq_len, 1)
        end_features = self.end_block(end_features)
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)
        outputs = {
            "grounding_start_loc": start_logits,
            "grounding_end_loc": end_logits,
        }
        return outputs


class LabelPriorPredictor(nn.Module):
    def __init__(self, dim, drop_rate=0.0, mask_ratio=0.5):
        super(LabelPriorPredictor, self).__init__()
        self.label_noise_prob = mask_ratio
        self.start_encoders = DynamicRNN(dim)
        self.end_encoders = DynamicRNN(dim)

        # start end mid label
        self.num_classes = 2
        self.label_encoder = nn.Embedding(self.num_classes, dim)
        
        self.start_block = nn.Sequential(
            Conv1D(in_dim=dim, out_dim=dim, kernel_size=1,
                   stride=1, padding=0, bias=True),
            nn.LayerNorm(dim, eps=1e-6),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1,
                   stride=1, padding=0, bias=True),
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=dim, out_dim=dim, kernel_size=1,
                   stride=1, padding=0, bias=True),
            nn.LayerNorm(dim, eps=1e-6),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1,
                   stride=1, padding=0, bias=True),
        )

    def forward(self, x, mask, s_labels, e_labels):
        batch_size, len, dim = x.shape
        s_labels = s_labels.flatten()
        e_labels = e_labels.flatten()
        # perturb labels
        noise_prob = self.label_noise_prob
        noised_s_labels = apply_label_noise(s_labels, noise_prob, self.num_classes)
        noised_e_labels = apply_label_noise(e_labels, noise_prob, self.num_classes)
        noised_s_labels = noised_s_labels.reshape(batch_size,  -1)
        noised_e_labels = noised_e_labels.reshape(batch_size,  -1)
        
        mask = mask.repeat(1, 2)
        
        start_features = x
        # # encoding labels
        noised_s_label_queries = self.label_encoder(noised_s_labels)
        noised_e_label_queries = self.label_encoder(noised_e_labels)
        start_features = torch.cat((start_features, noised_s_label_queries), dim=1)
        start_features = self.start_encoders(start_features, mask)
        end_features = torch.cat((start_features[:, :len], noised_e_label_queries), dim=1)
        end_features = self.end_encoders(start_features, mask)

        start_features = self.start_block(start_features)  # (batch_size, seq_len, 1)
        end_features = self.end_block(end_features)
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)

        dn_s_logits = start_logits[:, len:].contiguous()
        dn_e_logits = end_logits[:, len:].contiguous()
        start_logits = start_logits[:, :len].contiguous()
        end_logits = end_logits[:, :len].contiguous()
        
        outputs = {
            "grounding_start_loc": start_logits,
            "grounding_end_loc": end_logits,
            "dn_start_loc": dn_s_logits,
            "dn_end_loc": dn_e_logits,
            "dn_start_kl_loc": dn_s_logits,
            "dn_end_kl_loc": dn_e_logits

        }
        # self._set_aux_loss(all_dn_s_logits, all_dn_e_logits, outputs)
        return outputs

    @torch.jit.unused
    def _set_aux_loss(self, all_dn_s_logits, all_dn_e_logits, outputs):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        for i, (a, b) in enumerate(zip(all_dn_s_logits, all_dn_e_logits)):
            outputs.update({f"dn_start_loc_{i}": a})
            outputs.update({f"dn_end_loc_{i}": b})
            outputs.update({f"dn_start_kl_loc_{i}": a})
            outputs.update({f"dn_end_kl_loc_{i}": b})


def apply_label_noise(
    labels: torch.Tensor,
    label_noise_prob: float = 0.2,
    num_classes: int = 80,
):
    """
    Args:
        labels (torch.Tensor): Classification labels with ``(num_labels, )``.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        num_classes (int): Number of total categories.

    Returns:
        torch.Tensor: The noised labels the same shape as ``labels``.
    """
    if label_noise_prob > 0:
        p = torch.rand_like(labels.float())
        noised_index = torch.nonzero(p < label_noise_prob).view(-1)
        new_lebels = torch.randint_like(noised_index, 0, num_classes)
        noised_labels = labels.scatter_(0, noised_index, new_lebels)
        return noised_labels
    else:
        return labels
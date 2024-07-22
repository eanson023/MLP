import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare.paramUtil import joint_nums
from prepare.scripts.motion_process import recover_rot, recover_rot_pos, recover_rot_pos_velocity
from mlp.model.blocks import Conv1D


class MotionSpatioConvEncoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 dropout: int,
                 dataset: str,
                 data_rep: str = 'cont_6d',
                 use_spatial_encoder = False,
                 **kwargs):
        super(MotionSpatioConvEncoder, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        
        self.data_rep = data_rep
        self.use_spatial_encoder = use_spatial_encoder
        joint_num = joint_nums[dataset]
        
        f = 6
        # Tesla(guo263) does not perform recovery calculations
        if data_rep == 'guo263':
            f = 263
            joint_num = 1
            use_spatial_encoder = False
        elif data_rep == 'cont_6d_plus_rifke':
            f = 6+3
        elif data_rep == 'cont_6d_plus_rifke_plus_velocity':
            f = 6+3+3
        if use_spatial_encoder:
            embed_dim = f
        else:
            embed_dim = f * joint_num
            self.emb_enc = Conv1D(in_dim=f*joint_num, out_dim=latent_dim)
        
        self.embed_dim = embed_dim

    
    def transform_representation(self, motion):
        if self.data_rep == "guo263":
            return motion
        elif self.data_rep == 'cont_6d':
            motion = recover_rot(motion)
        elif self.data_rep == 'cont_6d_plus_rifke':
            motion = recover_rot_pos(motion)
        elif self.data_rep == 'cont_6d_plus_rifke_plus_velocity':
            motion = recover_rot_pos_velocity(motion)
        return motion

    def forward(self, motion_features, *args):
        b, t = motion_features.shape[:2]
        # convert the input motion in a skeleton-like input
        motion_features = self.transform_representation(motion_features)  # B x seqlen x num_joints x dims
        motion_features = self.drop(motion_features)
        if not self.use_spatial_encoder:
            # convert the input motion in a video-like input
            motion_features = motion_features.view(b, t, -1)
            motion_features = self.emb_enc(motion_features).unsqueeze(2) # (batch_size, seq_len, 1, dim)

        return motion_features

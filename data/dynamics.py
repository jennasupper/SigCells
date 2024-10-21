import numpy as np
import torch

"""Adapted from https://github.com/BoomStarcuc/GeneSegNet/tree/master"""

def gen_pose_target(joints, device, h=256, w=256, sigma=3):
    #print "Target generation -- Gaussian maps"
    if joints.shape[0]!=0:
        joint_num = joints.shape[0]
        gaussian_maps = torch.zeros((joint_num, h, w)).to(device)
        
        for ji in range(0, joint_num):
            gaussian_maps[ji, :, :] = gen_single_gaussian_map(joints[ji, :], h, w, sigma, device)

        # Get background heatmap
        max_heatmap = torch.max(gaussian_maps, 0).values
    else:
        max_heatmap = torch.zeros((h, w)).to(device)
    return max_heatmap

def gen_single_gaussian_map(center, h, w, sigma, device):
    #print "Target generation -- Single gaussian maps"
    '''
    center a gene spot #[2,]
    '''

    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    inds = torch.stack([grid_x,grid_y], dim=0).to(device)
    d2 = (inds[0] - center[0]) * (inds[0] - center[0]) + (inds[1] - center[1]) * (inds[1] - center[1]) #[256,256]
    exponent = d2 / 2.0 / sigma / sigma
    exp_mask = exponent > 4.6052
    exponent[exp_mask] = 0
    gaussian_map = torch.exp(-exponent)
    gaussian_map[exp_mask] = 0
    gaussian_map[gaussian_map>1] = 1

    return gaussian_map
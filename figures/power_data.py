import onnx
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import onnxruntime as ort

import sys
import os
import cv2
import time

sys.path.append("/scratch/user/s4702415/GeneSegNet/GeneSegNet")
from dynamics import gen_pose_target, compute_masks
from transforms import convert_image

sys.path.append("/scratch/user/s4702415/SigCells/data")
from simulations import gen_cells, get_ground_truth, gen_heatmap, display_confusion

sys.path.append("/scratch/user/s4702415/SigCells/experiments/cellpose/simulated")
from si import SI4ONNX

import argparse


if __name__=="__main__":
    #
    d = 56
    n = 3
    r = 4
    colour = (1, 0, 0)

    np.random.seed()

    # path_56 = "/scratch/user/s4702415/trained_models/cellpose/cellpose_n56.onnx/cyto3.onnx"
    path_56 = "/scratch/user/s4702415/trained_models/genesegnet/genesegnet_n56/GeneSegNet_hippocampus_residual_on_style_on_concatenation_off.929131_epoch_499.onnx"

    model_56 = onnx.load(path_56)

    print(f"Power experiment (cellpose), d = {d}, n_cells = {n}, r = {r}, colour = {colour}")

    centre = [(np.random.randint(low=0, high=d), np.random.randint(low=0, high=d)) for _ in range(n)]

    cells, centre = gen_cells(d, n, r, colour, centre)
    s = get_ground_truth(cells)
    gm = torch.zeros(size=(d, d))
    membrane, centre = gen_cells(d, n, r + 2, colour, centre)
    gm = gen_heatmap(centre, d)
    image = torch.stack([cells, gm])
    # try without noise?
    # image = image + torch.tensor(np.random.uniform(low=0.0, high=0.0001, size=(2, d, d)), dtype=torch.float)
    image = image + torch.tensor(np.random.normal(loc=0.0, scale=1, size=(2, d, d)), dtype=torch.float)

    image = image.unsqueeze(0)

    ort_sess = ort.InferenceSession(path_56)
    si_unet = SI4ONNX(model_56, thr=0.5)

    # try:
    output, _ = ort_sess.run(None, {'input': image.numpy()})

    # mask, p = compute_masks(output[0, :2, :, :], output[0, 2, :, :], confidence_threshold=0.5)

    mask = output[0, 2, :, :] > 0.5

    plt.imshow(image[0, 0])
    plt.colorbar()
    plt.savefig("/scratch/user/s4702415/SigCells/figures/cels_sim_1_gsn.png")
    plt.clf()

    plt.imshow(image[0, 1])
    plt.colorbar()
    plt.savefig("/scratch/user/s4702415/SigCells/figures/gm_sim_1_gsn.png")
    plt.clf()

    plt.imshow(np.array(s))
    plt.colorbar()
    plt.savefig("/scratch/user/s4702415/SigCells/figures/cells_gt_1_gsn.png")
    plt.clf()

    plt.imshow(output[0, 2, :, :])
    plt.colorbar()
    plt.savefig("/scratch/user/s4702415/SigCells/figures/sim_output_1_gsn.png")
    plt.clf()

    plt.imshow(mask)
    plt.colorbar()
    plt.savefig("/scratch/user/s4702415/SigCells/figures/mask_1_gsn.png")
    plt.clf()



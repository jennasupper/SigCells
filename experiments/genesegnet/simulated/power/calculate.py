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

sys.path.append("/scratch/user/s4702415/SigCells/experiments/genesegnet/simulated")
from si import SI4ONNX

import argparse


if __name__=="__main__":
    # experiment settings

    parser = argparse.ArgumentParser()
    parser.add_argument("colour")
    args = parser.parse_args()
    
    d = 56
    n = 3
    r = 4
    colour = (float(args.colour), 0, 0)

    path_56 = "/scratch/user/s4702415/trained_models/genesegnet/genesegnet_n56/GeneSegNet_hippocampus_residual_on_style_on_concatenation_off.929131_epoch_499.onnx"
    model_56 = onnx.load(path_56)

    print(f"Power experiment (genesegnet), d = {d}, n_cells = {n}, r = {r}, colour = {colour}")

    cells, centre = gen_cells(d, n, r, colour)
    s = get_ground_truth(cells)
    gm = gen_heatmap(centre, d)
    image = torch.stack([cells, gm])
    # try without noise?
    # image = image + torch.tensor(np.random.uniform(low=0.0, high=0.0001, size=(2, d, d)), dtype=torch.float)
    image = image + torch.tensor(np.random.normal(loc=0.0, scale=0.001, size=(2, d, d)), dtype=torch.float)

    ort_sess = ort.InferenceSession(path_56)
    si_unet = SI4ONNX(model_56, thr=0.5)

    image = image.unsqueeze(0)

    # try:
    output, _ = ort_sess.run(None, {'input': image.numpy()})

    # mask, p = compute_masks(output[0, :2, :, :], output[0, 2, :, :], confidence_threshold=0.5)

    mask = output[0, 2, :, :] > 0.5

    display_confusion(s, mask, d)

    start = time.time()
    p_value = si_unet.inference(image, var=1.0, termination_criterion='decision')
    print(f"p_value = {p_value}")
    print(f"Time = {time.time() - start}")
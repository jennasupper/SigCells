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
from simulations import gen_cells, gen_heatmap, display_confusion, calculate_mean_intensity

sys.path.append("/scratch/user/s4702415/SigCells/experiments/cellpose/simulated")
from si import SI4ONNX

import argparse


if __name__=="__main__":
    # experiment settings

    parser = argparse.ArgumentParser()
    parser.add_argument("variance")
    # parser.add_argument("n")
    args = parser.parse_args()

    # experiment settings
    d = 56
    # n = int(args.n)
    # r = int(args.r)
    n = 3
    r = 4
    colour = (1, 0, 0)
    variance = float(args.variance)
    # variance = 0.001

    path_56 = "/scratch/user/s4702415/trained_models/cellpose/cellpose_n56.onnx/cyto3.onnx"
    model_56 = onnx.load(path_56)

    print(f"Power experiment (cellpose), d = {d}, n_cells = {n}, r = {r}, colour = {colour}, variance = {variance}")
    centre = [(np.random.randint(low=0, high=d), np.random.randint(low=0, high=d)) for _ in range(n)]

    cells, centre, s = gen_cells(d, n, r, colour, centre, variance)
    # s = get_ground_truth(cells)
    membrane, centre, m_s = gen_cells(d, n, r + 2, colour, centre, variance)
    image = torch.stack([membrane, cells])
    # try without noise?
    # image = image + torch.tensor(np.random.uniform(low=0.0, high=0.0001, size=(2, d, d)), dtype=torch.float)
    #image = image + torch.tensor(np.random.normal(loc=0.0, scale=float(args.noise), size=(2, d, d)), dtype=torch.float)

    ort_sess = ort.InferenceSession(path_56)
    si_unet = SI4ONNX(model_56, thr=0.5)

    image = image.unsqueeze(0)

    # try:
    output, _, _, _, _, _ = ort_sess.run(None, {'input': image.numpy()})

    # mask, p = compute_masks(output[0, :2, :, :], output[0, 2, :, :], confidence_threshold=0.5)

    mask = output[0, 2, :, :] > 0.5

    # note: cellpose has more FNs due to absence of second channel
    tp, fp, fn, tn = display_confusion(s, mask, d)

    object_mean, bg_mean = calculate_mean_intensity(tp, fp, fn, tn, image[0, 1, :, :])
    print(f"DAPI Object_mean = {object_mean}, Bg_mean = {bg_mean}")

    print(f"m_s confusion")
    tp, fp, fn, tn = display_confusion(m_s, mask, d)

    object_mean, bg_mean = calculate_mean_intensity(tp, fp, fn, tn, image[0, 0, :, :])
    print(f"Membrane Object_mean = {object_mean}, Bg_mean = {bg_mean}")


    start = time.time()
    p_value = si_unet.inference(image, var=1.0, significance_level=0.05, over_conditioning=True)
    print(f"p_value = {p_value}")
    print(f"Time = {time.time() - start}")
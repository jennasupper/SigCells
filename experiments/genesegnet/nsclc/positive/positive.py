import cv2
import onnx
import pandas as pd

import sys
sys.path.append("/scratch/user/s4702415/SigCells/experiments/genesegnet/simulated")
from si import SI4ONNX
sys.path.append("/scratch/user/s4702415/SigCells/data")
from data_io import get_tile_dims, process_morphology, process_spots
import os
import torch
import time
import tifffile

sys.path.append("/scratch/user/s4702415/GeneSegNet/GeneSegNet")
from dynamics import gen_pose_target
from transforms import make_tiles
# from transforms import normalize99

import numpy as np

if __name__=="__main__":

    path_56 = "/scratch/user/s4702415/trained_models/genesegnet/genesegnet_n56/GeneSegNet_hippocampus_residual_on_style_on_concatenation_off.929131_epoch_499.onnx"
    model_56 = onnx.load(path_56)
    d = 56

    img_path = "/QRISdata/Q7417/Lung5_Rep1/Lung5_Rep1-RawMorphologyImages/20210907_180607_S1_C902_P99_N99_F001_Z001.TIF"

    j_max, i_max = get_tile_dims(img_path, "CosMx", d)

    spots_path = "/scratch/user/s4702415/Honours/models/nsclc/rna_processing/Lung5_Rep1_nsclc_rna_v2.csv"
    rna = pd.read_csv(spots_path, sep=',', header=0, index_col=0)
    rna = rna.values

    while True:
        j = np.random.randint(low=0, high=j_max)
        i = np.random.randint(low=0, high=i_max)
        print(f"NSCLC trial, SI, Positive Genesegnet, i = {i}, j = {j}, d = {d}")


        tile, image_shape, ysub, xsub = process_morphology(img_path, "CosMx", d, j, i, border=0)

        spots = process_spots(rna, image_shape, ysub, xsub, j, i)
        spots = np.array(spots)
        # print(spots)

        heatmap = gen_pose_target(spots, 'cpu', h=d, w=d, sigma=2)

        dapi = tile[1]
        membrane = tile[0]
        if heatmap.max() != 0 and dapi.max() != 0:
            break

    heatmap = torch.tensor(heatmap, dtype=torch.float64) + torch.randn(size=(d, d))

    dapi = torch.tensor(dapi, dtype=torch.float64) + torch.randn(size=(d, d))

    input_x = torch.stack([dapi, heatmap], dim=0)
    input_x = input_x.unsqueeze(0)

    si_unet = SI4ONNX(model_56, thr=0.5)
    start = time.time()
    p_value = si_unet.inference(input_x, var=1.0, termination_criterion="decision", significance_level=0.01)
    print(p_value)

    print(f"time = {time.time() - start}")

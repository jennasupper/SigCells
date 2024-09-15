import cv2
import onnx
import pandas as pd

import sys
sys.path.append("/scratch/user/s4702415/SigCells/experiments/cellpose/simulated")
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

    path_56 = "/scratch/user/s4702415/trained_models/cellpose/cellpose_n56.onnx/cyto3.onnx"
    model_56 = onnx.load(path_56)
    d = 56

    img_path = "/QRISdata/Q7417/Xenium_Prime_Human_Skin_FFPE_outs/morphology_focus/morphology_focus_0000.ome.tif"
    loaded = tifffile.imread(img_path)

    j_max, i_max = get_tile_dims(img_path, "Xenium", d)

    while True:
        j = np.random.randint(low=0, high=j_max)
        i = np.random.randint(low=0, high=i_max)
        print(f"Melanoma trial, SI, Positive Cellpose, i = {i}, j = {j}, d = {d}")

        # spots_path = "/scratch/user/s4702415/Honours/models/hippocampus/processing/DAPI_spots_3-1_left_processed.csv"
        # rna = pd.read_csv(spots_path, sep=',', header=0, index_col=0)
        # rna = rna.values
        
        tile, image_shape, ysub, xsub = process_morphology(img_path, "Xenium", d, j, i, border=0, loaded=loaded)

        dapi = tile[1]
        membrane = tile[0]
        if membrane.max != 0 and dapi.max() != 0:
            break

    membrane = torch.tensor(membrane, dtype=torch.float64)

    dapi = torch.tensor(dapi, dtype=torch.float64)

    input_x = torch.stack([membrane, dapi], dim=0)
    input_x = input_x.unsqueeze(0)

    si_unet = SI4ONNX(model_56, thr=0.5)
    start = time.time()
    p_value = si_unet.inference(input_x, var=1.0, termination_criterion="decision")
    print(p_value)

    print(f"time = {time.time() - start}")
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

sys.path.append("/scratch/user/s4702415/GeneSegNet/GeneSegNet")
from dynamics import gen_pose_target
from transforms import make_tiles
# from transforms import normalize99
import onnxruntime as ort

import numpy as np

if __name__=="__main__":

    path_56 = "/scratch/user/s4702415/trained_models/cellpose/cellpose_n56.onnx/cyto3.onnx"
    model_56 = onnx.load(path_56)
    d = 56

    # img_path = "/QRISdata/Q7417/data/hippocampus/raw/CA1DapiBoundaries_3-1_left.tif"
    img_path = "/QRISdata/Q7417/data/hippocampus/raw/DAPI_3-1_left.jpg"

    # j_max, i_max = get_tile_dims(img_path, "Custom", d)

    # while True:
    #     j = np.random.randint(low=0, high=j_max)
    #     i = np.random.randint(low=0, high=i_max)
        
    #     print(f"Hippocampus trial, SI, Positive Cellpose, i = {i}, j = {j}, d = {d}")

    #     # spots_path = "/scratch/user/s4702415/Honours/models/hippocampus/processing/DAPI_spots_3-1_left_processed.csv"
    #     # rna = pd.read_csv(spots_path, sep=',', header=0, index_col=0)
    #     # rna = rna.values
        
    #     tile, image_shape, ysub, xsub = process_morphology(img_path, "Custom", d, j, i, border=0)

    #     dapi = tile[0]
    #     if dapi.max() > 0.5:
    #         zeros = torch.zeros(size=(d, d), dtype=torch.float)

    #         dapi = torch.tensor(dapi, dtype=torch.float)

    #         input_x = torch.stack([dapi, zeros], dim=0)
    #         input_x = input_x.unsqueeze(0)
    #         ort_sess = ort.InferenceSession(path_56)
    #         output, _, _, _, _, _ = ort_sess.run(None, {'input': input_x.numpy()})
    #         if output[0, 2, :, :].max() > 4:
    #             break

    i = 41
    j = 43
        
    print(f"Hippocampus trial, SI, Positive Cellpose, i = {i}, j = {j}, d = {d}")


        
    tile, image_shape, ysub, xsub = process_morphology(img_path, "Custom", d, j, i, border=0)

    dapi = tile[0]
    zeros = torch.zeros(size=(d, d), dtype=torch.float)

    dapi = torch.tensor(dapi, dtype=torch.float)

    input_x = torch.stack([dapi, zeros], dim=0)
    input_x = input_x.unsqueeze(0)

    # zeros = torch.zeros(size=(d, d), dtype=torch.float64) + torch.randn(size=(d, d))

    # dapi = torch.tensor(dapi, dtype=torch.float64)

    # input_x = torch.stack([dapi, zeros], dim=0)
    # input_x = input_x.unsqueeze(0)

    si_unet = SI4ONNX(model_56, thr=0.5)
    start = time.time()
    p_value = si_unet.inference(input_x, var=1.0, termination_criterion="decision", significance_level=0.01)
    print(p_value)

    print(f"time = {time.time() - start}")


        # spots = process_spots(rna, image_shape, ysub, xsub, j, i)
    # spots = np.array(spots)

    #         heatmap = gen_pose_target(spots, 'cpu', h=d, w=d, sigma=1)

            # if heatmap.max() == 0:
            #     continue

            # plt.imshow(tile[0])
            # plt.colorbar()
            # plt.savefig(os.path.join(test_path, f"{j}_{i}.png"))
            # plt.clf()

            # plt.imshow(heatmap)
            # plt.colorbar()
            # plt.savefig(os.path.join(test_path, f"{j}_{i}_heatmap.png"))
            # plt.clf()

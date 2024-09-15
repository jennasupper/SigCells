import cv2
import onnx

import sys
sys.path.append("/scratch/user/s4702415/SigCells/experiments/cellpose/simulated")
from si import SI4ONNX
sys.path.append("/scratch/user/s4702415/SigCells/data")
from center_crop import center_crop
import os
import torch
import time

sys.path.append("/scratch/user/s4702415/GeneSegNet/GeneSegNet")
# from transforms import normalize99

import numpy as np

if __name__=="__main__":

    path_56 = "/scratch/user/s4702415/trained_models/cellpose/cellpose_n56.onnx/cyto3.onnx"
    model_56 = onnx.load(path_56)
    d = 56

    img_path = "/QRISdata/Q7417/data/hippocampus/train/DAPI_3-1_left/images"
    ls = os.listdir(img_path)
    index = np.random.randint(low=0, high=len(ls))
    image = cv2.imread(os.path.join(img_path, ls[index]))
    print(os.path.join(img_path, ls[index]))

    image = center_crop(image, (d, d))

    dapi = image[:, :, 0]
    zeros = torch.zeros(size=(d, d), dtype=torch.float) + torch.randn(size=(d, d))
    dapi = torch.randn(size=(d, d))

    input_x = torch.stack([dapi, zeros], dim=0)

    input_x = input_x.unsqueeze(0)

    print(f"SI P-Value, Cellpose, Negative, Hippocampus, index = {index}")

    si_unet = SI4ONNX(model_56, thr=0.5)
    start = time.time()
    p_value = si_unet.inference(input_x, var=1.0, termination_criterion="decision")
    print(p_value)

    print(f"time = {time.time() - start}")

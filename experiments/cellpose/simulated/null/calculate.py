import onnx
import torch
import numpy as np
import sys
sys.path.append("/scratch/user/s4702415/SigCells/experiments/cellpose/simulated")
from si import SI4ONNX
import time

import sys
# sys.path.append("/scratch/user/s4702415/SigCells/data")


if __name__=="__main__":
    
    print(f"SI P-Value, Negative control Cellpose N(0, 1)")
    # load model
    path_56 = "/scratch/user/s4702415/trained_models/cellpose/cellpose_n56.onnx/cyto3.onnx"
    model = onnx.load(path_56)

    d = 56
    # should i use uniform or Gaussian noise
    noise = torch.randn(1, 2, d, d)

    start = time.time()
    si_unet = SI4ONNX(model, thr=0.5)
    p_value = si_unet.inference(noise, var=1.0)
    print(p_value)
    print(f"time = {time.time() - start}")
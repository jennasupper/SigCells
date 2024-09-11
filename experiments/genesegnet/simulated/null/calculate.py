import onnx
import torch
import numpy as np
from si import SI4ONNX
import time

import sys
# sys.path.append("/scratch/user/s4702415/SigCells/data")


if __name__=="__main__":
    
    print(f"SI P-Value, Negative control GeneSegNet N(0, 1)")
    # load model
    path_56 = "/scratch/user/s4702415/trained_models/genesegnet/genesegnet_n56/GeneSegNet_hippocampus_residual_on_style_on_concatenation_off.929131_epoch_499.onnx"
    model = onnx.load(path_56)

    d = 56
    # should i use uniform or Gaussian noise - Gaussian
    noise = torch.randn(1, 2, d, d)

    start = time.time()
    si_unet = SI4ONNX(model, thr=0.5)
    p_value = si_unet.inference(noise, var=1.0)
    print(p_value)
    print(f"time = {time.time() - start}")
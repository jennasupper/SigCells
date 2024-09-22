import onnx
import torch
import sys

sys.path.append("/scratch/user/s4702415/SigCells/data")

from data_io import process_morphology, process_spots
import numpy as np
import onnxruntime as ort

import matplotlib.pyplot as plt
import tifffile
import pandas as pd

sys.path.append("/scratch/user/s4702415/GeneSegNet/GeneSegNet")
from dynamics import gen_pose_target


if __name__=="__main__":
    # i = 65
    # j = 78
    # d = 56

    path_56 = "/scratch/user/s4702415/trained_models/cellpose/cellpose_n56.onnx/cyto3.onnx"
    model_56 = onnx.load(path_56)

    # img_path = "/QRISdata/Q7417/data/hippocampus/raw/CA1DapiBoundaries_3-1_left.tif"

    # tile, image_shape, ysub, xsub = process_morphology(img_path, "Custom", d, j, i, border=0)

    # dapi = tile[0]

    # zeros = torch.zeros(size=(d, d), dtype=torch.float) + torch.randn(size=(d, d))

    # dapi = torch.tensor(dapi, dtype=torch.float)

    # input_x = torch.stack([dapi, zeros], dim=0)
    # input_x = input_x.unsqueeze(0)

    # ort_sess = ort.InferenceSession(path_56)
    # # si_unet = SI4ONNX(model_56, thr=0.5)

    # # try:
    # output, _, _, _, _, _ = ort_sess.run(None, {'input': input_x.numpy()})

    # plt.imshow(dapi)
    # plt.colorbar()
    # plt.savefig(f"/scratch/user/s4702415/SigCells/figures/dapi_{i}_{j}.png")
    # plt.clf()

    # plt.imshow(output[0, 2, :, :])
    # plt.colorbar()
    # plt.savefig(f"/scratch/user/s4702415/SigCells/figures/prob_{i}_{j}.png")
    # plt.clf()

    # plt.imshow(output[0, 2, :, :] > 0.5)
    # plt.colorbar()
    # plt.savefig(f"/scratch/user/s4702415/SigCells/figures/mask_{i}_{j}.png")
    # plt.clf()

    i = 65
    j = 78
    d = 56

    path_56 = "/scratch/user/s4702415/trained_models/genesegnet/genesegnet_n56/GeneSegNet_hippocampus_residual_on_style_on_concatenation_off.929131_epoch_499.onnx"
    model_56 = onnx.load(path_56)
    d = 56

    # img_path = "/QRISdata/Q7417/Xenium_Prime_Human_Skin_FFPE_outs/morphology_focus/morphology_focus_0000.ome.tif"
    # loaded = tifffile.imread(img_path)

    # spots_path = "/scratch/user/s4702415/Honours/models/xenium/melanoma_transcripts.csv"
    # rna = pd.read_csv(spots_path, sep=',', header=0, index_col=0)
    # rna = rna.values

    # tile, image_shape, ysub, xsub = process_morphology(img_path, "Xenium", d, j, i, border=0, loaded=loaded)

    # spots = process_spots(rna, image_shape, ysub, xsub, j, i, x_min=30000, y_min=5000)
    # spots = np.array(spots)
    # heatmap = gen_pose_target(spots, 'cpu', h=d, w=d, sigma=2)
    # dapi = tile[1]

    # heatmap = heatmap

    img_path = "/QRISdata/Q7417/data/hippocampus/raw/CA1DapiBoundaries_3-1_left.tif"

    tile, image_shape, ysub, xsub = process_morphology(img_path, "Custom", d, j, i, border=0)

    dapi = tile[0]

    dapi = torch.tensor(dapi, dtype=torch.float)

    input_x = torch.stack([dapi, torch.zeros(size=(d, d))], dim=0)
    input_x = input_x.unsqueeze(0)

    ort_sess = ort.InferenceSession(path_56)
    # # si_unet = SI4ONNX(model_56, thr=0.5)

    # # try:
    output, _ = ort_sess.run(None, {'input': input_x.numpy()})

    plt.imshow(dapi)
    plt.colorbar()
    plt.savefig(f"/scratch/user/s4702415/SigCells/figures/dapi_hippo_{i}_{j}.png")
    plt.clf()

    plt.imshow(output[0, 2, :, :])
    plt.colorbar()
    plt.savefig(f"/scratch/user/s4702415/SigCells/figures/prob_hippo_{i}_{j}.png")
    plt.clf()

    plt.imshow(output[0, 2, :, :] > 0.5)
    plt.colorbar()
    plt.savefig(f"/scratch/user/s4702415/SigCells/figures/mask_hippo_{i}_{j}.png")
    plt.clf()

    # plt.imshow(heatmap)
    # plt.colorbar()
    # plt.savefig(f"/scratch/user/s4702415/SigCells/figures/heatmap_hippo_{i}_{j}.png")
    # plt.clf()

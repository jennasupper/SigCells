import sys
from time import time

import numpy as np
import torch

from sicore import SelectiveInferenceNorm
sys.path.append("/scratch/user/s4702415/SigCells/SigCells")

import si
from utils import threshold

sys.path.append("/scratch/user/s4702415/SigCells/data")
from dynamics import compute_masks

import matplotlib.pyplot as plt

class SI4ONNX(si.SI4ONNX):
    def __init__(self, model, thr):
        super().__init__(model)
        self.model = model
        self.thr = torch.tensor(thr, dtype=torch.float64)

    def construct_hypothesis(self, X):
        self.shape = X.shape  # use in 'algorithm' method
        input_x = X
        input_vec = input_x.reshape(-1).double()

        output_x, _, _, _, _, _= self.si_model.forward(input_x)

        mask = output_x[0, 2, :, :]
        mask = torch.stack([mask, mask])

        anomaly_region = mask > self.thr
        anomaly_index = anomaly_region.reshape(-1).int()
        self.anomaly_index_obs = anomaly_index

        # construct eta vector
        eta = (
            anomaly_index / torch.sum(anomaly_index)
            - (1 - anomaly_index) / torch.sum(1 - anomaly_index)
        ).double()

        self.si_calculator = SelectiveInferenceNorm(
            input_vec, self.var, eta, use_torch=True
        )
        # self.si_calculator_naive = NaiveInferenceNorm(
        #     input_vec, self.var, eta, use_torch=True
        # )
        
        # assert not torch.isnan(self.si_calculator.stat) # if nan, anomaly_index is all the same
        assert not torch.isnan(self.si_calculator.stat)

        # sd: float = np.sqrt(self.si_calculator.eta_sigma_eta)
        # self.max_tail = sd * 10 + torch.abs(self.si_calculator.stat) # Exploration range

    def model_selector(self, anomaly_index):
        return torch.all(torch.eq(self.anomaly_index_obs, anomaly_index))

    def algorithm(self, a, b, z):
        x = a + b * z # a, b need to be torch.tensor(dtype=torch.float64)
        B, C, H, W = self.shape
        input_x = x.reshape(B, C, H, W).double()
        input_bias = torch.zeros(B, C, H, W).double()
        input_a = a.reshape(B, C, H, W)
        input_b = b.reshape(B, C, H, W)
        INF = torch.tensor(float("inf"), dtype=torch.float64)
        l = -INF
        u = INF
        # l = -self.max_tail
        # u = self.max_

        output_x, output_bias, output_a, output_b, l, u = self.si_model.forward_si(
            input_x, input_bias, input_a, input_b, l, u
        )
        conf_x = output_x[0][:, 2, :, :]
        conf_x = torch.stack([conf_x, conf_x], dim=1)
        conf_bias = output_bias[0][:, 2, :, :]
        conf_bias = torch.stack([conf_bias, conf_bias], dim=1)
        conf_a = output_a[0][:, 2, :, :]
        conf_a = torch.stack([conf_a, conf_a], dim=1)
        conf_b = output_b[0][:, 2, :, :]
        conf_b = torch.stack([conf_b, conf_b], dim=1)

        conf_l = l[0]
        conf_u = u[0]

        anomaly_index, l, u = threshold(
            self.thr, conf_x, conf_bias, conf_a, conf_b, conf_l, conf_u,
            apply_abs=False,
            use_sigmoid=False # If sigmoid is used in the final output layer
        )

        return anomaly_index, [l, u]
    
    # def naive_inference(self, inputs, var, **kwargs):
    #     self.var = var
    #     self.construct_hypothesis(inputs)
    #     result = self.si_calculator_naive.inference()
    #     return result
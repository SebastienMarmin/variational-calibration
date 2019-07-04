# Original code by Simone Rossi

import torch
import torch.nn as nn

from typing import Union

from . import BaseNet
from vcal.stats import GaussianMatrix
from torch.distributions import Normal

import numpy as np # TODO remove this line


def log_cond_prob( output: torch.Tensor,
                      latent_val: torch.Tensor,varia) -> torch.Tensor:
    log_noise_var = np.log(varia)
    result = - 0.5 * (log_noise_var + np.log2(np.pi) +
                    torch.pow(output - latent_val, 2) * np.exp(-log_noise_var))

    return result

class RegressionNet(BaseNet):

    def __init__(self, *args,**kwargs):
        super(RegressionNet, self).__init__(*args,**kwargs)
        self.likelihood = GaussianMatrix(1,1,centered=True)
        self.likelihood.col_cov.parameter.requires_grad = False ## more user friendly here

    def compute_error(self, Y_pred: torch.Tensor, Y_true: torch.Tensor,n_over_m) -> torch.Tensor:
        return torch.sqrt(torch.mean(torch.pow((Y_true - Y_pred), 2))/n_over_m).unsqueeze(0)

    def compute_nell(self, Y_pred, Y_true, n_over_m) -> torch.Tensor:    
        return - n_over_m * torch.sum(torch.mean(self.likelihood.log_prob((Y_true - Y_pred).unsqueeze(-1).unsqueeze(-1)), 0))
        # unsqueeze twice because self.likelihood is a one-by-one Gaussian matrix and all the shapes of Y are for i.i.d. repetitions
    
    def initialize(self,X,Y):
        self.eval()
        f = list(self.layers)[0] # TODO more layers, with Rossi 2018
        f.set_to_posterior(X,Y,self.likelihood.row_cov.parameter.item())
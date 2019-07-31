# Original code by Simone Rossi

import torch
import torch.nn as nn

from typing import Union

from . import BaseNet
from vcal.stats import GaussianVector
from torch.distributions import Normal

class RegressionNet(BaseNet):

    def __init__(self, *args,**kwargs):
        super(RegressionNet, self).__init__(*args,**kwargs)
        self.likelihood = GaussianVector(1,centered=True)
        self.likelihood.optimize(False)

    def compute_error(self, Y_pred: torch.Tensor, Y_true: torch.Tensor,n_over_m):
        return torch.sqrt(torch.mean(torch.pow((Y_true - Y_pred), 2))/n_over_m).unsqueeze(0)

    def compute_nell(self, Y_pred, Y_true, n_over_m) -> torch.Tensor:    
        return - n_over_m * torch.sum(torch.mean(self.likelihood.log_prob((Y_true - Y_pred).unsqueeze(-1)), 0))
        # unsqueeze because self.likelihood is a one Gaussian vector and
        # each component of Y are for i.i.d. repetitions

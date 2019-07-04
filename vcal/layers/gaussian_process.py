import numpy as np
import torch
import torch.nn as nn
from . import BaseLayer
from . import Constant
from ..utilities import log
from ..kernels import Matern
import abc

class GaussianProcess(BaseLayer, metaclass=abc.ABCMeta):
    def __init__(self, in_features, out_features, **kwargs):
        super(GaussianProcess, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.prior_means = Constant(out_features) #  can also be any type of layer, e.g. learnable function  TOTEST
        self.prior_means.optimize(False)   # can be changed by user but default behavior is to not learn the prior mean
        self.pf = log() # univariate transfo for lengthscale (log) use pf.i() for inverse transfo
        self._lengthscales = nn.Parameter(self.pf(np.sqrt(self.in_features)*.2*torch.ones(in_features)))#nn.Parameter(self.pf(np.sqrt(self.in_features)*.2*torch.ones(in_features)))
        # can be size 1 or size in_features (or TODO? in_features times out_features)

        self._stddevs = nn.Parameter(torch.ones(out_features)) # can be size 1 or size out_features
        self.cov_structure = Matern(self.in_features,smoothness=np.Inf)

    @property
    def variances(self):
        return self._stddevs**2
    @variances.setter
    def variances(self,x):
        self._stddevs.data = x.sqrt()

    @property
    def lengthscales(self):
        return self.pf.i(self._lengthscales)
    @lengthscales.setter
    def lengthscales(self,x):
        self._lengthscales.data =  self.pf(x)

    @property
    def stddevs(self):
        return self._stddevs
    @stddevs.setter
    def stddevs(self,x):
        self._stddevs.data = x

    def fix_hyperparameters(self):
        self.lengthscales.requires_grad = False
        self.prior_means.optimize(False)
        self.prior_variances.requires_grad = False
    @abc.abstractmethod
    def reset_parameters(self):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def set_to_prior(self):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def set_to_posterior(self,X,Y,noise_distribution=None):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def kl_divergence(self):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def forward(self, input): # nmc times n times in_features
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    def optimize(self, train: bool = True):
        for param in self.parameters():
            param.requires_grad = train


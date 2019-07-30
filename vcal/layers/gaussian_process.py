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
        self._means = nn.Parameter(torch.zeros(out_features),requires_grad=False)# tensor of size [1] (constant mean accross output dim) or [out_features] #  can also be any type of layer, e.g. learnable function, as long as it returns a tensor  TOTEST
        self._self_means_is_tensor = True
        
        
        self.pf = log() # univariate transfo for lengthscale (log) use pf.i() for inverse transfo
        self._lengthscales = nn.Parameter(self.pf(np.sqrt(self.in_features)*.2*torch.ones(in_features)))#nn.Parameter(self.pf(np.sqrt(self.in_features)*.2*torch.ones(in_features)))
        # can be size 1 or size in_features (or TODO? in_features times out_features)

        self._stddevs = nn.Parameter(torch.ones(out_features)) # can be size 1 or size out_features
        self.cov_structure = Matern(self.in_features,smoothness=np.Inf)

    @property
    def mean_function(self):
        if self._self_means_is_tensor:
            return (lambda x: self._means)
        else:
            M = self._means
    @mean_function.setter
    def mean_function(f):
        self._means = f
        if issubclass(type(f),torch.Tensor):
            self._self_means_is_tensor = True
        else:
            self._self_means_is_tensor = False


    @property
    def means(self): # may return a function and not a tensor
        return self._means
    @means.setter
    def means(self,x):#tensor #  can also be any type of callable returning a tensor, like layer
        # e.g. learnable function  TOTEST
        try:
            self._means = x
        except TypeError:
            self._means = torch.nn.Parameter(x,requires_grad=self._mean.requires_grad)

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
        self._lengthscales.requires_grad = False
        try:
            self.means.optimize(False)
        except AttributeError:
            self.means.requires_grad = False
        self._stddevs.requires_grad = False
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
    def forward(self, input): 
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    def optimize(self, train: bool = True):
        for param in self.parameters():
            param.requires_grad = train


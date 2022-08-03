
# Original code by Simone Rossi

import torch
import abc

import logging
logger = logging.getLogger(__name__)


class BaseLayer(torch.nn.Module, abc.ABC):
    def __init__(self, **kwargs):
        super(BaseLayer, self).__init__()
        self.nmc_train = kwargs.pop('nmc_train') if 'nmc_train' in kwargs else 1
        self.nmc_test = kwargs.pop('nmc_test') if 'nmc_test' in kwargs else 1
        self.dtype = kwargs.pop('dtype') if 'dtype' in kwargs else torch.float32
        self.nmc = self.nmc_train
        self.eval()

    def kl_divergence(self):
        return 0
        # a subclass needs to implement an homonymal method for introducing in the loss
        # Bayesian regularization of some parameters.

    def train(self, training=True):
        self.nmc = self.nmc_train if training else self.nmc_test

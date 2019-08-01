# Adapted from Simone Rossi 2018

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(f):
        return f



from torch.utils import hooks as hooks
from torch.utils.data import DataLoader

import numpy as np
import torch
from typing import Union

from . import BaseInitializer
from ...utilities import SingleSpaceBatchLoader
from ...nets import CalibrationNet

import logging
logger = logging.getLogger(__name__)


def save_input_to_layer(self, input, output):
    self.input_to_layer = input[0]
    self.output_to_layer = output[0]


class IBLMInitializer(BaseInitializer):

    def __init__(self, model, train_dataloader, noise_var=.01,device=None):
        """
        Implements a I-BLM initializer
        Args:
            model:
            train_dataloader (SingleSpaceBatchLoader):
            device:
            noise_var: 
        """
        super(IBLMInitializer, self).__init__(model)
        self.device = device
        self.train_dataloader = train_dataloader
        self.train_dataloader_iterator = self.train_dataloader.iterable(cycle=True,out_device=device)
        self.noise_cov = noise_var * torch.ones(1,1, device=self.device)
        
        logger.info('Initialization with I-BLM')

    def _initialize_layer(self, layer, layer_index=None):

        hook_hadler = layer.register_forward_hook(save_input_to_layer)  # type: hooks.RemovableHandle

        in_features = layer.in_features
        out_features = layer.out_features

        Y_dim = self.train_dataloader.out_dims[-1]
        
        """
        try:
            X, Y = next(self.train_dataloader_iterator)
        except StopIteration:
            self.train_dataloader_iterator = self.train_dataloader.iterable(cycle=True,out_device=self.device)
            X, Y = next(self.train_dataloader_iterator)
        """

        for out_index in tqdm(range(out_features)):

            if out_index % Y_dim == 0:
                try:
                    X, Y = next(self.train_dataloader_iterator)
                except StopIteration:
                    self.train_dataloader_iterator = self.train_dataloader.iterable(cycle=True,out_device=self.device)
                    X, Y = next(self.train_dataloader_iterator)
            if issubclass(type(self.model),CalibrationNet):
                # need some data manip in calibration as the output of the discrepancy is not the data output
                X,Y = self.model.give_discrepancy_input_output(X,Y)
            index_Y = out_index % Y_dim  # np.random.random_integers(0, Y.size(-1) - 1)
            if layer_index == 0:
                new_in_data = X
            else:
                # Run a forward pass (the hook will save the input to the layer)
                self.model(X)
                new_in_data = layer.input_to_layer.mean(0)
            layer.set_to_posterior(new_in_data,Y[...,index_Y],self.noise_cov,output_index=out_index)
        hook_hadler.remove()


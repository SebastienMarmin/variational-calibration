# Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# TODO: zoe has the previous version installed

from torch.utils import hooks as hooks
from torch.utils.data import DataLoader

import numpy as np
import torch
from typing import Union


from . import BaseInitializer
from ..layers import VariationalLinear, VariationalConv2d
from ..distributions import FullyFactorizedMatrixGaussian

import logging
logger = logging.getLogger(__name__)


def save_input_to_layer(self, input, output):
    self.input_to_layer = input
    self.output_to_layer = output


class LSUVInitializer(BaseInitializer):

    def __init__(self, model, train_dataloader: DataLoader,
                 tollerance: float, max_iter: int, device: torch.device):
        super(LSUVInitializer, self).__init__(model)
        self.train_dataloader = train_dataloader
        self.train_dataloader_iterator = iter(self.train_dataloader)
        self.tollerance = tollerance
        self.max_iter = max_iter
        self.device = device

        logger.info('Initialization with LSUV')

    def _initialize_layer(self, layer, layer_index=None):

        hook_hadler = layer.register_forward_hook(save_input_to_layer)  # type: hooks.RemovableHandle

        torch.nn.init.orthogonal_(layer.posterior_weights.mean)

        current_output_variance = torch.tensor(np.inf)
        step = 0

        while torch.abs(current_output_variance - 1.) > self.tollerance:
            if step >= self.max_iter:
                break
            step += 1
            try:
                data, t = next(self.train_dataloader_iterator)
            except StopIteration:
                self.train_dataloader_iterator = iter(self.train_dataloader)
                data, t = next(self.train_dataloader_iterator)

            data = data.to(self.device)
            # Run a forward pass (the hook will save the input to the layer
            self.model(data)
            # Calculate its variance
            current_output_variance = layer.output_to_layer.var()

            layer.posterior_weights.mean.data = layer.posterior_weights.mean / torch.sqrt(current_output_variance)

        logger.info('Variance at layer %d (iter #%d): %.3f' % (layer_index, step, current_output_variance.cpu()))

        if isinstance(layer.posterior_weights, FullyFactorizedMatrixGaussian):
            in_features = layer.posterior_weights.n
            var = 2. / in_features
            layer.posterior_weights.logvars.data.fill_(np.log(var))
        else:
            logger.warning('Distribution not available yet for this type of initialization. Skipping')
        
        hook_hadler.remove()

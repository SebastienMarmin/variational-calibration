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

import numpy as np
import torch
from typing import Union


from . import BaseInitializer
from ..layers import VariationalLinear, VariationalConv2d
from ..distributions import FullyFactorizedMatrixGaussian

import logging
logger = logging.getLogger(__name__)


class OrthogonalInitializer(BaseInitializer):

    def __init__(self, model, ):
        super(OrthogonalInitializer, self).__init__(model)

        logger.info('Initialization with Orthogonal Matrix')

    def _initialize_layer(self, layer, layer_index=None):
        if isinstance(layer.posterior_weights, FullyFactorizedMatrixGaussian):
            torch.nn.init.orthogonal_(layer.posterior_weights.mean)
            var = 2. / layer.posterior_weights.n
            layer.q_posterior_W.logvars.data.fill_(np.log(var))
        else:
            logger.warning('Distribution not available yet for this type of initialization. Skipping')

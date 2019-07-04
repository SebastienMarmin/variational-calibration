#  Copyright (c) 2019
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  Authors:
#      Simone Rossi <simone.rossi@eurecom.fr>
#      Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
#

from typing import Union

from . import BaseInitializer
from ..layers import VariationalLinear, VariationalConv2d
from ..distributions import FullyFactorizedMatrixGaussian

import logging
logger = logging.getLogger(__name__)


class UninformativeInitializer(BaseInitializer):

    def __init__(self, model, ):
        super(UninformativeInitializer, self).__init__(model)

        logger.info('Initialization with Uninformative posterior')

    def _initialize_layer(self, layer, layer_index=None):
        if isinstance(layer.posterior_weights, FullyFactorizedMatrixGaussian):
            layer.posterior_weights.mean.data.fill_(0.)
            layer.posterior_weights.logvars.data.fill_(0.)
        else:
            logger.warning('Distribution not available yet for this type of initialization. Skipping')

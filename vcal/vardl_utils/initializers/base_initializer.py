# Adapted from Rossi 2018

from time import time
import abc
import torch
import torch.nn as nn
from typing import Union

from ...layers import BaseLayer
from ...nets import BaseNet

import logging
logger = logging.getLogger(__name__)


class BaseInitializer(abc.ABC):
    def __init__(self, model):
        self.model = model  # type: Union[nn.Module, BaseBayesianNet]
        #self.layers = []
        #self._layers_to_initialize()

    #def _layers_to_initialize(self):
    #    for i, layer in enumerate(self.model.modules()):
    #        if issubclass(type(layer), BaseLayer):
    #            self.layers.append((i, layer))

    @abc.abstractmethod
    def _initialize_layer(self, layer, layer_index=None):
        raise NotImplementedError()

    def initialize(self):
        self.model.eval()
        t_start = time()
        i = 0
        for layer in self.model.layers:#self.layers:
        #for i, layer in self.model.layers:#self.layers:
            logger.info('Initialization of layer %d' % i)
            self._initialize_layer(layer, i)
            i += 1
        t_end = time()
        logger.info('Initialization done in %.3f sec.' % (t_end - t_start))

    def __repr__(self):
        return str(self.layers)

# Original code by Simone Rossi
import abc
import os
import torch
import humanize

from typing import Union


from ..layers import BaseLayer
from ..stats import GaussianMatrix


import logging
logger = logging.getLogger(__name__)


class BaseNet(torch.nn.Module):

    def __init__(self, layers=None):
        super(BaseNet, self).__init__()
        if issubclass(type(layers),BaseLayer):
            self.layers = torch.nn.Sequential(layers)
        elif isinstance(layers, torch.nn.Sequential):
            self.layers = layers
            
        #  torch.nn.Sequential() of BaseLayer
        # The point is the self.forward() calls list(self.layers)[0].nmc
        # Can be left to None if a child class defines its own forward
        self.nmc = None
        self.eval()

    def train(self, training_mode=True):
        super().train(training_mode)  # acts on all submodules
        try:
            self.nmc = list(self.layers)[0].nmc
        except AttributeError:
            pass

    def kl_divergence(self):
        total_dkl = 0.
        for layer in self.modules():  # type: Union[BaseLayer, nn.Module]
            total_dkl += layer.kl_divergence() if issubclass(type(layer), BaseLayer) else 0
        return total_dkl

    """ 
    def forward(self, input):
        input = input * torch.ones(self.architecture.nmc, *input.size()).to(input.device)  # type: torch.Tensor
        # TODO: check how to retrieve nmc
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return nn.parallel.data_parallel(self.architecture, inputs=input, dim=1)
        else:
            return self.architecture(input)
    """

    def forward(self, input,input_nmc_rep=True):
        if input_nmc_rep:
            input_rep = input.expand(torch.Size([self.nmc])+input.size())
        else:
            input_rep = input
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not str(input_rep.device)=="cpu":
            #TODO check, no str()
            return torch.nn.parallel.data_parallel(self.layers, inputs=input_rep, dim=1)
        else:
            return self.layers(input_rep)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

        

    def save_model(self, path):
        logger.info('Saving model in %s' % path)
        torch.save(self.state_dict(), path)
        logger.info('Model saved (%s)' % humanize.naturalsize(os.path.getsize(path), gnu=True))

    def load_model(self, path):
        logger.info('Loading model from %s' % path)
        if torch.cuda.is_available():
            map_location= 'cuda'
        else:
            map_location= 'cpu'
        self.load_state_dict(torch.load(path, map_location=map_location))

    @abc.abstractmethod
    def compute_error(self, Y_pred, Y_true):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_nell(self, Y_pred, Y_true, n, m) -> torch.Tensor:
        raise NotImplementedError

    def string_parameters_to_optimize(self):
        info_para='Parameters to optimize:'
        for name, p in self.named_parameters():
            if p.requires_grad:
                info_para += '\n {:10} {:45} : {:6d} {:.12}'.format("",name,(p.numel()),str(list(p.shape)))
        return info_para
    
    
    def to(self,*args, **kwargs):
        for mod in self.modules():
            if issubclass(type(mod),GaussianMatrix):
                mod.tensors_to(*args, **kwargs)
        try:
            self.tensors_to(*args, **kwargs)
        except AttributeError:
            pass
        return super().to(*args, **kwargs)

import torch
import torch.nn as nn
from . import BaseLayer

class Constant(BaseLayer):
    def __init__(self, out_features,prior=None, values = 0,parameter=True,**kwargs):
        super(Constant, self).__init__(**kwargs)
        self.out_features = out_features
        if parameter:
            self.values = nn.Parameter(values*torch.ones(out_features))
        else:
            self.values = (values*torch.ones(out_features)).detach()
        self.prior = prior
    def forward(self,input):
        return self.values
    def kl_divergence(self):
        if self.prior is None:
            return 0
        else:
            NotImplementedError("TODO, but one really want to put a prior on the prior constant mean?")
    def optimize(self, train: bool = True): # put in parent class?
        self.values.requires_grad = train


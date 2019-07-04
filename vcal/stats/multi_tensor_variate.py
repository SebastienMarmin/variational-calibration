import torch
from .distribution import Distribution
from . import constraints

class MultiTensorVariate(Distribution):
    has_rsample = True
    has_lrsample = True
    def __init__(self, ls):# ls is a modul list
        super(MultiTensorVariate, self).__init__()
        self.l = len(ls)
        self.module_list = ls

    def expand(self, batch_shape, _instance=None):
        new = list()
        for i in range(self.l):
            new+=[self.ls[i].expand(batch_shape[i],_instance[i])]
        return new

    @property 
    def ls(self):
        return list(self.module_list)

    @property
    def mean(self):
        return [self.ls[i].mean for i in range(self.l)]

    def rsample(self, sample_shape=torch.Size()):
        return [self.ls[i].rsample(sample_shape) for i in range(self.l)]

    def lrsample(self, X):
        return [self.ls[i].lrsample(X[i]) for i in range(self.l)]

    def log_prob(self, value,sep=False):
        if sep: # when each log_prob have a different batch_size, we need the terms of the sum
            return [self.ls[i].log_prob(value[i]) for i in range(self.l)]
        else:
            res = self.ls[0].log_prob(value[0])
            for i in range(self.l-1):
                res += self.ls[i+1].log_prob(value[i+1]) # the tensors are independent
            return res


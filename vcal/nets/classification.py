# Original code by Simone Rossi

import torch
import torch.nn as nn

from . import BaseNet
from vcal.stats import LogisticNormal


class ClassificationNet(BaseNet):

    def __init__(self, architecure: nn.Sequential, dtype: torch.dtype = torch.float32):
        super(ClassificationNet, self).__init__()

        self.dtype = dtype
        self.architecture = architecure
        self.likelihood = LogisticNormal() # todo

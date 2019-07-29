import torch
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import vcal
from vcal.layers import FourierFeaturesGaussianProcess as GP
#from torch.utils import hooks as hooks
import matplotlib
from matplotlib import pyplot as plt

if __name__ == '__main__':
    nmc_train = 5
    nmc_test  = 10
    input_dim = 2
    output_dim = 1
    n = 8
    eta   = GP(input_dim,output_dim,nfeatures=100, nmc_train=nmc_train, nmc_test=nmc_test,full_cov_W = True)

    inp = torch.rand(n,input_dim)
    out = torch.randn(n,output_dim)
    eta.set_to_posterior(inp,out[:,0],0.010**2)

    out_pred = eta(inp.unsqueeze(0).expand(torch.Size([eta.nmc])+inp.size()))
    for i in range(eta.nmc):
        plt.plot(out_pred[i,:,0].detach().numpy(),out[:,0].numpy())
    plt.plot([-2,2],[-2,2])
    plt.title("Everything should be close to identity line")
    plt.show()

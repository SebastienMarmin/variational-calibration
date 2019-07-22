import torch
import timeit
import numpy as np
from torch import matmul, randn

if __name__ == '__main__':
    # test on expand
    
    d1 = 2
    d2 = 1000
    d3 = 3
    print(np.log(2*np.pi))
    A1 = randn(d1,d2)
    A2 = randn(d2,d3)
    B =  randn(1,d3)

    #t1 = timeit.timeit('matmul(A1,A2)', number=10000,setup="from __main__ import matmul, A1,A2")
    
    t1 = timeit.timeit('A2**2', number=100,setup="from __main__ import matmul, A2,B,d2,d3")
    #t1 = timeit.timeit('matmul(A1,B.expand(d2,d3))', number=10000,setup="from __main__ import matmul, A1,B,d2,d3")
    print(t1)
    

    ### Test on extracting row or col covariances
    d1,d2 = 2,3
    batch_shape = []

    scale = torch.randn(d2,d1,d1)
    L_blocks = torch.tril(scale).expand(*batch_shape,d2,d1,d1)
    L = torch.diag_embed(L_blocks.unsqueeze(-1).transpose(-1,-4).squeeze(-4)).transpose(-2,-3).contiguous().view(*batch_shape,d1*d2,d1*d2)
    print("start")

    L_r = torch.diagonal(L.view(d1,d2,d1,d2).transpose(-3,-2),dim1=-2,dim2 = -1).transpose(-1,-3).transpose(-1,-2)
    print(L)
    print(L_blocks)
    print(L_r)


    scale = torch.randn(d1,d2,d2)
    L_blocks = torch.tril(scale).expand(*batch_shape,d1,d2,d2)
    L = torch.diag_embed(L_blocks.unsqueeze(-1).transpose(-1,-4).squeeze(0)).unsqueeze(-5).transpose(-2,-5).squeeze(-2).transpose(-1,-2).contiguous().view(*batch_shape,d1*d2,d1*d2)
    print(L_blocks)
    print(L)
    L_r = torch.diagonal(L.view(d1,d2,d1,d2).transpose(-3,-2),dim1=-3,dim2 = -4).transpose(-1,-3).transpose(-1,-2)
    print(L_r)
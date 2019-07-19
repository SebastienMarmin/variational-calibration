import torch
import timeit
from torch import matmul, randn

if __name__ == '__main__':
    d1 = 2
    d2 = 100000
    d3 = 3
    
    A1 = randn(d1,d2)
    A2 = randn(d2,d3)
    B =  randn(1,d3)

    #t1 = timeit.timeit('matmul(A1,A2)', number=10000,setup="from __main__ import matmul, A1,A2")

    t1 = timeit.timeit('A2**2', number=10000,setup="from __main__ import matmul, A2,B,d2,d3")
    #t1 = timeit.timeit('matmul(A1,B.expand(d2,d3))', number=10000,setup="from __main__ import matmul, A1,B,d2,d3")
    print(t1)
    t1 = timeit.timeit('torch.dot(B.expand(d2,d3),B.expand(d2,d3))', number=10000,setup="from __main__ import matmul, torch, A1,B,d2,d3")
    print(t1)
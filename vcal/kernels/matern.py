import numpy as np
import torch
from . import CovarianceStructure

class Matern(CovarianceStructure):
    def __init__(self,dimension,smoothness=5/2,range=1):
        super(Matern, self).__init__(dimension)
        self.smoothness = smoothness
        self.range = 1
    def correlation(self,X1,X2=None, stability=False):
        nu = self.smoothness
        h2 = dist_matrix_sq(X1,X2,stability)/self.range**2
        if nu < 0 or np.isinf(nu):
            return (-h2/2).exp()
        h  = h2.sqrt()
        if nu == 5/2:
            sqrt5 = np.sqrt(5)
            return (1+sqrt5*h+5/3*h2)*(-sqrt5*h).exp()
        if nu == 3/2:
            sqrt3 = np.sqrt(3)
            return (1+sqrt3*h)*(-sqrt3*h).exp()
        if nu == 1/2:
            return (-h).exp()
        print("Smoothness %d is not handled for analytic correlation (requires to evaluate the modified Bessel function)."%(nu))
    def sample_spectrum(self,n_samples):
        p     = n_samples
        d     = self.dimension
        theta = self.range
        if self.smoothness > 0 and self.smoothness < np.Inf:
            nu    = self.smoothness
            dl    = 2*nu
            lambd = 1/theta

            y = torch.randn(p,d)*lambd             # t(λ*matrix(rnorm(d*p),ncol=p))
            u = torch.Tensor(chi2.rvs(dl, size=p)) # rchisq(p,dl,ncp = 0)
            Omega  = y/(u/dl).sqrt().unsqueeze(1)    # ω <- y/sqrt(u/dl)
        else: # infinite smoothness, i.e. RBF
            V      = 1/theta              # Omega's std
            Omega  = V*torch.randn(d,p)
        return Omega


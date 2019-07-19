import numpy as np
import torch
from torch import nn as nn
from torch import matmul
from torch.distributions import kl_divergence
from . import GaussianProcess
from ..utilities import regress_linear as regress
from vcal.stats import Normal,  GaussianMatrix, CovarianceMatrix
import vcal.stats.kl


class FeaturesGaussianProcess(GaussianProcess):
    def __init__(self,  in_features, out_features, nfeatures=None,full_cov_W = False, **kwargs):
        super(FeaturesGaussianProcess, self).__init__( in_features, out_features, **kwargs)
        if nfeatures is None:
            self.nfeatures = int(np.sqrt(10*self.in_features))
        else:
            self.nfeatures = nfeatures
        self.full_cov_W = full_cov_W
        self.local_reparam = True
        # subclass must define self.W and self.W_prior, instances of matrixGaussian distrib
    def reset_parameters(self):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")

    def activation(self,X):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")

    def set_to_prior(self):
        self.W.loc.data = self.W_prior.loc.expand(self.W.loc.shape).detach().clone()
        self.W.row_cov.detach_clone(self.W_prior.row_cov)
        self.W.col_cov.detach_clone(self.W_prior.col_cov)
         # basically do the same as detach().clone() for replacing W.row_cov.parameter.data,
         # except it handle the flag .diagonal and .homoscedastic consistantly
         # and do proper expand if the bach_size are different.
         # self.W.col_cov.tril = self.W_prior.col_cov.tril.detach().clone()


    def set_to_posterior(self,X,Y,noise_covariance):
        M = self.prior_means(X)
        Yc = Y - M
        Phi = self.activation(X)
        prior_std = self.stddevs
        Lambda = noise_covariance
        i = 0 # TODO D_out>1
        mu = self.W_prior.loc
        Gamma = self.W_prior.covariance_matrix

        meanBeta, covBeta, _ = regress(prior_std*Phi,Yc,Lambda,Gamma,mu) # infere beta in: Yc = Phi beta + eps
        # Lambda: prior in eps; mu and Gamma: prior on beta
        meaRp = self.W.loc.data.clone().detach()
        meaRp[:,0] = meanBeta
        self.W.loc.data = meaRp
        self.W.full_tril = torch.cholesky(covBeta,upper=False).detach()
        



    def forward(self,input):
        nmc = self.nmc
        m = self.prior_means(input)
        Phi = self.activation(input)
        if self.local_reparam:
            F = self.W.lrsample(Phi)
        else:
            F = matmul(Phi,self.W.rsample(torch.Size([nmc])))
        output = m+self.stddevs*F
        return output

    def kl_divergence(self):
         return kl_divergence(self.W, self.W_prior)


class FourierFeaturesGaussianProcess(FeaturesGaussianProcess):
    def __init__(self, in_features, out_features, **kwargs):
        super(FourierFeaturesGaussianProcess, self).__init__(in_features, out_features, **kwargs)
        self.Phi_fun = lambda input: torch.cat((input.sin(),input.cos()),-1)
        self.nfeatures_W = 2*self.nfeatures
        self.sqrt_nfeatures = np.sqrt(self.nfeatures)
        #distrW = 'full_covariance_matrix_gaussian' if self.full_cov_W else 'fully_factorized_matrix_gaussian'
        d1 = self.nfeatures_W
        d2 = self.out_features
        self.W = GaussianMatrix(d1,d2,dependent_rows=True)# full cov
        self.W_prior = GaussianMatrix(d1,d2,same_col_cov=True,same_row_cov=True,centered=True,parameter=False)
        #self.W_prior.optimize(False)
        self.Omega = nn.Parameter(self.cov_structure.sample_spectrum(self.nfeatures),requires_grad=False)
        # if user set Omega.grad=True, it's recommanded to put lengthscales.grad = False.

    def reset_parameters(self):# when change nfeatures, full_cov_W, cov_structure. Lose train values for W and Omega
        W_optim_status = self.W.loc.requires_grad
        self.nfeatures_W = 2*self.nfeatures
        self.sqrt_nfeatures = np.sqrt(self.nfeatures)
        d1 = self.nfeatures_W
        d2 = self.out_features
        self.W = GaussianMatrix(d1,d2,independent_cols=True,independent_rows=False)
        self.W_prior = GaussianMatrix(d1,d2,homoscedastic_cols=True,homoscedastic_rows=True,centered=True)
        self.W_prior.optimize(False)
        self.W.optimize(W_optim_status)
        self.Omega.data = self.cov_structure.sample_spectrum(self.nfeatures)

    def activation(self,X):
        SP = matmul(X/self.lengthscales,self.Omega)
        return self.Phi_fun(SP)/self.sqrt_nfeatures
    
    def fix_hyperparameters(self):
        super(FourierFeaturesGaussianProcess, self).fix_hyperparameters()
        self.Omega.requires_grad = False

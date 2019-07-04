#  Copyright (C) 2019   Sébastien Marmin <marmin@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
from os.path import isfile, join
from os import listdir
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torch import matmul
from collections import OrderedDict
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../vardl"))#sys.path.append(os.path.join(".", "../../"))#
import timeit
import vardl
import humanize
import json
import numpy as np


from vardl.layers import BaseVariationalLayer
from vardl.distributions import available_distributions, kl_divergence
from vardl.distributions import FullyFactorizedMultivariateGaussian



def pfi(X):
        return X.exp()
def pf(X):
        return X.log()

def dist_matrix_sq(self, X1, X2=None, stability=False):
    x1 = X1
    if X2 is None:
        if self.stability:
            mu = torch.mean(x1, 0)
            x1.sub_(mu) #inline; subtracting the mean for stability
        D = -2 * torch.mm(x1, x1.t())
        sum_x1x1 = torch.sum(x1*x1, 1).unsqueeze(1).expand_as(D)
        D.add_(sum_x1x1)
        D.add_(sum_x1x1.t())
    else:
        x2 = X2
        if self.stability:
            n = x1.shape[0]
            m = x2.shape[0]
            mu = (m/(n+m))*torch.mean(x2, 0) + (n/(n+m))*torch.mean(x1, 0)
            x1.sub_(mu) #inline; subtracting the mean for stability
            x2.sub_(mu) #inline; subtracting the mean for stability
        D = -2 * torch.mm(x1, x2.t())
        sum_x1x1 = torch.sum(x1*x1, 1).unsqueeze(1).expand_as(D)
        sum_x2x2 = torch.sum(x2*x2, 1).unsqueeze(0).expand_as(D)
        D.add_(sum_x1x1)
        D.add_(sum_x2x2)
    return D



import abc

import torch.nn as nn


class CovarianceStructure(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        super(CovarianceStructure, self).__init__()
        NotImplementedError("Subclass of CovarianceStructure should implement __init__().")
    @abc.abstractmethod
    def correlation(self,X1,X2=None):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def sample_spectrum(self,n_samples):
        NotImplementedError("Subclass of CovarianceStructure should implement sample_spectrum().")




class FourierGaussianProcess(BaseVariationalLayer, metaclass=abc.ABCMeta):
    def __init__(self, in_features, out_features, **kwargs):
        super(GaussianProcess, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = lambda input: 0 # TODO can also be any type of layer, e.g. learnable function 
        self.lengthscale = nn.Parameter(pf(torch.ones(in_features)))
        # can be size 1 or size in_features (or TODO? in_features times out_features)
        self.variance = nn.Parameter(pf(torch.ones(out_features))) # can be size 1 or size out_features
        self.nfeatures = int(np.sqrt(10*in_features))
        self.full_cov_W = False
        self.cov_structure = Matern(smoothness=-1)
        self.local_reparam = True
        self.reset_parameters()
    #@property
    #def W(self):
    #    raise NotImplementedError("Subclass of CovarianceStructure should implement self.W layer instance.")
    @abc.abstractmethod
    def reset_parameters(X):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def set_to_prior():
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def set_to_posterior(X,Y,noise_distribution=None):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def features_forward(X):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def kl_divergence(self):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    def forward(self, input): # nmc times n times in_features
        nmc = input.shape[0]
        m = self.prior_mean(input)
        Phi = self.features_forward(input) 
        if self.local_reparam:
            F = self.W.sample_local_reparam_linear(self.nmc,Phi)
        else:
            F = matmul(Phi,self.W.sample(self.nmc))
        output = pfi(self.variance)*F
        return output
    def optimize(self, train: bool = True):
        for param in self.parameters():
            param.requires_grad = train

class FourierFeaturesGaussianProcess(GaussianProcess):
    def __init__(self, nfeatures, **kwargs):
        super(FeaturesGaussianProcess, self).__init__(**kwargs)

    def reset_parameters():
        self.nfeatures_W = 2*self.nfeatures
        self.Phi_fun = lambda input: torch.cat((input.sin(),input.cos()),-1)
        distrW = 'full_covariance_matrix_gaussian' if self.full_cov_W else 'fully_factorized_matrix_gaussian'
        self.W =  available_distributions[distrW](self.nfeatures_W, self.out_features, self.dtype)
        self.W.logvars.data = torch.zeros_like(self.W.logvars)
        self.Omega = nn.Parameter(self.cov_structure.sample_spectrum(self.nfeatures))


    def features_forward(X):
        SP = matmul((X*pfi(self.lengthscales)),self.Omega)
        return self.Phi_fun(SP)







    def predict(self, Xtest):
        if not self.is_trained():
            self.forward()
        return_np = not (torch.is_tensor(Xtest) or isinstance(Xtest,Variable))
        Xtest = construct_variable(Xtest)
        # Adding singleton dimensions for consistency
        # dim0: number of points;  dim1: input dimension
        if self.X.dim() == 1:
            Xtest.unsqueeze_(1)
        fmu, fs2 = self._predict_f(Xtest)
        fmu += self.mu_function
        if fmu.ndimension() > 1 and fs2.ndimension() == 1:
            fs2 = fs2.unsqueeze(1).repeat(1, fmu.ndimension())
        ymu, ys2 = self._predict_y(fmu, fs2)
        # return numpy objects, for simplicity
        if return_np:
            ymu = ymu.data.numpy()
            ys2 = ys2.data.numpy()
            fmu = fmu.data.numpy()
            fs2 = fs2.data.numpy()
        return ymu, ys2, fmu, fs2

    def __repr__(self):
        string = 'Model: ' + type(self).__name__ + '\n'
        for name, param in self.named_parameters():
            if param.requires_grad:
                data = param.data.numpy()
                string += name + ': ' + str(data) + '\n'
        return string



class SVGPR(GPModule):
    """
    Sparse Variational GP regression (Titsias 2009)
    Implementation loosely based on gpflow.SGPR
    """
    def __init__(self, X, Y, Z, kernel, sn2=0.01):
        super(SVGPR, self).__init__(X, Y, sn2)
        self.kernel = kernel
        self.Z = nn.Parameter(construct_tensor(Z), False)
        self.U = None
        self.UB = None
        self.c = None

    def is_trained(self):
        return not (self.U is None or self.UB is None or self.c is None)

    def _predict_y(self, fmu, fs2):
        ymu = fmu
        ys2 = fs2 + torch.mean(self.get_sn2())
        return ymu, ys2

    def _predict_f(self, Xtest):
        Kus = self.kernel(self.Z, Xtest)
        Kss = self.kernel(Xtest, diag=True)
        tmp1 = autopotrs.Trtrs_T.apply(Kus, self.U)
        tmp2 = autopotrs.Trtrs_T.apply(tmp1, self.UB)
        fmu = torch.mm(tmp2.t(), self.c).squeeze()
        fs2 = Kss + torch.sum(tmp2**2, 0) - torch.sum(tmp1**2, 0)
        return fmu, fs2


    def forward(self):
        n = self.Y.shape[0]
        dim = self.Y.shape[1]      
        n_inducing = self.Z.shape[0]
        Kdiag = self.kernel(self.X, diag=True)
        Kuu = self.kernel(self.Z) + Variable(torch.eye(n_inducing) * 1e-6)

        sn2_vec = self.get_sn2()
        sn2_vec = sn2_vec.repeat(n) if sn2_vec.numel() == 1 else sn2_vec
        sn_vec = torch.sqrt(sn2_vec)
        Kuf = self.kernel(self.Z, self.X)

        self.U = torch.potrf(Kuu)
        A = autopotrs.Trtrs_T.apply(Kuf, self.U) / sn_vec.view_as(self.Y)
        ## For the record: A'*A = Kuf' * inv(Kuu) * Kuf = Qff
        AAT = torch.mm(A, A.t())
        Ay = torch.mm(A, self.Y / sn_vec.view_as(self.Y))
        B = AAT + Variable(torch.eye(n_inducing) * 1e-6)
        self.UB = torch.potrf(B)
        self.c = autopotrs.Trtrs_T.apply(Ay, self.UB)

        # compute log marginal bound
        bound = -0.5 * n * dim * math.log(2 * math.pi)
        bound += -dim * self.UB.diag().log().sum()
        bound -= 0.5 * dim * torch.log(sn2_vec).sum()
        bound += -0.5 * torch.sum(self.Y**2 / sn2_vec.view_as(self.Y))
        bound += 0.5 * torch.sum(self.c**2)
        bound += -0.5 * dim * torch.sum(Kdiag / sn2_vec)
        bound += 0.5 * dim * AAT.diag().sum()
        print(bound)
        return bound




class GPRFITC(GPModule):
    '''GPRFITC
    Sparse GP Regression with the FITC method
    '''
    def __init__(self, X, Y, Z, kernel, sn2=0.01):
        super(GPRFITC, self).__init__(X, Y, sn2)
        self.kernel = kernel
        self.Z = nn.Parameter(construct_tensor(Z), False)
        self.Uuu = None
        self.U = None
        self.gamma = None

    def _predict_y(self, fmu, fs2):
        ymu = fmu
        ys2 = fs2 + torch.mean(self.get_sn2())
        return ymu, ys2

    def is_trained(self):
        return not (self.U is None or self.Uuu is None or self.gamma is None)

    def _predict_f(self, Xtest):
        Kus = self.kernel(self.Z, Xtest)
        Kss = self.kernel(Xtest, diag=True)
        w = autopotrs.Trtrs_T.apply(Kus, self.Uuu)
        tmp = autopotrs.Trtrs.apply(self.gamma, self.U)
        intermediateA = autopotrs.Trtrs_T.apply(w, self.U)
        fmu = torch.mm(w.t(), tmp).squeeze()
        fs2 = Kss - torch.sum(w**2, 0) + torch.sum(intermediateA**2, 0)
        return fmu, fs2

    def forward(self):
        n = self.Y.shape[0]
        d = self.Y.shape[1]
        n_inducing = self.Z.shape[0]

        sn2_vec = self.get_sn2()
        sn2_vec = sn2_vec.repeat(n) if sn2_vec.numel() == 1 else sn2_vec
        sn_vec = torch.sqrt(sn2_vec)

        Kdiag = self.kernel(self.X, diag=True)
        Kuu = self.kernel(self.Z) + Variable(torch.eye(n_inducing) * 1e-6)
        Kuf = self.kernel(self.Z, self.X)

        ## Qff = Kfu * inv(Kuu) * Kuf
        ## diag(Qff) == sum(Kuf.*invKuu_Kuf,1)

        self.Uuu = torch.potrf(Kuu)
        V = autopotrs.Trtrs_T.apply(Kuf, self.Uuu)
        diagQff = torch.sum(V**2, 0)
        nu = Kdiag - diagQff + sn2_vec
        B = Variable(torch.eye(n_inducing)) + torch.mm(V / nu, V.t())
        self.U = torch.potrf(B)
        beta = self.Y / nu.unsqueeze(1)
        alpha = torch.mm(V, beta)
        self.gamma = autopotrs.Trtrs_T.apply(alpha, self.U)

        mahalanobisTerm = -0.5 * torch.sum(self.Y**2 / nu.unsqueeze(1)) \
                          + 0.5 * torch.sum(self.gamma**2)
        constantTerm = -0.5 * n * math.log(2.*math.pi)
        logDeterminantTerm = -0.5 * nu.log().sum() - self.U.diag().log().sum()
        logNormalizingTerm = constantTerm + logDeterminantTerm
        return mahalanobisTerm + logNormalizingTerm * d




class GPR(GPModule):
    '''GPR
    Gaussian Process Regression

    Technical details:
        It uses autopotrs.PosdefSolveInv which involves 
        1 call of potrf and 1 call of potri.
        Backward computation only has matrix-to-vector multiplications.
    '''

    def __init__(self, X, Y, kernel, sn2=0.01):
        super(GPR, self).__init__(X, Y, sn2)
        self.kernel = kernel
        self.invC = None
        self.invC_Y = None

    def is_trained(self):
        return self.invC is not None and self.invC_Y is not None

    def _predict_y(self, fmu, fs2):
        ymu = fmu
        ys2 = fs2 + torch.mean(self.get_sn2())
        ## ***(predict y in classification?)
        return ymu, ys2

    def _predict_f(self, Xtest):
        Kmm = self.kernel(Xtest, diag=True)
        Knm = self.kernel(self.X, Xtest)
        invC_Knm = self.invC.mm(Knm)
        fmu = torch.mm(Knm.t(), self.invC_Y).squeeze()
        fs2 = Kmm - torch.sum(Knm * invC_Knm, 0)
        return fmu, fs2

    def forward(self):
        n = self.Y.shape[0]
        d = self.Y.shape[1]
        C = self.kernel(self.X)
        sn2_vec = self.get_sn2()
        sn2_vec = sn2_vec.repeat(n) if sn2_vec.numel() == 1 else sn2_vec
        C.add_(torch.diag(sn2_vec))
        self.invC_Y, logdet, self.invC = autopotrs.PosdefSolveInv.apply(self.Y,C)
        Y_invC_Y = (self.Y*self.invC_Y).sum()
        ##EQUIVALENTLY: Y_invC_Y=torch.mm(self.Y.t(), self.invC_Y).diag().sum()
        negloglik = d*logdet/2 + 0.5*Y_invC_Y + n*d*math.log(2*math.pi)/2
        return negloglik







class GPRh(GPR):
    '''GPRh
    Heteroskedastic GPR (with different noise term per output dimension)
    
    Note:
        If the noise terms are the same for all output dimensions,
        then standard GPR also handles heteroskedastic noise.
    '''

    def forward(self):
        noise = self.get_sn2()
        noisedim = noise.shape[1] if noise.ndimension() > 1 else 1
        if noisedim == 1:
            return super(GPRh, self).forward()

        n = self.Y.shape[0]
        self.invC = Variable(torch.zeros(n,n, noisedim))
        self.invC_Y = Variable(torch.zeros(n, noisedim))
        K = self.kernel(self.X)
        negloglik = 0
        for i in range(noisedim):
            sn2_vec = noise[:,i]
            sn2_vec = sn2_vec.repeat(n) if sn2_vec.numel() == 1 else sn2_vec
            C = K.add(torch.diag(sn2_vec))
            Y = self.Y[:,[i]]
            invC_Y, logdet, invC = autopotrs.PosdefSolveInv.apply(Y, C)
            #Y_invC_Y = torch.dot(Y, invC_Y)
            Y_invC_Y = (Y*invC_Y).sum()
            ##EQUIVALENTLY: Y_invC_Y=torch.mm(Y.t(), invC_Y).diag().sum()
            negloglik += logdet/2 + 0.5*Y_invC_Y + n*math.log(2*math.pi)/2
            self.invC_Y[:,i] = invC_Y
            self.invC[:,:,i] = invC
        return negloglik

    def _predict_f(self, Xtest):
        noise = self.get_sn2()
        noisedim = noise.shape[1] if noise.ndimension() > 1 else 1
        if noisedim == 1:
            return super(GPRh, self)._predict_f(Xtest)
        Kmm = self.kernel(Xtest, diag=True)
        Knm = self.kernel(self.X, Xtest)
        fmu = torch.mm(Knm.t(), self.invC_Y).squeeze()
        fs2 = Variable(torch.zeros(fmu.shape))
        for i in range(noisedim):
            invC_Knm = self.invC[:,:,i].mm(Knm)
            fs2[:,i] = Kmm - torch.sum(Knm * invC_Knm, 0)
        return fmu, fs2






class DirichletGPC(GPRh):
    '''DirichletGPC
    Dirichlet-based GP Classification
    '''
    def __init__(self, X, Y, kernel, alpha):
        super(DirichletGPC, self).__init__(X, Y, kernel)
        self.logalpha = Hyperparameter(np.log(alpha), False)
        self.Y_onehot = DirichletGPC.onehot_encoding(Y)
        self._update_latent_labels()

    def _update_latent_labels(self):
        alpha = torch.exp(self.logalpha())
        s2_tilde = torch.log(1.0/(self.Y_onehot+alpha) + 1)
        mu_tilde = torch.log(self.Y_onehot+alpha) - 0.5 * s2_tilde
        self.logsn2.param.set_(torch.log(s2_tilde))
        self.Y = mu_tilde
        self.mu_function = torch.mean(self.Y, 0)
        self.Y -= self.mu_function

    def onehot_encoding(Y):
        '''Y must be one of the following:
        - vector of integers
        - one-hot encoded matrix
        '''
        Y = construct_tensor(Y)
        if Y.ndimension() > 1 and Y.shape[1] > 1:
            if Y.numel() != torch.sum((Y==0)+(Y==1)):
                raise ValueError("Y must be one-hot encoded")
            if Y.shape[0] != torch.sum(Y.sum(1)==1):
                raise ValueError("Y must be one-hot encoded")
            return Variable(Y)
        yvec = Y.squeeze().int()
        classes = yvec.max() + 1
        Y = torch.zeros(len(yvec), classes)
        for i in range(len(yvec)):
            Y[i, yvec[i]] = 1
        return Variable(Y)


    def forward(self):
        if self.logalpha.param.requires_grad == True:
            self._update_latent_labels()
        g_nloglik = super(DirichletGPC, self).forward()
        # b_nloglik = self._bernoulli_negloglik()
        return g_nloglik

    def _bernoulli_negloglik(self):
        fmu, fs2 = self._predict_f(self.X)
        fmu += self.mu_function
        # mean of Dirichlet with parameters a_i
        gamma_a = torch.exp(fmu + 0.5 * fs2) + 1e-8
        p = gamma_a / torch.sum(gamma_a, 1).view(-1,1)
        logp = torch.log(p)
        loglik = torch.sum(self.Y_onehot * logp, 1)
        return -torch.sum(loglik)












class FeatureGPR(GPModule):

    def __init__(self, X, Y, phi_func, weight_var=1, sn2=0.01):
        super(FeatureGPR, self).__init__(X, Y, sn2)
        self.phi_func = phi_func
        self.invA = None
        self.invA_Zy = None
        self.logwvar = Hyperparameter(np.log(weight_var))

    def get_weight_var(self):
        return torch.exp(self.logwvar())

    def is_trained(self):
        return self.invA is not None and self.invA_Zy is not None

    def _predict_y(self, fmu, fs2):
        ymu = fmu
        ys2 = fs2 + self.get_sn2()
        return ymu, ys2

    def _predict_f(self, Xtest):
        ztest = self.phi_func(Xtest)
        fmu = torch.mm(ztest, self.invA_Zy).squeeze()
        invA_ztest = self.invA.mm(ztest.t())
        fs2 = torch.sum(ztest.t() * invA_ztest, 0) * self.get_sn2()
        return fmu, fs2
    
    def forward(self):
        N = self.X.shape[0]
        Z = self.phi_func(self.X)
        a2 = self.get_weight_var()
        features = Z.shape[1]
        s2_a2 = self.get_sn2() / a2
        termInA = torch.diag(s2_a2.expand(features)) # eye * (s2 / a2)
        A = torch.mm(Z.t(), Z) + termInA
        Zy = torch.mm(Z.t(), self.Y)
        self.invA_Zy, logdet, self.invA = autopotrs.PosdefSolveInv.apply(Zy, A)
        yty = (self.Y**2).sum()
        ZyAyZ = (Zy*self.invA_Zy).sum()
        term1 = 0.5*(yty-ZyAyZ) / self.get_sn2()
        term3 = features/2*torch.log(s2_a2)
        term4 = N/2*torch.log(2*math.pi*self.get_sn2())
        negloglik = term1 + logdet / 2 - term3 + term4
        return negloglik




class SigmoidFeatures(nn.Module):
    def __init__(self, dim, nfeatures):
        super(SigmoidFeatures, self).__init__()
        sample = torch.randn(nfeatures, dim+1)
        self.layer_weights = nn.Parameter(sample)
        self.eyeplus = Variable(torch.eye(dim, dim+1))

    def forward(self, X):
        X = torch.mm(X, self.eyeplus); X[:,-1] = 1
        wX = torch.mm(X, self.layer_weights.t())
        return torch.tanh(wX)
        #return nn.functional.relu(wX)



def _fourier_feature_func(X, freqs):
    wX = torch.mm(X, freqs.t())
    return torch.cos(wX - math.pi/4)

class FourierFeatures(nn.Module):
    def __init__(self, freqs, optimise_freqs=False):
        super(FourierFeatures, self).__init__()
        self.freqs = Hyperparameter(freqs, optimise_freqs)
        self.nfeatures = freqs.shape[0]
                
    def forward(self, X):
        return _fourier_feature_func(X, self.freqs())


class RFF(FourierFeatures):
    def __init__(self, dim, nfeatures, leng=1, ARD=False):
        super(RFF, self).__init__(torch.zeros(nfeatures, dim))
        sample = torch.randn(nfeatures, dim)
        self.sqrt_2_D = math.sqrt(2 / nfeatures)
        self.logleng = Hyperparameter(np.log(leng))
        self.freqs = Hyperparameter(sample, False)

    def get_lengthscale(self):
        return torch.exp(self.logleng())
    
    def forward(self, X):
        ## multiply by standard deviations: freq*diag(1/L)
        ## Z = cos(w * X' - pi / 4) .* sqrt(2 / D);
        w = self.freqs() / torch.exp(self.logleng())
        return _fourier_feature_func(X, w) * self.sqrt_2_D






def construct_rbf_spectrum (N, periods, leng):
    if np.isscalar(periods):
        leng = [periods]
    if np.isscalar(leng):
        leng = [leng]
    if not torch.is_tensor(periods):
        periods = torch.FloatTensor(periods)
    # if not torch.is_tensor(leng):
    #     leng = torch.FloatTensor(leng)
    dim = len(periods)
    if leng.numel() == 1:
        leng = leng.expand(dim)

    N_per_dimension = np.ceil(np.power(N, 1/dim))
    N_per_dimension += 1-N_per_dimension%2
    freqs_1D = torch.arange(N_per_dimension) - np.floor(N_per_dimension/2)
    freqs = cartesian([freqs_1D]*dim)
    freqs = torch.FloatTensor(freqs)

    w0 = Variable(2*math.pi / periods)
    w = Variable(freqs) * w0
    eigenvalues = torch.exp(-0.5*(w*leng)**2) * leng * w0/2.5066282746310002
    eigenvalues = eigenvalues.prod(1)
    errorBound = 1 - torch.sum(eigenvalues)
    return w, eigenvalues, errorBound


class FSF(FourierFeatures):
    def __init__(self, dim, nfeatures, periods, leng=1, ARD=False):
        super(FSF, self).__init__(torch.zeros(nfeatures, dim))
        self.dim = dim
        self.nfeatures = nfeatures
        self.logleng = Hyperparameter(np.log(leng))
        self.periods = construct_tensor(periods)

    def get_lengthscale(self):
        return torch.exp(self.logleng())

    def forward(self, X):
        N = self.nfeatures
        leng = torch.exp(self.logleng())
        periods = self.periods
        w, eigvals, err = construct_rbf_spectrum (N, periods, leng)
        return _fourier_feature_func(X, w) * torch.sqrt(eigvals)








class RBF(ShiftInvariantKernel):
    def __init__(self, dim, ampl=1, leng=1, ARD=False, stability=False):
        super(RBF, self).__init__(dim, ampl, leng, ARD)
        self.stability = stability
    def forward(self, X1, X2=None, diag=False):
        if diag:
            return torch.exp(self.logamp2())*Variable(torch.ones(X1.shape[0]))
        D = self.dist_matrix_sq(X1, X2)
        K = torch.exp(self.logamp2()) * torch.exp(-0.5 * D)
        return K



class Matern(ShiftInvariantKernel):
    def __init__(self, dim, nu, ampl=1, leng=1, ARD=False, stability=False):
        super(Matern, self).__init__(dim, ampl, leng, ARD)
        self.stability = stability
        self.nu = nu
        if nu != 1 and nu != 3 and nu != 5:
            raise ValueError('Matern kernel: nu can be 1, 3 or 5')
   
    def forward(self, X1, X2=None, diag=False):
        D = self.dist_matrix_sq(X1, X2, diag)
        if X2 is None and not diag: # FIXME: this is ugly
            D.add_(Variable(torch.eye(X1.shape[0]) * 1e-6))
        D.sqrt_()
        K = torch.exp(self.logamp2()) * torch.exp(-D)
        if self.nu == 3:
            K.mul_(1 + D)
        if self.nu == 5:
            K.mul_(1 + D*(1 + D/3))
        return K





class NNKernel(nn.Module):
    '''
    Neural network kernel (Rasmussen & Williams page 91, Eq: 4.29)
    K(x2,x2) = asin(2x1'*x2 / sqrt((2x1'*x1+1) * (2x2'*x2+1)))

    Transfer function: h(x) = u0 + u*x
    leng: the variance of u
    leng0: the variance of u0
    '''
    def __init__(self, dim, ampl=1, leng=1, leng0=1, ARD=False):
        super(NNKernel, self).__init__()
        if np.isscalar(leng):
            leng *= np.ones(dim if ARD else 1)
        leng = np.append(leng,leng0)
        self.eyeplus = Variable(torch.eye(dim, dim+1))            
        self.logamp2 = Hyperparameter(np.log(ampl**2))
        self.logleng = Hyperparameter(np.log(leng))

    def forward(self, X1, X2=None, diag=False):
        ## NNKernel requires that we augment input space
        X1 = torch.mm(X1, self.eyeplus); X1[:,-1] = 1
        x1 = X1 / torch.exp(self.logleng())
        if diag:
            arg = torch.sum(x1*x1, 1) * 2
            arg.div_(arg + 1)
        else:
            if X2 is not None:
                X2 = torch.mm(X2, self.eyeplus); X2[:,-1] = 1
                x2 = X2 / torch.exp(self.logleng())
                arg = torch.mm(x1, x2.t()) * 2
                XXt1 = torch.sum(x1*x1, 1).unsqueeze(1).expand_as(arg) * 2 + 1
                XXt2 = torch.sum(x2*x2, 1).unsqueeze(0).expand_as(arg) * 2 + 1
                arg.div_(torch.sqrt(XXt1 * XXt2))
            else:
                arg = torch.mm(x1, x1.t()) * 2
                XXt1 = torch.sum(x1*x1, 1).unsqueeze(1).expand_as(arg) * 2 + 1
                XXt2 = torch.sum(x1*x1, 1).unsqueeze(0).expand_as(arg) * 2 + 1
                arg.div_(torch.sqrt(XXt1 * XXt2))
        
        K = torch.asin(arg)
        return K * torch.exp(self.logamp2())




class FeatureKernel(nn.Module):
    
    def __init__(self, phi_func, weight_var=1):
        super(FeatureKernel, self).__init__()
        self.phi_func = phi_func
        self.logwvar = Hyperparameter(np.log(weight_var))

    def get_weight_var(self):
        return torch.exp(self.logwvar())

    def forward(self, X1, X2=None, diag=False):
        a2 = self.get_weight_var()
        Z1 = self.phi_func(X1)
        if X2 is None:
            Z2 = Z1
        else:
            Z2 = self.phi_func(X2)
        if diag:
            # sum(A.*B',2) == sum(A'.*B,1) == diag(A*B)
            return torch.sum(Z1 * (a2 * Z2), 1)
        K = torch.mm(Z1, a2 * Z2.t())
        return K



class GpRfNet(vardl.models.BaseBayesianNet):
	def __init__(self, input_dim, output_dim,nmc_train, nmc_test,nlayers,
		         nfeatures,activation_functions,full_cov_W = False,fastfood=False,reshape=False,bias=0,std=1):
		"""
		Parameters
		----------
		input_dim: int
		output_dim: int
		nmc_train: int
		nmc_test: int
		nfeatures: int
		"""
		super(GpRfNet, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.nmc_train = nmc_train
		self.nmc_test = nmc_test
		self.nfeaturesSqrt = float(math.sqrt(nfeatures))
		self.nfeatures = nfeatures
		self.output_dim = output_dim
		sqrt2 = math.sqrt(2)
		initial_lengthscale = math.sqrt(2*input_dim)/2.0
		self.loglengthscales = torch.nn.Parameter(initial_lengthscale*torch.ones(input_dim),requires_grad=True)
		
		defaultDistr = 'fully_factorized_matrix_gaussian'
		self.layers = torch.nn.ModuleList()
		self.Omega = torch.nn.Parameter(sqrt2*torch.randn(input_dim,nfeatures),requires_grad=False)
		self.fastfood = fastfood
		self.reshape = reshape
		if fastfood:
			defaultDistr = 'fully_factorized_multivariate_gaussian'
			distrW = 'full_covariance_multivariate_gaussian' if full_cov_W else 'fully_factorized_multivariate_gaussian'
			if reshape:
				nbComponents = 2*nfeatures*output_dim
				closestPower = int(math.ceil(math.log2(nbComponents)/2))
				d_f_in = d_f_out = 2**closestPower
			else:
				d_f_in = 2*nfeatures
				d_f_out = output_dim
			layerW = vardl.layers.VariationalFastfoodLinear(d_f_in, d_f_out, bias=0,
		                                                           prior=defaultDistr,
		                                                           posterior=distrW,
		                                                        nmc_train=nmc_train,
		                                                        nmc_test=nmc_test)
		else :
			defaultDistr = 'fully_factorized_matrix_gaussian'
			distrW = 'full_covariance_matrix_gaussian' if full_cov_W else 'fully_factorized_matrix_gaussian'
			layerW = vardl.layers.VariationalLinear(2*nfeatures, output_dim, bias=False,
		                                    prior=defaultDistr,
		                                    posterior=distrW,
		                                    nmc_train=nmc_train,
		                                    nmc_test=nmc_test)
		self.layers.add_module("W", layerW)

		self.likelihood = vardl.likelihoods.Gaussian()

		self.activation_function = None
		self.basis_functions = None
		self.bias = torch.nn.Parameter(bias*torch.ones(1),requires_grad=False)
		self.std = torch.nn.Parameter(std*torch.ones(1),requires_grad=False)
		self.name = 'GP with RF'
		self.train()

	def forward(self, input,reparam=True):
		lsLayers = list(self.layers)
		WLayer = lsLayers[0]
		#print(input.size())
		x = input / self.loglengthscales.exp()#.to(input.device)# * torch.ones(1, *input.size()).to(input.device)
		#print(x.size())
		SP = torch.matmul(x,self.Omega)
		Phi = torch.cat((torch.sin(SP),torch.cos(SP)),-1)
		#print(Phi.size())
		#print((Phi/self.nfeaturesSqrt * torch.ones(WLayer.nmc, *Phi.size())).size())
		if self.reshape and self.fastfood:
			input_for_W = Phi/self.nfeaturesSqrt * torch.ones(WLayer.nmc, *Phi.size(),device = input.device)
			Fnorm = fastfood_reshape_forward(input_for_W,WLayer,2*self.nfeatures,self.output_dim)
		else:
			Fnorm = WLayer(Phi/self.nfeaturesSqrt * torch.ones(WLayer.nmc, *Phi.size()),device = input.device)
		F = self.bias + self.std * Fnorm
		self.basis_functions = F
		#print(F.size())
		return F


def fastfood_reshape_forward(input_for_W,ff_layer,din,dout):
	nmc = input_for_W.shape[0]
        #if ff_layer.local_reparameterization:
	#	output = ff_layer.posterior_weights.sample_local_reparam_linear(nmc, input)
	#else:
	d = ff_layer.in_features
	if (d**2!=din*dout):
		print("Dim error")
	W = ff_layer(torch.eye(d,device = input_for_W.device).unsqueeze(0).expand(nmc,d,d))
	W_reshaped = W.view(-1,din,dout)
	return torch.matmul(input_for_W,W_reshaped)

#        self.layers = torch.nn.ModuleList()
#        if Om_variational: 
#            layerOm = vardl.layers.VariationalLinear(input_dim, nfeatures, bias=False,
#                                            prior=defaultDistr,
#                                            posterior=defaultDistr,
#                                            nmc_train=nmc_train, nmc_test=nmc_test)
#            self.layers.add_module('Om', layerOm)

#    def forward(self, input,reparam=True):
#        lsLayers = list(self.layers)
#        if self.Om_variational:
#            OmegaLayer = lsLayers[0]
#            WLayer = lsLayers[1]
#            x = input * self.loglengthscales.exp().to(input.device) * torch.ones(OmegaLayer.nmc, *input.size()).to(input.device)
#            SP = OmegaLayer(x)

    
class FastfoodNet(vardl.models.BaseBayesianNet):
    def __init__(self, input_dim, output_dim, nmc_train, nmc_test, nlayers, nfeatures, activation_function):
        """

        Parameters
        ----------
        input_dim: int
        output_dim: int
        nmc_train: int
        nmc_test: int
        nlayers: int
        nfeatures: int
        activation_function
        """
        super(FastfoodNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.nlayers = nlayers
        self.nfearures = nfeatures
        self.activation_function = activation_function

        distr = 'fully_factorized_multivariate_gaussian'
        
        self.layers = torch.nn.ModuleList()
        if nlayers == 0:
            self.layers.add_module('fc', vardl.layers.VariationalLinear(input_dim, output_dim, bias=False,
                                                                        prior='fully_factorized_matrix_gaussian',
                                                                        posterior='fully_factorized_matrix_gaussian',
                                                                        nmc_train=nmc_train, nmc_test=nmc_test))
        else:
            for i in range(nlayers):
                if i == 0:
                    # First layer
                    name = 'vff0'
                    layer = vardl.layers.VariationalFastfoodLinear(input_dim, nfeatures, bias=0,
                                                                   prior=distr,
                                                                   posterior=distr,
                                                                nmc_train=nmc_train,
                                                                nmc_test=nmc_test)

                elif i == nlayers - 1:
                    # Last layer
                    name = 'fc'
#                    layer = vardl.layers.VariationalFastfoodLinear(nfeatures, output_dim, bias=False,
#                                                                   prior=distr,
#                                                                   posterior=distr,
#                                                                   nmc_train=nmc_train,
#                                                                   nmc_test=nmc_test)
                    layer = vardl.layers.VariationalLinear(nfeatures, output_dim, bias=False, nmc_train=nmc_train,
                                                           prior='fully_factorized_matrix_gaussian',
                                                           posterior='fully_factorized_matrix_gaussian',
                                                        nmc_test=nmc_test)
                else:
                    # Everything else in the middle
                    name = 'vff%d' % i
                    layer = vardl.layers.VariationalFastfoodLinear(nfeatures, nfeatures, bias=False,
                                                                   prior=distr,
                                                                   posterior=distr,
                                                                nmc_train=nmc_train,
                                                                nmc_test=nmc_test)

                self.layers.add_module(name, layer)

        self.likelihood = vardl.likelihoods.Gaussian()

        self.activation_function = activation_function
        self.basis_functions = None
        self.name = 'BayesianFastfood'
        self.train()

    def forward(self, input):
        x = input * torch.ones(list(self.layers)[0].nmc, *input.size()).to(input.device)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation_function(x)
            else:
                self.basis_functions = x
        return x
class BayesianVanillaNet(vardl.models.BaseBayesianNet):
    def __init__(self, input_dim, output_dim, nmc_train, nmc_test, nlayers, nfearures, activation_function):
        """

        Parameters
        ----------
        input_dim: int
        output_dim: int
        nmc_train: int
        nmc_test: int
        nlayers: int
        nfearures: int
        activation_function
        """
        super(BayesianVanillaNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.nlayers = nlayers
        self.nfearures = nfearures
        self.activation_function = activation_function

        self.layers = torch.nn.ModuleList()
        
        distr = 'fully_factorized_matrix_gaussian'
        
        if nlayers == 0:
            self.layers.add_module('fc', vardl.layers.VariationalLinear(input_dim, output_dim, bias=False,
                                                                        prior=distr, posterior=distr, nmc_train=nmc_train, nmc_test=nmc_test))
        for i in range(nlayers):
            if i == 0:
               # First layer
                name = 'bll0'
                layer = vardl.layers.VariationalLinear(input_dim, nfearures, bias=False,
                                                       prior=distr, posterior=distr, nmc_train=nmc_train, nmc_test=nmc_test)
            elif i == nlayers - 1:
               # Last layer
                name = 'fc'
                layer = vardl.layers.VariationalLinear(nfearures, output_dim, bias=False,
                                                       prior=distr, posterior=distr, nmc_train=nmc_train, nmc_test=nmc_test)
            else:
               # Everything else in the middle
                name = 'bll%d' % i
                layer = vardl.layers.VariationalLinear(nfearures, nfearures, bias=False,
                                                    prior=distr, posterior=distr, nmc_train=nmc_train, nmc_test=nmc_test)

            self.layers.add_module(name, layer)

        self.likelihood = vardl.likelihoods.Gaussian()

        self.activation_function = activation_function
        self.basis_functions = None
        self.name = 'g-svi'
        self.train()

    def forward(self, input):
        x = input * torch.ones(list(self.layers)[0].nmc, *input.size()).to(input.device)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation_function(x)
            else:
                self.basis_functions = x
        return x

def parse_args():

    available_models = models.keys()
    available_activation_functions = ['tanh', 'erf', 'relu']
    available_datasets = ['yacht', 'boston', 'concrete', 'energy', 'kin8nm', 'naval', 'powerplant', 'protein', "simple1"]


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/workspace/research/datasets.gitlab/export',
                        help='Dataset directory')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                        help='Train/test split ratio')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbosity of training steps')
    parser.add_argument('--nmc_train', type=int, default=1,
                        help='Number of Monte Carlo samples during training')
    parser.add_argument('--nmc_test', type=int, default=256,
                        help='Number of Monte Carlo samples during testing')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size during training')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('--nfeatures', type=int, default=16,
                        help='Dimensionality of hidden layers',)
    parser.add_argument('--activation_function', choices=available_activation_functions, type=str, default='tanh',
                        help='Activation functions',)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for training', )
    parser.add_argument('--model', choices=available_models, type=str, required=True,
                        help='Type of Bayesian model')
    parser.add_argument('--outdir', type=str,
                        default='workspace/',
                        help='Output directory base path',)
    parser.add_argument('--seed', type=int, default=2018,
                        help='Random seed',)
    parser.add_argument('--iterations_fixed_noise', type=int, default=500000,
                        help='Training iteration without noise optimization')
    parser.add_argument('--iterations_free_noise', type=int, default=500000,
                        help='Training iteration with noise optimization')
    parser.add_argument('--test_interval', type=int, default=500,
                        help='Interval between testing')
    parser.add_argument('--time_budget', type=int, default=720,
                        help='Time budget in minutes')
    parser.add_argument('--cuda', action='store_true',
                        help='Training on gpu or cpu')
    parser.add_argument('--save_model', action='store_true',
                        help='Save resulting model')
    parser.add_argument('--full_cov_W', type=int,default=0,
                        help='Switch from fully factorized to full cov for q(W)')
    parser.add_argument('--fastfood', type=int,default=0,
                        help='Switch to Fastfood for for q(W)')
    parser.add_argument('--reshape', type=int,default=0,
                        help='Reshape fastfood output to fit W')

    args = parser.parse_args()

    args.dataset_dir = os.path.abspath(args.dataset_dir)+'/'
    args.outdir = os.path.abspath(args.outdir)+'/'

    if args.cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    return args
def setup_dataset():
    dataset_unidir = join(args.dataset_dir, args.dataset,"pytorch")
    onlyfiles = [f for f in listdir(dataset_unidir) if isfile(join(dataset_unidir, f))]
    if len(onlyfiles)==1:
        dataset_path = join(dataset_unidir,args.dataset+'.pth')
        dataset = TensorDataset(*torch.load(dataset_path))

        input_dim = dataset.tensors[0].size(1)
        output_dim = dataset.tensors[1].size(1)
        size = len(dataset)
        train_size = int(args.split_ratio * size)
        test_size = size - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    elif len(onlyfiles)==2:
        train_dataset_path = join(dataset_unidir,'train_' + args.dataset + '.pth')
        test_dataset_path = join(dataset_unidir,'test_' + args.dataset + '.pth')
        train_dataset = TensorDataset(*torch.load(train_dataset_path))
        test_dataset = TensorDataset(*torch.load(test_dataset_path))
        input_dim = train_dataset.tensors[0].size(1)
        output_dim = train_dataset.tensors[1].size(1)
    logger.info('Loading dataset from %s' % str(dataset_unidir))
        
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,# * torch.cuda.device_count(),
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)
    return train_dataloader, test_dataloader, input_dim, output_dim

def give_alpha(WLayer):
	WLayer.posterior_weights.logvars.data = WLayer.prior_weights.logvars.clone().detach().data
	d = WLayer.in_features
	W = WLayer(torch.eye(d,device=WLayer.S.device).unsqueeze(0).expand(50,d,d))
	return (1/(d**2*50)*(W**2).sum()).item()

def calibrate_prior(WLayer,mini,maxi,step):
        WLayer.posterior_weights.logvars.data = WLayer.prior_weights.logvars.clone().detach().data
        alpha = give_alpha(WLayer)
        d = WLayer.in_features
        while alpha < mini or alpha > maxi:
        	if alpha < mini:
        		WLayer.prior_weights.logvars.data = (step*WLayer.prior_weights.logvars.data.clone().detach().exp()).log()
        	else:
        		WLayer.prior_weights.logvars.data = (1/step*WLayer.prior_weights.logvars.data.clone().detach().exp()).log()
        	WLayer.posterior_weights.logvars.data = WLayer.prior_weights.logvars.clone().detach().data
        	alpha = give_alpha(WLayer)



if __name__ == '__main__':
    args = parse_args()
    #args = parse_args()
    outdir = vardl.utils.next_path('%s/%s/%s/' % (args.outdir, args.dataset, args.model) + 'run-%04d/')
    try:  
        os.makedirs(outdir)
    except OSError:  
        print ("Creation of the directory %s failed" % outdir)
    else:  
        print ("Successfully created the directory %s " % outdir)
    if args.verbose:
            logger = vardl.utils.setup_logger('vardl', outdir, 'DEBUG')
    else:
            logger = vardl.utils.setup_logger('vardl', outdir)
    logger.info('Configuration:')
    for key, value in vars(args).items():
            logger.info('  %s = %s' % (key, value))

    # Save experiment configuration as yaml file in logdir
    with open(outdir + 'experiment_config.json', 'w') as fp:
            json.dump(vars(args), fp, sort_keys=True, indent=4)
    vardl.utils.set_seed(args.seed)
    train_dataloader, test_dataloader, input_dim, output_dim = setup_dataset()
    radialRange = 0.1                       # lengthscale
    V = math.sqrt(2)/radialRange            # Omega's std
    sigma = train_dataloader.dataset.tensors[1].var(0).sqrt()   # W'std (sqrt(Nrf) taken into account in the model)
    bias = train_dataloader.dataset.tensors[1].mean(0)
    logger.info("Bias: "+str(bias.mean().item())+". Std: "+str(sigma.mean().item()))
    model = models[args.model](input_dim, output_dim, args.nmc_train, args.nmc_test, args.nlayers, args.nfeatures,
                                   activation_functions[args.activation_function],args.full_cov_W==1,args.fastfood==1,args.reshape==1,bias=bias,std=sigma)
    model.likelihood.log_noise_var.data = (((0.01*sigma)**2)*torch.ones_like(model.likelihood.log_noise_var)).log().to(model.likelihood.log_noise_var.device)
    
    logger.info("Trainable parameters: %d" % model.trainable_parameters)
    
    tb_logger = vardl.logger.TensorboardLogger(path=outdir, model=model, directory=None)

    trainer = vardl.trainer.TrainerRegressor(model, 'SGD', {'lr': args.lr}, train_dataloader, test_dataloader, args.device, 
                                                 args.seed, tb_logger, debug=False)
    if args.model == 'bnn':
        init = vardl.initializer.IBLMInitializer(model, train_dataloader, args.device, 1)
    #        init.initialize()
    if args.model == 'gprf':
        WLayer = list(model.layers)[0]
        model.loglengthscales.data = (radialRange*torch.ones_like(model.loglengthscales)).log().to(model.loglengthscales.device)
        model.loglengthscales.requires_grad=False
        model.Omega.requires_grad=False
        if args.fastfood==1:
            d_ff = WLayer.in_features
            bidist = torch.distributions.binomial.Binomial(1,probs=.5*torch.ones(d_ff))
            gamdist = torch.distributions.gamma.Gamma(d_ff/2*torch.ones(d_ff), 2*torch.ones(d_ff))
            SnoNorm = 2*gamdist.sample().sqrt()
            B = bidist.sample()*2-1
            G = torch.randn(d_ff)
            S = SnoNorm/math.sqrt((G**2).sum().item())	
            WLayer.B.data = WLayer.B.data.clone().detach()*B.to(WLayer.B.device)
            WLayer.S.data = WLayer.B.data.clone().detach()*S.to(WLayer.S.device)
            WLayer.S.requires_grad=True
            WLayer.B.requires_grad=True
            if args.reshape==1:
                # calibrate prior
                calibrate_prior(WLayer,0.98,1.02,1.01)
                Wconf = WLayer(torch.eye(WLayer.in_features,device=WLayer.S.device).unsqueeze(0).expand(100,d_ff,d_ff))#1/math.sqrt(alpha)*W.detach()#
                logger.info("Pointwise variances of W calibrated to: " + str((1/(d_ff**2*100)*(Wconf**2).sum()).item()))
            else:
                1+1# TODO program a sensible prior ?
        else:
            WLayer.prior_weights.logvars.data = torch.zeros_like(WLayer.prior_weights.logvars.data).to(WLayer.prior_weights.logvars.device)
            WLayer.posterior_weights.logvars.data = WLayer.prior_weights.logvars.clone().detach().data
        logger.info("Trainable parameters: %d" % model.trainable_parameters)

        #list(model.layers)[0].posterior_weights.cov_lower_triangular.requires_grad = False
        #list(model.layers)[1].posterior_weights.logvars.requires_grad = True # fixed covariance
        #list(model.layers)[1].posterior_weights.mean.requires_grad = True
        #list(model.layers)[1].posterior_weights.cov_lower_triangular.requires_grad = True
         #self.logvars = nn.Parameter(
         #   torch.ones(self.n, self.m, dtype=dtype) * np.log(2. / (self.n + self.m)),
         #   requires_grad=False)


    model.likelihood.log_noise_var.requires_grad = False
    trainer.fit(args.iterations_fixed_noise, args.test_interval, 1, time_budget=args.time_budget//2)

    model.likelihood.log_noise_var.requires_grad = True
    trainer.fit(args.iterations_free_noise, args.test_interval, 1, time_budget=args.time_budget//2)


    # Save results
    logger.info("Testing and saving results...")
    test_mnll, test_error = trainer.test()

    import timeit
    import numpy as np
    #trainer.test_verbose = False
    #t = timeit.Timer("trainer.test()", setup="from __main__ import trainer")
    #times = np.array(t.repeat(100, 1)) * 1000.
    #logger.info("Inference time on entire test set (98 percentile): %.4f ms" % (np.percentile(times, 0.98)))
    results = {}


    logger.info('Starting benchmarking...')
    trainer.test_verbose = False
    t = timeit.Timer("trainer.test()", setup="from __main__ import trainer")
    times = np.array(t.repeat(100, 1)) * 1000.
    logger.info("Inference time on entire test set (90 percentile): %.4f ms" % (np.percentile(times, 0.90)))
    results['inference_times'] = times.tolist()
    logger.info('Benchmarking done')

    #results['model'] = args.model
    #results['dataset'] = args.dataset
    #results['nlayers'] = args.nlayers
    #results['nfeatures'] = args.nfeatures

    for key, value in vars(args).items():
        results[key] = value
    results['trainable_parameters'] = model.trainable_parameters
    results['test_mnll'] = float(test_mnll.item())
    results['test_error'] = float(test_error.item())
    results['total_iters'] = trainer.current_iteration
    
    with open(outdir + 'results.json', 'w') as fp:
        json.dump(results, fp, sort_keys=True, indent=4)
    if args.save_model:
        model.save_model(join(outdir,"model.vdl"))

    
#class arguments:
#    def __init__(self):
#        self.dataset_dir = "/home/sebastien/Jeux de données/"
#        self.dataset = "simple1"#"boston"
#        self.split_ratio = .2
#        self.batch_size = 100
#        self.outdir = "/home/sebastien/Inférence variationnelle/vardl/experiments/fastfood-gp-exp/results"
#        self.model = "gprf"#"fastfood"
#        self.verbose = True
#        self.seed = 0
#        self.nmc_train = 30
#        self.nmc_test = self.nmc_train
#        self.nlayers = 1
#        self.nfeatures = 20
#        self.activation_function = "tanh"
#        self.lr = 5e-4
#        self.device = "cpu" # "cuda"
#        self.iterations_fixed_noise = 100#500000
#        self.iterations_free_noise = 1000#500000
#        self.test_interval = 100
#        self.time_budget = 720
#args = arguments()

# --dataset simple1 --dataset_dir /home/sebastien/Jeux\ de\ données/ --nmc_train 10 --lr 5e-3 --iterations_fixed_noise 1000 --iterations_free_noise 1000 --cuda 0 --model gprf



#inverse delta range et delta std .1 <-> .5
#range .5 and std to 1
#fix delta rerun 2
# free rdl rais lr -2 <- -3 rerun 2
# change range to sqrt rerun all



#%%
import time

import os
import pickle # to save synthetic_data_prior object
import numpy as np
import torch
print("Torch version: "+torch.__version__)
from torch.autograd import Variable
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    print("GPU usage.")
else:
    dtype = torch.FloatTensor
    print("CPU usage.")

    ## Copyright 2018 Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.


#  This class implements a Deep Gaussian process model approximated using random features and infered using mini-batch
#  stochastic variational inference. The model is constructed as a torch.nn.Module.

import os.path # only for saveStates
import pickle # only for saveStates
import numpy as np
import torch

from torch.autograd import Variable

import core.utilities as utilities

import core.likelihoods  as likelihoods


class nn_layer(torch.nn.Module):
        def __init__(self, Din, Dout, NRFs, kernel, kernel_ard, factorized, learn_Omega, learn_hyperparam, add_mean, 
                 local_reparameterization, dtype,seedForOmega=None):
            super(nn_layer, self).__init__()
            self.Din = Din
            self.Dout = Dout
            self.NRFs = NRFs
            self.kernel = kernel
            self.kernel_ard = kernel_ard
            self.factorized = factorized
            self.learn_Omega = learn_Omega
            self.learn_hyperparam = learn_hyperparam
            self.add_mean = add_mean
            self.local_reparameterization = local_reparameterization
            self.dtype = dtype
            if self.learn_hyperparam == "optim":
                    self.log_sigma2 = torch.nn.Parameter((0.02**2*torch.ones(1, 1)).log().type(self.dtype), requires_grad=True)
                    if not kernel_ard:
                            self.log_lengthscale = torch.nn.Parameter((0.27*torch.ones(1, 1)).log().type(self.dtype),requires_grad=True)
                    if kernel_ard:
                        self.log_lengthscale = torch.nn.Parameter(
                            torch.ones(self.Din,1).type(self.dtype) * 0.5 * np.log((self.Din) * 1.0) - np.log(2.0),
                            requires_grad=True)  # Initialize lengthscale to sqrt(D) / 2
                    # If random features are fixed from the prior we fix the randomness - the lengthscale
                    # will be learnt using the reparameterization trick
            if self.learn_Omega == "prior_fixed":
                        if seedForOmega is not None:
                            torch.manual_seed(seedForOmega)
                        self.epsilon_for_Omega_sample = Variable(torch.randn(self.Din, self.NRFs).type(self.dtype),
                                                                 requires_grad=False)
                        # TODO : If random features are learned variationally, we learn the approximate
                    # posterior q(Omega) fixing the randomness 
                    #if self.learn_Omega == "var_fixed":
                    #    self.epsilon_for_Omega_sample = Variable(torch.randn(1, self.Din, self.NRFs).type(self.dtype),
                    #                                             requires_grad=False)

                    #    self.q_Omega_m = torch.nn.Parameter(torch.zeros(self.Din, self.NRFs).type(self.dtype),
                    #                                        requires_grad=True)
                    #    self.q_Omega_logv = torch.nn.Parameter(
                    #        torch.ones(self.Din, self.NRFs).type(self.dtype) * -2.0 * self.log_theta_lengthscale.data,
                    #        requires_grad=True)  # We initialize q(Omega) as p(Omega)

                # Prior over Omega will be defined in the forward to make it depend explicitly on the lengthscale

                # Prior and posterior over W
            self.dimension_correction_factor = 1
            self.dimension_correction_add = 0
                # In the rbf case, we need to double the size of W because Phi is obtained
                # by concatenating sin(X Omega) and cos(X Omega)
            if self.kernel == "rbf":
                 self.dimension_correction_factor = 2
            if self.add_mean:
                 self.dimension_correction_add = self.Din

                ## ********** Prior and posterior over W
                ## -- Mean
            self.prior_W_m = Variable(torch.zeros(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                      self.Dout).type(self.dtype), requires_grad=False)
            self.q_W_m = torch.nn.Parameter(torch.zeros_like(self.prior_W_m.data).type(self.dtype), requires_grad=True)

            if self.add_mean:
                    ## The first block of W will contain the parameters of the linear model for the mean function of the GP
                    ## Put the prior over W to mean = 1 for the parameters of the linear part - similar to residual-net idea
                    self.prior_W_m = Variable(torch.eye(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                            self.Dout).type(self.dtype), requires_grad=False)

                    self.q_W_m = torch.nn.Parameter(torch.eye(self.prior_W_m.shape[0], self.prior_W_m.shape[1]).type(self.dtype), requires_grad=True)

                ## -- Variance/Covariance
            self.prior_W_logv = Variable(torch.zeros(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                         self.Dout).type(self.dtype), requires_grad=False)
            #TODO: non factorized prior
            if self.factorized:
                self.q_W_logv = torch.nn.Parameter(torch.zeros_like(self.prior_W_logv.data).type(self.dtype), requires_grad=True)

        def predict(self,X,Nmc):# X is Nmc x n x dout
            Omega_sample = torch.exp(-self.log_lengthscale)*2.**0.5*self.epsilon_for_Omega_sample
            Phi_before_activation = torch.matmul(X, Omega_sample)
            # Forward computation of F # TODO: save computation if Phi doesn't not change during optim
            if self.kernel == "rbf":
                self.Phi_noStd = torch.cat((torch.sin(Phi_before_activation), torch.cos(Phi_before_activation)), 2)
            Phi = self.Phi_noStd * torch.sqrt(torch.exp(self.log_sigma2) / self.NRFs)
            mean_F = torch.matmul(Phi, self.q_W_m)
            var_F = torch.matmul(Phi**2, torch.exp(self.q_W_logv))
            epsilon_for_F_sample   = Variable(torch.randn(Nmc, Phi.shape[1]  , self.Dout).type(self.dtype), requires_grad=False)
            return epsilon_for_F_sample*torch.sqrt(var_F)+mean_F
           
class calib_GP_RFs_SVI_layer(torch.nn.Module):
        def __init__(self, D1, D2, Dout, NRFs, kernel, kernel_ard, factorized, learn_Omega, learn_hyperparam, add_mean, 
                 local_reparameterization, dtype,priorThetaMean=None,priorThetaCovRoot=None,seedForOmega=None,additiveDiscr=True):
            super(calib_GP_RFs_SVI_layer, self).__init__()
            self.D1 = D1
            self.D2 = D2
            self.Dout = Dout
            self.NRFs = NRFs
            self.kernel = kernel
            self.kernel_ard = kernel_ard
            self.factorized = factorized
            self.learn_Omega = learn_Omega
            self.learn_hyperparam = learn_hyperparam
            self.add_mean = add_mean
            self.local_reparameterization = local_reparameterization
            self.dtype = dtype
            self.additiveDiscr = additiveDiscr

            if self.learn_hyperparam == "optim":
                    self.log_eta_sigma2 = torch.nn.Parameter((0.02**2*torch.ones(1, 1)).log().type(self.dtype), requires_grad=True)
                    self.log_delta_sigma2 = torch.nn.Parameter((0.002**2*torch.ones(1, 1)).log().type(self.dtype), requires_grad=True)
                    if not kernel_ard:
                            self.log_eta_lengthscale = torch.nn.Parameter((0.27*torch.ones(1, 1)).log().type(self.dtype),requires_grad=True)
                            self.log_delta_lengthscale = torch.nn.Parameter((0.1*torch.ones(1, 1)).log().type(self.dtype),requires_grad=True)
                    if kernel_ard:
                        self.log_eta_lengthscale = torch.nn.Parameter(
                            torch.ones(self.D1+self.D2,1).type(self.dtype) * 0.5 * np.log((self.D1+self.D2) * 1.0) - np.log(2.0),
                            requires_grad=True)  # Initialize lengthscale to sqrt(D) / 2
                        if additiveDiscr:
                            self.log_delta_lengthscale = torch.nn.Parameter(
                                torch.ones(self.D1,1).type(self.dtype) * 0.5 * np.log(self.D1 * 1.0) - np.log(2.0),
                                requires_grad=True)  # Initialize lengthscale to sqrt(D) / 2
                        else:
                            self.log_delta_lengthscale = torch.nn.Parameter(
                                torch.ones(self.D1+1,1).type(self.dtype) * 0.5 * np.log((self.D1+1) * 1.0) - np.log(2.0),
                                requires_grad=True)  # Initialize lengthscale to sqrt(D) / 2
                    # If random features are fixed from the prior we fix the randomness - the lengthscale
                    # will be learnt using the reparameterization trick
            if self.learn_Omega == "prior_fixed":
                        if seedForOmega is not None:
                            torch.manual_seed(seedForOmega)
                        self.epsilon_for_Omega_eta_sample = Variable(torch.randn(self.D1+self.D2, self.NRFs).type(self.dtype),
                                                                 requires_grad=False)
                        if additiveDiscr:
                            self.epsilon_for_Omega_delta_sample = Variable(torch.randn(self.D1, self.NRFs).type(self.dtype),
                                                                 requires_grad=False)
                        else:
                            self.epsilon_for_Omega_delta_sample = Variable(torch.randn(self.D1+self.Dout, self.NRFs).type(self.dtype),
                                                                 requires_grad=False)
                            self.etaAnisotropy = torch.nn.Parameter(torch.ones(1).log().type(self.dtype), requires_grad=True)
                            
                        # TODO : If random features are learned variationally, we learn the approximate
                    # posterior q(Omega) fixing the randomness 
                    #if self.learn_Omega == "var_fixed":
                    #    self.epsilon_for_Omega_sample = Variable(torch.randn(1, self.Din, self.NRFs).type(self.dtype),
                    #                                             requires_grad=False)

                    #    self.q_Omega_m = torch.nn.Parameter(torch.zeros(self.Din, self.NRFs).type(self.dtype),
                    #                                        requires_grad=True)
                    #    self.q_Omega_logv = torch.nn.Parameter(
                    #        torch.ones(self.Din, self.NRFs).type(self.dtype) * -2.0 * self.log_theta_lengthscale.data,
                    #        requires_grad=True)  # We initialize q(Omega) as p(Omega)

                # Prior over Omega will be defined in the forward to make it depend explicitly on the lengthscale

                # Prior and posterior over W
            self.dimension_correction_factor = 1
            self.dimension_correction_add = 0
                # In the rbf case, we need to double the size of W because Phi is obtained
                # by concatenating sin(X Omega) and cos(X Omega)
            if self.kernel == "rbf":
                 self.dimension_correction_factor = 2
            if self.add_mean:
                 self.dimension_correction_add = self.D1+self.D2

                ## ********** Prior and posterior over W
                ## -- Mean
            self.prior_W_eta_m = Variable(torch.zeros(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                      self.Dout).type(self.dtype), requires_grad=False)
            self.prior_W_delta_m = Variable(torch.zeros(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                      self.Dout).type(self.dtype), requires_grad=False)
            self.q_W_eta_m = torch.nn.Parameter(torch.zeros_like(self.prior_W_eta_m.data).type(self.dtype), requires_grad=True)
            self.q_W_delta_m = torch.nn.Parameter(torch.zeros_like(self.prior_W_delta_m.data).type(self.dtype), requires_grad=True)

            if self.add_mean:
                    ## The first block of W will contain the parameters of the linear model for the mean function of the GP
                    ## Put the prior over W to mean = 1 for the parameters of the linear part - similar to residual-net idea
                    self.prior_W_eta_m = Variable(torch.eye(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                            self.Dout).type(self.dtype), requires_grad=False)
                    self.prior_W_delta_m = Variable(torch.eye(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                            self.Dout).type(self.dtype), requires_grad=False)

                    self.q_W_eta_m = torch.nn.Parameter(torch.eye(self.prior_W_m.shape[0], self.prior_W_m.shape[1]).type(self.dtype), requires_grad=True)
                    self.q_W_delta_m = torch.nn.Parameter(torch.eye(self.prior_W_m.shape[0], self.prior_W_m.shape[1]).type(self.dtype), requires_grad=True)
            if (priorThetaMean is not None):
                self.prior_theta_m = Variable(priorThetaMean, requires_grad=False)
            else:
                self.prior_theta_m = Variable(torch.zeros(D2).type(self.dtype), requires_grad=False)
            self.q_theta_m = torch.nn.Parameter(self.prior_theta_m.data.clone(), requires_grad=True)
                ## -- Variance/Covariance
            self.prior_W_eta_logv = Variable(torch.zeros(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                         self.Dout).type(self.dtype), requires_grad=False)
            self.prior_W_delta_logv = Variable(torch.zeros(self.NRFs * self.dimension_correction_factor + self.dimension_correction_add, 
                                                         self.Dout).type(self.dtype), requires_grad=False)
            #TODO: non factorized prior
            if self.factorized:
                if (priorThetaCovRoot is not None):
                    self.prior_theta_logv = Variable((torch.diag(priorThetaCovRoot)**2).log().type(self.dtype), requires_grad=False)
                else :
                    self.prior_theta_logv = Variable((0.1**2*torch.ones(D2)).log().type(self.dtype), requires_grad=False)
                self.q_W_eta_logv = torch.nn.Parameter(torch.zeros_like(self.prior_W_eta_logv.data).type(self.dtype), requires_grad=True)
                self.q_W_delta_logv = torch.nn.Parameter(torch.zeros_like(self.prior_W_delta_logv.data).type(self.dtype), requires_grad=True)
            self.q_theta_logv = torch.nn.Parameter(self.prior_theta_logv.data.clone(), requires_grad=True)

        def predictGivenTheta(self,X,XStar,T,Nmc,theta_sample=None,forcePhiCompute=False):
            if (self.learn_Omega == "prior_fixed"):
                if (not self.log_eta_lengthscale.requires_grad):
                    computePhiEta_computerCode = False
                    if forcePhiCompute:
                        computePhiEta_computerCode = True
                else:
                    computePhiEta_computerCode = True
                if (not self.log_delta_lengthscale.requires_grad):
                    computePhiDelta = False
                    if forcePhiCompute:
                        computePhiDelta = True
                        Omega_delta_sample = torch.exp(-self.log_delta_lengthscale)*2.**0.5*self.epsilon_for_Omega_delta_sample
                else:
                    computePhiDelta = True
                    Omega_delta_sample = torch.exp(-self.log_delta_lengthscale)*2.**0.5*self.epsilon_for_Omega_delta_sample
                ## NB: Omega_eta_sample is always computed even with fixed lengthscale because of random theta
                Omega_eta_sample = torch.exp(-self.log_eta_lengthscale)*2.**0.5*self.epsilon_for_Omega_eta_sample
            if theta_sample is None:
                epsilon_for_theta_sample = Variable(torch.randn(Nmc, self.D2).type(self.dtype), requires_grad=False)
                if self.factorized:
                    theta_sample = torch.add(torch.mul(epsilon_for_theta_sample, torch.exp(self.q_theta_logv / 2.0)), self.q_theta_m)
            XthetaRep = torch.cat(((X.unsqueeze(0)).expand(Nmc,-1,-1),theta_sample.unsqueeze(1).expand(-1,X.size()[0],-1)),2)

            Phi_eta_before_activation = torch.matmul(XthetaRep, Omega_eta_sample)
            if computePhiEta_computerCode:
                Phi_eta_before_activation_computerCode = torch.mm(torch.cat((XStar,T),1),Omega_eta_sample)
            # Forward computation of F # TODO: save computation if Phi doesn't not change during optim
            if self.kernel == "rbf":
                self.Phi_eta_noStd = torch.cat((torch.sin(Phi_eta_before_activation), torch.cos(Phi_eta_before_activation)), 2)
                if computePhiEta_computerCode:
                    self.Phi_eta_computerCode_noStd = torch.cat((torch.sin(Phi_eta_before_activation_computerCode), torch.cos(Phi_eta_before_activation_computerCode)), 1)
                else:
                    self.Phi_eta_computerCode_noStd = self.Phi_eta_computerCode_noStd.detach()
            Phi_eta = self.Phi_eta_noStd * torch.sqrt(torch.exp(self.log_eta_sigma2) / self.NRFs)
            Phi_eta_computerCode = self.Phi_eta_computerCode_noStd * torch.sqrt(torch.exp(self.log_eta_sigma2) / self.NRFs)
            mean_F_eta = torch.matmul(Phi_eta, self.q_W_eta_m)
            var_F_eta = torch.matmul(Phi_eta**2, torch.exp(self.q_W_eta_logv))
            mean_F_eta_computerCode = torch.matmul(Phi_eta_computerCode, self.q_W_eta_m)
            var_F_eta_computerCode = torch.matmul(Phi_eta_computerCode**2, torch.exp(self.q_W_eta_logv))
            if computePhiDelta:
                if self.additiveDiscr:
                    Phi_delta_before_activation = torch.matmul(XthetaRep[:,:,0:(self.D1)], Omega_delta_sample)
                else:
                    epsilon_for_F_eta_sample = Variable(torch.randn(Nmc, XthetaRep.size()[1],self.Dout).type(self.dtype), requires_grad=False)
                    F_eta_sample =epsilon_for_F_eta_sample*var_F_eta.sqrt()+ mean_F_eta
                    Phi_delta_before_activation = torch.matmul(torch.cat((self.etaAnisotropy.exp()*(F_eta_sample),XthetaRep[:,:,0:(self.D1)]),2), Omega_delta_sample)#s
            if self.kernel == "rbf":
                if computePhiDelta:
                    self.Phi_delta_noStd = torch.cat((torch.sin(Phi_delta_before_activation), torch.cos(Phi_delta_before_activation)), 2)
                else:
                    self.Phi_delta_noStd = self.Phi_delta_noStd.detach()
            Phi_delta = self.Phi_delta_noStd * torch.sqrt(torch.exp(self.log_delta_sigma2) / self.NRFs)
            mean_F_delta = torch.matmul(Phi_delta, self.q_W_delta_m)
            var_F_delta = torch.matmul(Phi_delta**2, torch.exp(self.q_W_delta_logv))
            return mean_F_eta+mean_F_delta, var_F_eta+var_F_delta,mean_F_eta_computerCode,var_F_eta_computerCode
        
        
        def predictGivenThetaRep(self,XthetaRep,XStarT,Nmc):
            if (self.learn_Omega == "prior_fixed"):
                Omega_delta_sample = torch.exp(-self.log_delta_lengthscale)*2.**0.5*self.epsilon_for_Omega_delta_sample
                Omega_eta_sample = torch.exp(-self.log_eta_lengthscale)*2.**0.5*self.epsilon_for_Omega_eta_sample
            Phi_eta_before_activation = torch.matmul(XthetaRep, Omega_eta_sample)
            Phi_eta_before_activation_computerCode = torch.matmul(XStarT,Omega_eta_sample)
            # Forward computation of F # TODO: save computation if Phi doesn't not change during optim
            if self.kernel == "rbf":
                self.Phi_eta_noStd = torch.cat((torch.sin(Phi_eta_before_activation), torch.cos(Phi_eta_before_activation)), 2)
                self.Phi_eta_computerCode_noStd = torch.cat((torch.sin(Phi_eta_before_activation_computerCode), torch.cos(Phi_eta_before_activation_computerCode)), 2)
            Phi_eta = self.Phi_eta_noStd * torch.sqrt(torch.exp(self.log_eta_sigma2) / self.NRFs)
            Phi_eta_computerCode = self.Phi_eta_computerCode_noStd * torch.sqrt(torch.exp(self.log_eta_sigma2) / self.NRFs)
            mean_F_eta = torch.matmul(Phi_eta, self.q_W_eta_m)
            var_F_eta = torch.matmul(Phi_eta**2, torch.exp(self.q_W_eta_logv))
            mean_F_eta_computerCode = torch.matmul(Phi_eta_computerCode, self.q_W_eta_m)
            var_F_eta_computerCode = torch.matmul(Phi_eta_computerCode**2, torch.exp(self.q_W_eta_logv))
            
            if self.additiveDiscr:
                Phi_delta_before_activation = torch.matmul(XthetaRep[:,:,0:(self.D1)], Omega_delta_sample)
            else:
                epsilon_for_F_eta_sample = Variable(torch.randn(Nmc, XthetaRep.size()[1],self.Dout).type(self.dtype), requires_grad=False)
                F_eta_sample =epsilon_for_F_eta_sample*var_F_eta.sqrt()+ mean_F_eta
                Phi_delta_before_activation = torch.matmul(torch.cat((self.etaAnisotropy.exp()*(F_eta_sample),XthetaRep[:,:,0:(self.D1)]),2), Omega_delta_sample)#s

            if self.kernel == "rbf":
                self.Phi_delta_noStd = torch.cat((torch.sin(Phi_delta_before_activation), torch.cos(Phi_delta_before_activation)), 2)

            Phi_delta = self.Phi_delta_noStd * torch.sqrt(torch.exp(self.log_delta_sigma2) / self.NRFs)


            mean_F_delta = torch.matmul(Phi_delta, self.q_W_delta_m)
            var_F_delta = torch.matmul(Phi_delta**2, torch.exp(self.q_W_delta_logv))

            return mean_F_eta+mean_F_delta, var_F_eta+var_F_delta,mean_F_eta_computerCode,var_F_eta_computerCode
        
     
# ****************************** Define model as a nn module
class calib_GP_RFs_SVI(torch.nn.Module):
    def __init__(self, D1,D2, Dout, NRFs, batch_size, kernel, kernel_ard, factorized, learn_Omega, learn_hyperparam, likelihood, add_mean, local_reparameterization,
                 dtype,priorThetaMean=None,priorThetaCovRoot=None,seedForOmega=None,nn_layers=None,additiveDiscr=True):
        super(calib_GP_RFs_SVI, self).__init__()
        self.D1 = D1
        self.D2 = D2
        self.Dout = Dout
        self.NRFs = NRFs
        self.batch_size = batch_size
        self.kernel = kernel
        self.kernel_ard = kernel_ard
        self.factorized = factorized
        self.learn_Omega = learn_Omega
        self.learn_hyperparam = learn_hyperparam
        self.likelihood_name = likelihood
        self.add_mean = add_mean
        self.local_reparameterization = local_reparameterization
        self.dtype = dtype
        self.log_2_pi_torch = Variable(torch.ones(1), requires_grad=False).type(dtype) * np.log(
            2.0 * np.pi)

        # Create model as a nn.ModuleList - syntax requires += of a list with any layers that need to be added
        # TODO!! Look into nn.Sequential as it might be the easiest way to concatenate layers!

        self.layers = torch.nn.ModuleList()

        self.layers += [
                calib_GP_RFs_SVI_layer(self.D1,self.D2, self.Dout, self.NRFs, self.kernel, self.kernel_ard, self.factorized, self.learn_Omega, 
                self.learn_hyperparam, self.add_mean, self.local_reparameterization, self.dtype,priorThetaMean,priorThetaCovRoot,seedForOmega=seedForOmega,additiveDiscr=additiveDiscr)]
        if nn_layers is None:
            self.deepness= 0
        else:
            self.deepness= len(nn_layers)
        if nn_layers is not None:
            self.layers += nn_layers

        if self.likelihood_name == "gaussian":
            # We define the variance of the noise here instead of the nested class to make it easy
            # to optimize all model parameters using the torch.nn module
            self.log_Y_noise_var = torch.nn.Parameter((torch.ones(self.Dout) * 0.1**2).log(), requires_grad=True)
            self.log_Z_noise_var = torch.nn.Parameter((torch.ones(self.Dout) * 0.1**2).log(), requires_grad=True)
            self.likelihood = likelihoods.Gaussian(self)

        if self.likelihood_name == "softmax":
            self.likelihood = likelihoods.Softmax()

    # Define forward computation throughout the layers
    def forward(self, X,XStar,T, Nmc):
        if self.deepness>0:
            epsilon_for_theta_sample = Variable(torch.randn(Nmc, self.D2).type(self.dtype), requires_grad=False)
            if self.layers[0].factorized:
                theta_sample = torch.add(torch.mul(epsilon_for_theta_sample, torch.exp(self.layers[0].q_theta_logv / 2.0)), self.layers[0].q_theta_m)  
            XthetaRep = torch.cat(((X.unsqueeze(0)).expand(Nmc,-1,-1),theta_sample.unsqueeze(1).expand(-1,X.size()[0],-1)),2)
            XStarT = torch.cat((XStar,T),1).unsqueeze(0).expand(Nmc,-1,-1)
            for i in range(self.deepness):
                XthetaRep = self.layers[i+1].predict(XthetaRep,Nmc)
                XStarT = self.layers[i+1].predict(XStarT,Nmc)
            meanYGivenTheta,varYGivenTheta,meanZGivenTheta,varZGivenTheta =  self.layers[0].predictGivenThetaRep(XthetaRep, XStarT, Nmc)
        else:
            meanYGivenTheta,varYGivenTheta,meanZGivenTheta,varZGivenTheta =  self.layers[0].predictGivenTheta(X, XStar, T, Nmc,theta_sample=None,forcePhiCompute=True)
        return meanYGivenTheta,varYGivenTheta,meanZGivenTheta,varZGivenTheta

    ## Computation of the negative expected log-likelihood
    def compute_nell(self, X, XStar, T, Y, Z, Nmc, n, N, batch_size,batch_sizeY=None):

        if batch_sizeY is None:
            batch_sizeY=n
        
        meanYGivenTheta,varYGivenTheta,meanZGivenTheta,varZGivenTheta = self.forward(X,XStar,T, Nmc)
            #latent_valY,latent_valZ =  self.forward(X, XStar, T, Nmc)
            #likelY = - 0.5 * (n*(self.log_2_pi_torch+self.log_Y_noise_var) + ((Y-latent_valY)**2).sum(1) * torch.exp(-self.log_Y_noise_var))
        likelY = - 0.5 * (n*(self.log_2_pi_torch+self.log_Y_noise_var) + (Y**2-2*Y*meanYGivenTheta+varYGivenTheta+meanYGivenTheta**2).sum(1) * torch.exp(-self.log_Y_noise_var))
        nellY  = - n/batch_sizeY*torch.mean(likelY, 0)
        if self.deepness>0:
            likelZ = - 0.5 * (batch_size*(self.log_2_pi_torch+self.log_Z_noise_var) + (Z**2-2*Z*meanZGivenTheta+varZGivenTheta+meanZGivenTheta**2).sum(1) * torch.exp(-self.log_Z_noise_var))
            nellZ  = - N / batch_size * torch.mean(likelZ, 0)
        
        else:
            #likelZ  = - 0.5 * (batch_size*(self.log_2_pi_torch+self.log_Z_noise_var) + ((Z-latent_valZ)**2).sum(1) * torch.exp(-self.log_Z_noise_var))
            likelZ = - 0.5 * (batch_size*(self.log_2_pi_torch+self.log_Z_noise_var) + (Z**2-2*Z*meanZGivenTheta+varZGivenTheta+meanZGivenTheta**2).sum(0) * torch.exp(-self.log_Z_noise_var))
            #nellZ  = - N / batch_size * torch.mean(likelZ, 0)
            nellZ  = - N / batch_size * likelZ
            
        self.nell = (nellY + nellZ)
        return self.nell
        
    ## Computation of the DKL
    def compute_dkl(self):
        ## DKL
        self.dkl = 0
        # DKL for W
        if self.factorized:
            self.dkl += utilities.DKL_gaussian_q_diag_p_diag(self.layers[0].q_W_eta_m, self.layers[0].q_W_eta_logv, self.layers[0].prior_W_eta_m, self.layers[0].prior_W_eta_logv)
            self.dkl += utilities.DKL_gaussian_q_diag_p_diag(self.layers[0].q_W_delta_m, self.layers[0].q_W_delta_logv, self.layers[0].prior_W_delta_m, self.layers[0].prior_W_delta_logv)
        if self.deepness > 0:
            for i in range(self.deepness):
                self.dkl += utilities.DKL_gaussian_q_diag_p_diag(self.layers[i+1].q_W_m, self.layers[i+1].q_W_logv, self.layers[i+1].prior_W_m, self.layers[i+1].prior_W_logv)
        # DKL for Omega if necessary
        #if self.learn_Omega == "var_fixed":
        #    for i in range(self.Nlayers):
        #        self.dkl += utilities.DKL_gaussian_q_diag_p_diag(self.layers[i].q_Omega_m, self.layers[i].q_Omega_logv,
        #                                                         self.layers[i].prior_Omega_m,
        #self.layers[i].prior_Omega_logv)
        # DKL for theta
        if self.factorized:
            self.dkl += utilities.DKL_gaussian_q_diag_p_diag(self.layers[0].q_theta_m, self.layers[0].q_theta_logv,
                                                                  self.layers[0].prior_theta_m, self.layers[0].prior_theta_logv)    
        return self.dkl

    ## Computation of the NELBO
    def compute_nelbo(self,X,XStar,T, Y, Z, Nmc, n, N, batch_size,batch_sizeY=None):
        self.nelbo =  self.compute_nell(X,XStar,T, Y, Z, Nmc, n, N, batch_size,batch_sizeY) + self.compute_dkl()
        return self.nelbo

    ## Learn
    def learn(self, X, XStar, T, Y, Z, n, N, Nmc_train, batch_size, XTtest, YZtest, Ntest, Nmc_test,
              verbose=5,learning_rate=10**(-3),optimisationDescription=None,saveStates=None,batch_sizeY=None,displayZlearn=False,displayYlearn=False,displayDef=False):

        
        
        if (optimisationDescription is None):
            optimisationDescription = torch.zeros(1,18)
        #0 nIt 1learning,2"NLB",3"NLL",4"DKL", 5grad norm,6"NsY",7"NsZ",8"Ret",9Set",10"Rdl",11"Sdl",12"Mth",13"Sth"
        paraNames = ["Lrt","NLB","NLL","DKL","GdN","NsY","NsZ","Ret","Set","Rdl","Sdl","Mth","Sth","MWE","SWE","MWD","SWD"] 
        
        if self.batch_size > N:
            print("Warning: you selected batch_size > N - the batch size will be reduced to N\n\n")
            self.batch_size = N
        def displayLine(st,t,optimisationDescription,header=True,saveStates=None):
            Niterations = optimisationDescription[st,0]
            nbDisplaThetam = int(np.min([((self.layers[0].q_theta_m.data)).size()[0],8]))
            if header:
                headerLine = " Ite | % |"
                for i in range(17):
                    if (optimisationDescription[st,i+1]%2==0):
                        if (i==11):
                            for k in range(nbDisplaThetam):
                                headerLine += "   "+paraNames[i]+"   |"
                        else:
                            headerLine = headerLine+"   "+paraNames[i]
                            if i<16:
                                headerLine+="   |"
                print(headerLine)
            model_grad_val = 0
            if (not (t==0 and st==0)):
                model_grad = filter(lambda p: p.requires_grad, self.parameters())
                #totalGrad = torch.zeros(0)
                model_grad_val=0
                for j in model_grad:
                    if j.grad is not None:
                        model_grad_val = model_grad_val+(j.grad.data**2).sum()
                    #totalGrad = torch.cat((totalGrad,j.grad.data.squeeze()),0)
                    #print(j.grad.data.squeeze())
                #print(model_grad_val)
            myStr = ""
            collect = [st]
            myStrAdd = "%5.0f" % t
            collect += [t]
            myStr += myStrAdd+"|"
            myStrAdd = "%3.0f" % (float(100*t)/float(Niterations))
            collect += [float(100*t)/float(Niterations)]
            myStr += myStrAdd+"|"
            if (optimisationDescription[st,1]%2==0):
                myStrAdd = "%9.3f" % self.learning_rate
                myStr += myStrAdd+"|"
                collect += [self.learning_rate]
            if (optimisationDescription[st,2]%2==0):
                myStrAdd = "%9.3f" % loss.data.item()
                myStr += myStrAdd+"|"
                collect += [loss.data.item()]
            if (optimisationDescription[st,3]%2==0):
                myStrAdd = "%9.3f" % self.nell.data.item()
                myStr += myStrAdd+"|"
                collect += [self.nell.data.item()]
            if (optimisationDescription[st,4]%2==0):
                myStrAdd = "%9.3f" % self.dkl.data.item()
                myStr += myStrAdd+"|"
                collect += [self.dkl.data.item()]
            if (optimisationDescription[st,5]%2==0):
                myStrAdd = "%9.3f" % model_grad_val
                myStr += myStrAdd+"|"
                collect += [float(model_grad_val)]
            if (optimisationDescription[st,6]%2==0):
                myStrAdd = "%9.3f" % ((self.log_Y_noise_var.data/2).exp()).item()
                myStr += myStrAdd+"|"
                collect += [float(((self.log_Y_noise_var.data/2).exp()).item())]
            if (optimisationDescription[st,7]%2==0):
                myStrAdd = "%9.3f" % ((self.log_Z_noise_var.data/2).exp()).item()
                myStr += myStrAdd+"|"
                collect += [float(((self.log_Z_noise_var.data/2).exp()).item())]
            if (optimisationDescription[st,8]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].log_eta_lengthscale.data).exp().mean(0)).item()
                myStr += myStrAdd+"|"
                collect += [float(((self.layers[0].log_eta_lengthscale.data).exp().mean(0)).item())]
            if (optimisationDescription[st,9]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].log_eta_sigma2.data/2).exp()).item()
                myStr += myStrAdd+"|"
                collect += [float(((self.layers[0].log_eta_sigma2.data/2).exp()).item())]
            if (optimisationDescription[st,10]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].log_delta_lengthscale.data).exp().mean(0)).item()
                myStr += myStrAdd+"|"
                collect += [float(((self.layers[0].log_delta_lengthscale.data).exp().mean(0)).item())]
            if (optimisationDescription[st,11]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].log_delta_sigma2.data/2).exp()).item()
                myStr += myStrAdd+"|"
                collect += [float(((self.layers[0].log_delta_sigma2.data/2).exp()).item())]
            if (optimisationDescription[st,12]%2==0):
                for k in range(nbDisplaThetam):
                    myStrAdd = "%9.3f" % ((self.layers[0].q_theta_m.data))[k]
                    myStr += myStrAdd+"|"
                    collect += [((self.layers[0].q_theta_m.data))[k]]
            if (optimisationDescription[st,13]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].q_theta_logv.data/2).exp()).mean()
                myStr += myStrAdd+"|"
                collect += [((self.layers[0].q_theta_logv.data/2).exp()).mean()]
            if (optimisationDescription[st,14]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].q_W_eta_m.data).mean())
                myStr += myStrAdd+"|"
            if (optimisationDescription[st,15]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].q_W_eta_logv.data/2).exp().mean())
                myStr += myStrAdd+"|"
            if (optimisationDescription[st,16]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].q_W_delta_m.data).mean())
                myStr += myStrAdd+"|"
            if (optimisationDescription[st,17]%2==0):
                myStrAdd = "%9.3f" % ((self.layers[0].q_W_delta_logv.data/2).exp().mean())
                myStr += myStrAdd

            print(myStr)
            if saveStates is not None:
                if t==0 and st==0:
                    with open(saveStates, 'wb') as output:
                        pickle.dump([[self,collect]], output)
                else:
                    with open(saveStates, 'rb') as input:
                        modelList = pickle.load(input)
                    with open(saveStates, 'wb') as output:
                        pickle.dump(modelList+[[self,collect]], output)

        for st in range(optimisationDescription.size()[0]):
            if (optimisationDescription[st,1]==0):
                self.learning_rate = learning_rate
            else:
                self.learning_rate = optimisationDescription[st,1]
            if (optimisationDescription[st,0]==0):
                optimisationDescription[st,0] = 10000
            Niterations = int(optimisationDescription[st,0])
            if self.likelihood_name == "gaussian":
                if self.learn_hyperparam == "optim":
                    self.log_Y_noise_var.requires_grad                 = (float(optimisationDescription[st,6])  <= 1)
                    self.log_Z_noise_var.requires_grad                 = (float(optimisationDescription[st,7])  <= 1)
                    self.layers[0].log_eta_lengthscale.requires_grad   = (float(optimisationDescription[st,8])  <= 1)
                    self.layers[0].log_eta_sigma2.requires_grad        = (float(optimisationDescription[st,9])  <= 1)
                    self.layers[0].log_delta_lengthscale.requires_grad = (float(optimisationDescription[st,10]) <= 1)
                    self.layers[0].log_delta_sigma2.requires_grad      = (float(optimisationDescription[st,11]) <= 1)
                    self.layers[0].q_theta_m.requires_grad             = (float(optimisationDescription[st,12]) <= 1)
                    self.layers[0].q_theta_logv.requires_grad          = (float(optimisationDescription[st,13]) <= 1)
                    self.layers[0].q_W_eta_m.requires_grad             = (float(optimisationDescription[st,14]) <= 1)
                    self.layers[0].q_W_eta_logv.requires_grad          = (float(optimisationDescription[st,15]) <= 1)
                    self.layers[0].q_W_delta_m.requires_grad           = (float(optimisationDescription[st,16]) <= 1)
                    self.layers[0].q_W_delta_logv.requires_grad        = (float(optimisationDescription[st,17]) <= 1)

            ## Define optimizer and SVI loop
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
            if (verbose != 0):
                b = 1/9*np.log(Niterations/2)
                A = 2/np.exp(b)
                modulo = int(Niterations/(A*np.exp(b*verbose)))
                if (modulo==0):
                    modulo=1
            for t in (range(Niterations)):
                if batch_size < N:
                    indices_batch = np.random.choice(N, batch_size, replace=False)
                    if batch_sizeY is None:
                        loss = self.compute_nelbo(X,XStar[indices_batch,:],T[indices_batch,:], Y, Z[indices_batch,:], Nmc_train, n, N, batch_size)
                    else:
                        indices_batchY = np.random.choice(n, batch_sizeY, replace=False)
                        loss = self.compute_nelbo(X[indices_batchY,:],XStar[indices_batch,:],T[indices_batch,:], Y[indices_batchY,:], Z[indices_batch,:], Nmc_train, n, N, batch_size,batch_sizeY)
                # confuse the gradient computation!!
                else:
                    if batch_sizeY is None:
                        loss = self.compute_nelbo(X,XStar ,T , Y, Z , Nmc_train, n, N, batch_size)
                    else:
                        indices_batchY = np.random.choice(n, batch_sizeY, replace=False)
                        loss = self.compute_nelbo(X[indices_batchY,:],XStar ,T , Y[indices_batchY,:], Z , Nmc_train, n, N, batch_size, batch_sizeY)
                ## Report test error if test set is provided
                #if (t % 2000) == 0:
                #    if Nmc_test > 0:
                #        if self.likelihood_name == "softmax":
                #            output = self(Xtest, Nmc_test)
                #            prediction = torch.max(torch.mean(self.likelihood.predict(output), 0), 1)[1]
                #            target = torch.max(Ytest, 1)[1]
                #            correct = sum(prediction.data == target.data)
                #            
                #            print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / Ntest))

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                #model_grad = filter(lambda p: p.requires_grad, self.parameters())
                if (verbose != 0):
                    displayFigs=False
                    if (t==0):
                        displayLine(st,0,optimisationDescription,saveStates=saveStates)
                        displayFigs=True
                    if (modulo>1):
                        if ((t+1) % modulo  == 0):
                            displayLine(st,t+1,optimisationDescription,header=False,saveStates=saveStates)
                        if (t==Niterations-1 and Niterations%modulo!=0):
                            displayLine(st,Niterations,optimisationDescription,header=False,saveStates=saveStates)
                            displayFigs=True
                    else:
                        displayLine(st,t+1,optimisationDescription,header=False,saveStates=saveStates)
                    if displayFigs:
                        if displayZlearn:
                            if self.deepness>0:
                                print(self.layers[1].log_lengthscale.data.exp())
                                if self.deepness>1:
                                    print(self.layers[2].log_lengthscale.data.exp())
                            indices_batchplt = np.random.choice(N, np.min([N,1000]), replace=False)
                            yy, yyv, zz, zzv = self.forward(X[1:2,:],XStar[indices_batchplt,:],T[indices_batchplt,:],10)
                            if torch.cuda.is_available():
                                if len(zz.squeeze().size())==1:
                                    plt.scatter(zz.squeeze().data.cpu().numpy(),Z[indices_batchplt,:].squeeze().data.cpu().numpy())
                                else:
                                    plt.scatter(zz.squeeze().mean(0).data.cpu().numpy(),Z[indices_batchplt,:].squeeze().data.cpu().numpy())
                            else:
                                if len(zz.squeeze().size())==1:
                                    plt.scatter(zz.squeeze().data.numpy(),Z[indices_batchplt,:].squeeze().data.numpy())
                                else:
                                    plt.scatter(zz.squeeze().mean(0).data.numpy(),Z[indices_batchplt,:].squeeze().data.cpu().numpy())
                            plt.show()
                        if displayYlearn:
                            mask0 = XStar[:,0]==0
                            lower1 = torch.tensor([XStar.min(),XStar.min()]).type(self.dtype)
                            upper1 = torch.tensor([XStar.max(),XStar.max()]).type(self.dtype)
                            axialPre = 80
                            x = (giveGrid(axialPre,1).type(self.dtype)*(upper1[0]-lower1[0]).unsqueeze(0)+lower1[0].unsqueeze(0))
                            meanYGivenTheta,varYGivenTheta,meanZGivenTheta,varZGivenTheta =  self.forward(x, XStar[0:2,:], T[0:2,:], 50)
                            
                            plt.plot(x.squeeze().cpu().numpy(),meanYGivenTheta.data[:,:,0].t().squeeze().cpu().numpy())
                            plt.scatter(XStar.squeeze().cpu().numpy(),Z.squeeze().cpu().numpy(),c="gray")
                            plt.scatter(X.squeeze().cpu().numpy(),Y.squeeze().cpu().numpy())
                            plt.show()
                        if displayDef:
                            lowerDef = torch.tensor([Z.min(),XStar.min()]).type(dtype)
                            upperDef = torch.tensor([Z.max(),XStar.max()]).type(dtype)
                            axialPre = 80
                            actualGrid = (giveGrid(axialPre,D1+1).type(self.dtype)*(upperDef-lowerDef).unsqueeze(0)+lowerDef.unsqueeze(0)).squeeze()
                            Omega_delta_sample = torch.exp(-self.layers[0].log_delta_lengthscale)*2.**0.5*self.layers[0].epsilon_for_Omega_delta_sample
                            Phi_delta_before_activation=torch.matmul(actualGrid, Omega_delta_sample)
                            Phi_delta_noStd = torch.cat((torch.sin(Phi_delta_before_activation), torch.cos(Phi_delta_before_activation)), 1)
                            Phi_delta = Phi_delta_noStd * torch.sqrt(torch.exp(self.layers[0].log_delta_sigma2) / self.layers[0].NRFs)
                            mean_F_delta = torch.matmul(Phi_delta, self.layers[0].q_W_delta_m)
                            #var_F_delta = torch.matmul(Phi_delta**2, torch.exp(model.layers[0].q_W_delta_logv))
                            defo = mean_F_delta+actualGrid[:,0].unsqueeze(1)
                            CS = plt.contourf(defo.data.cpu().view(axialPre,axialPre))
                            plt.clabel(CS,fontsize=9, inline=1,colors="black")
                            plt.show()
                            
                optimizer.step()

## ******************************  
from core.synthetic_data import synthetic_data_prior
from core.utilities import giveGrid # only for display
from core.display import plotCalibDomain # only for display
from core.display import plotVariableDomain # only for display
from core.display import normalPDF # only for debug
seedForOmega=None#int(np.random.randint(0,100000,1)[0])
rds= int(np.random.randint(0,100000,1)[0])
torch.manual_seed(0)


##
display = True    
if display:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mli
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
##
experimentName = "testCase2"
experimentFolder = "./experiments/"+experimentName
##
generateData = True
overwriteDataSet = False
readData = True
saveModelObj = False
outputDisplayFile = experimentFolder+"/images/"+experimentName+"_images.pdf"
inLineDisplay=False

if generateData:
    D1 = 3
    D2 = 2
    Dout = 1
    N    = 10#max(int(3.2**(D1+D2)),40*(D1+D2)**2)
    n    = 10#max(int(8**(D1)),50*(D1)**2)
    print("D1: "+str(D1)+" ; D2: "+str(D2)+" ; N: "+str(N)+" ; n: "+str(n))


#%%
## Generate data
if generateData:
    NRFsSynthetic = 100
    dataPrior = synthetic_data_prior(thetaMean=0.5*torch.ones(D2).type(dtype),etaRange = float(2*np.sqrt(D1+D2))*(torch.ones(Dout,1).type(dtype)),
                                     dtype=dtype,D1=D1,nbFeatures=NRFsSynthetic,seedForOmega=seedForOmega,deltaStd=0.05*torch.ones(Dout).type(dtype))
    X_data, XStar_data, T_data, Y_data, Z_data  = dataPrior.generate(n=n,N=N,repLHS=1)
if overwriteDataSet:
    if not experimentFolder.is_dir():
        os.makedirs(experimentFolder)
        os.makedirs(experimentFolder+"/data")
        os.makedirs(experimentFolder+"/results")
        os.makedirs(experimentFolder+"/images")
    np.savetxt(experimentFolder+"/data/X.csv", X_data.numpy(),delimiter=";")
    np.savetxt(experimentFolder+"/data/Y.csv", Y_data.numpy(),delimiter=";")
    np.savetxt(experimentFolder+"/data/XStar.csv", XStar_data.numpy(),delimiter=";")
    np.savetxt(experimentFolder+"/data/T.csv", T_data.numpy(),delimiter=";")
    np.savetxt(experimentFolder+"/data/Z.csv", Z_data.numpy(),delimiter=";")
    with open(experimentFolder+'/data/priorInfo.pkl', 'wb') as output:
        pickle.dump(dataPrior, output, pickle.HIGHEST_PROTOCOL)
if readData:
    if saveModelObj:
        with open(experimentFolder+'/data/priorInfo.pkl', 'rb') as input:
            dataPrior = pickle.load(input)
    X_np     = np.genfromtxt(experimentFolder+"/data/X.csv",delimiter=";")
    XStar_np = np.genfromtxt(experimentFolder+"/data/XStar.csv",delimiter=";")
    T_np     = np.genfromtxt(experimentFolder+"/data/T.csv",delimiter=";")/10 #!!!!!!!!!!!!!!!!!!!!!!!!!
    Y_np     = (np.genfromtxt(experimentFolder+"/data/Y.csv",delimiter=";")-.3)*5
    Z_np     = (np.genfromtxt(experimentFolder+"/data/Z.csv",delimiter=";")-.3)*5

    X_data     = torch.from_numpy(X_np).type(dtype)
    XStar_data = torch.from_numpy(XStar_np).type(dtype)
    T_data     = torch.from_numpy(T_np).type(dtype)
    Y_data     = torch.from_numpy(Y_np).type(dtype)
    Z_data     = torch.from_numpy(Z_np).type(dtype)
if len(X_data.size())==1:
    X_data = X_data.unsqueeze(1)
if len(XStar_data.size())==1:
    XStar_data = XStar_data.unsqueeze(1)
if len(T_data.size())==1:
    T_data = T_data.unsqueeze(1)
if len(Y_data.size())==1:
    Y_data = Y_data.unsqueeze(1)
if len(Z_data.size())==1:
    Z_data = Z_data.unsqueeze(1)

D1   = X_data.size()[1]
D2   = T_data.size()[1]
Dout = Y_data.size()[1]
n    = Y_data.size()[0]
N    = Z_data.size()[0]

X = Variable(X_data,requires_grad=False)
XStar = Variable(XStar_data,requires_grad=False)
T = Variable(T_data,requires_grad=False)
Y = Variable(Y_data,requires_grad=False)
Z = Variable(Z_data,requires_grad=False)    
print("D1: "+str(D1)+" ; D2: "+str(D2)+" ; N: "+str(N)+" ; n: "+str(n))


#%%
kernel = "rbf"

kernel_ard = False # kernel_ard True or False
learn_Omega = "prior_fixed" # "prior_fixed" (only imple) is fixed from the prior and "var_fixed" is variational with fixed randomness
learn_hyperparam = "optim" # How to treat hyperparam - "optim" or "variational" - only "optim" implemented
add_mean = False
factorized = True  ## only True implemented
local_reparameterization = True ## 
likelihood = "gaussian" ## only "gaussian" implemented


#%%
Nmc = 10
NRFs=200
batch_size=int(np.min([N,500]))


#%%
additiveDiscr = False


#%%
priorThetaMean = .5*(torch.ones(D2).type(dtype))#dataPrior.thetaMean
priorThetaCovRoot =  .5*torch.diag(torch.ones(D2).type(dtype))

model = calib_GP_RFs_SVI(D1,D2, Dout, NRFs, batch_size, kernel, kernel_ard, factorized, learn_Omega, learn_hyperparam, likelihood,
                         add_mean, local_reparameterization, dtype,priorThetaMean,priorThetaCovRoot,seedForOmega=seedForOmega,additiveDiscr=additiveDiscr)

learning_rate = 0.5*1e-3


#%%
# 0 log_Z_noise_var
model.log_Z_noise_var.data = (0.01**2*torch.ones(model.Dout)).log().type(model.dtype)
# 1 log_Y_noise_var
model.log_Y_noise_var.data = (0.001**2*torch.ones(model.Dout)).log().type(model.dtype)


#%%
# 2 log_eta_sigma2
model.layers[0].log_eta_sigma2.data   = (1**2 *torch.ones(1, 1).type(dtype)).log().type(model.dtype)

# 4 log_eta_lengthscale

model.layers[0].log_eta_lengthscale.data   = (float(.5)*torch.ones(1, 1).type(dtype)).log().type(model.dtype)
deltaRangeSaved=float(np.sqrt(D1+Dout))
deltaStdSaved=1
model.layers[0].log_delta_lengthscale.data = (deltaRangeSaved*torch.ones(1, 1).type(dtype)).log().type(model.dtype)
model.layers[0].log_delta_sigma2.data = (deltaStdSaved**2*torch.ones(1, 1).type(dtype)).log().type(model.dtype)
# 6 q_W_eta_m
# 7 q_W_delta_m
# 8 q_theta_m
# 9 q_W_eta_logv
#10 q_W_delta_logv
#11 q_theta_logv
model.layers[0].q_theta_m.data = priorThetaMean
model.layers[0].q_theta_logv.data = torch.diag(priorThetaCovRoot**2).log()
#for i in model.parameters():
#    print(i.size())
#mF = model.forward(X,XStar,T, Nmc)


#%%
if not additiveDiscr:
    model.layers[0].etaAnisotropy.requires_grad=False


#%%
if inLineDisplay:
    axialPre = 10
    lower1 = (-0*torch.ones(D1).type(dtype));upper1 = (1*torch.ones(D1).type(dtype))
    lower2 = torch.min(T,0)[0].data.type(dtype);upper2 = torch.max(T,0)[0].data.type(dtype)

    if D2 <= 2:
        tGrid, qTheta  = plotCalibDomain(X.data, XStar.data, T.data, Y.data, Z.data,  model,lower2,upper2,model.layers[0].prior_theta_m.data,
                                         torch.diag(model.layers[0].prior_theta_logv.data.exp().sqrt()),dataPrior.trueTheta,axialPre,outputFile=None)#experimentFolder+'/results/calibDomain.pdf')

        values, indices = qTheta.max(0)
        theta_MAP = tGrid[indices,:].unsqueeze(1)

        thetaAnaly = theta_MAP

        print(dataPrior.trueTheta.unsqueeze(0))
        print(theta_MAP)


    if (D1==1 and D2<=3):
        if D2 > 2:
            thetaAnaly = dataPrior.trueTheta.unsqueeze(0)
        epsilon_for_theta_sample = torch.randn(3, model.layers[0].D2).type(model.dtype)
        thetaNum = torch.add(torch.mul(epsilon_for_theta_sample, torch.exp(model.layers[0].q_theta_logv.data / 2.0)), model.layers[0].q_theta_m.data)
        plotVariableDomain(X.data, XStar.data, T.data, Y.data, Z.data,  model,lower1,upper1,
                          model.layers[0].prior_theta_m.data,torch.diag(model.layers[0].prior_theta_logv.data.exp()),
                          thetaNum,thetaAnaly,axialPre)


        x = giveGrid(axialPre,1).type(model.dtype)*(upper1-lower1).unsqueeze(0)+lower1.unsqueeze(0)
        meanYx_vi, varYx_vi,meanZx_vi,varZx_vi = model.layers[0].predictGivenTheta(
                Variable(x,requires_grad=False),
                XStar,
                T,
                thetaNum.size()[0],
                Variable(thetaNum,requires_grad=False))


#%%
##### LEARNING  

optimisationDescription = torch.Tensor( # > 1 means fixed, even means display
[#0nIt     1learnRate 2NLB  3NLL, 4DKL, 5gdN, 6NsY, 7NsZ, 8Ret, 9Set,10Rdl,11Sdl,12Mth,13Sth 14 Mwe 15 Swe 16 Mwd 17 Swd
    #[  300 , 10**(-0),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2],
    #[  200 , 10**(-1),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2],
    #[  100 , 10**(-1),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 2, 2, 0, 0],
    #[  100 , 10**(-2),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2]
#    [  300 , 10**(-1),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2],
#    [  200 , 10**(-2),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2],
#    [  100 , 10**(-2),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 2, 2, 0, 0],
#    [  100 , 10**(-3),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2]
#    [  300 , 10**(-2),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2],
#    [  200 , 10**(-3),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2],
#    [  100 , 10**(-3),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 2, 2, 0, 0],
#    [  100 , 10**(-4),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2]
    [ 2000 , 10**(-1),   0,   0 ,    0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 0, 0, 2, 2],
    [  200 , 10**(-3),   0,   0 ,    0,    0,    2,    2,    0,    2,    2,    2,    2,    2, 0, 0, 2, 2],
    [  200 , 10**(-3),   0,   0 ,    0,    0,    2,    2,    2,    0,    2,    2,    2,    2, 0, 0, 2, 2]
]
)

noiseYrelaxationValue = 2
noiseYsaveValue = float(model.log_Y_noise_var.data.exp().sqrt()[0])
model.log_Y_noise_var.data = (noiseYrelaxationValue**2*torch.ones(model.Dout)).log().type(model.dtype)

batch_size=N
start = time.time()


#%%
model.learn(X, XStar, T, Y, Z, n, N, Nmc, batch_size, 0, 0, 0, 0,
            verbose=3,learning_rate=10**(-3),optimisationDescription=optimisationDescription,
            saveStates=None,displayZlearn=True,displayYlearn=True,displayDef=True)#experimentFolder+'/results/states.pkl'
model.log_Y_noise_var.data = (noiseYsaveValue**2*torch.ones(model.Dout)).log().type(model.dtype)
if saveModelObj:
    with open(experimentFolder+'/results/startingModel.pkl', 'wb') as output:
        pickle.dump(model, output)


#%%
yy, yyv, zz, zzv = model.layers[0].predictGivenTheta(X,XStar,T,Nmc,theta_sample=None,forcePhiCompute=True)
plt.scatter(zz.squeeze().data.cpu().numpy(),Z.squeeze().data.cpu().numpy())


#%%
if not additiveDiscr:
    if model.kernel_ard:
        model.layers[0].etaAnisotropy.requires_grad=False
    else:
        model.layers[0].etaAnisotropy.requires_grad=True
print("ani")
print(model.layers[0].etaAnisotropy)
nbStarts = 6#2*D2
#modelLog_Y_noise_varSave = model.log_Y_noise_var.data.clone()
#model.log_Y_noise_var.data = (0.1**2*torch.ones(model.Dout)).log().type(model.dtype)

#model.compute_nelbo(X,XStar,T, Y, Z, Nmc, n, N, batch_size=N)
bestPerf=10000000000
print("initial Nelbo: "+str(bestPerf))
#model.log_Y_noise_var.data = modelLog_Y_noise_varSave
saveTheta_m = model.layers[0].q_theta_m.data.clone()
saveTheta_logv = model.layers[0].q_theta_logv.data.clone()

results = torch.zeros(nbStarts,2*D2+1).type(dtype)

Nmc=50

optimisationDescription = torch.Tensor( # > 1 means fixed, even means display
[#0nIt     1learnRate 2NLB  3NLL, 4DKL, 5gdN, 6NsY, 7NsZ, 8Ret, 9Set,10Rdl,11Sdl,12Mth,13Sth 14 Mwe 15 Swe 16 Mwd 17 Swd
    [   1000  , 10**(-2),    0,   0 ,   0,    0,    2,    2,    2,    2,    2,    2,    0,    2, 2, 2, 2, 2],
    [    250  , 10**(-2),    0,   0 ,   0,    0,    2,    2,    2,    2,    2,    2,    0,    0, 2, 2, 2, 2],
    [  10000  , 10**(-2),    0,   0 ,   0,    0,    2,    2,    2,    2,    2,    2,    2,    2, 2, 2, 0, 0],
    [  10000  , 10**(-3),    0,   0 ,   0,    0,    2,    2,    2,    0,    2,    0,    2,    2, 2, 2, 2, 2]
    #[  100000 , 10**(-3),   0,   0 ,   0,    0,    2,    2,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0]
]
)

for j in range(nbStarts):
    print("New startpoints")
    #with open(experimentFolder+'/results/startingModel.pkl', 'rb') as input:
    #    startingModel = pickle.load(input)
    model.layers[0].etaAnisotropy.data = torch.ones(1).type(dtype)
    model.layers[0].q_theta_m.data = torch.randn(D2).type(dtype)*.5*torch.diag(priorThetaCovRoot)+priorThetaMean
    model.layers[0].q_theta_logv.data = (0.01**2*torch.ones(D2)).log().type(dtype)
    model.layers[0].q_W_delta_m.data.zero_()  #  = torch.zeros_like(model.layers[0].prior_W_delta_m.data).type(dtype)
    model.layers[0].q_W_delta_logv.data.zero_()# = torch.zeros_like(model.layers[0].prior_W_delta_m.data).type(dtype)
    model.layers[0].log_delta_lengthscale.data = (deltaRangeSaved*torch.ones(1, 1).type(dtype)).log().type(model.dtype)
    model.layers[0].log_delta_sigma2.data = (deltaStdSaved**2*torch.ones(1, 1).type(dtype)).log().type(model.dtype)
    model.log_Y_noise_var.data = (0.01**2*torch.ones(model.Dout)).log().type(model.dtype)
    #startingModel.log_Y_noise_var.data = (0.1**2*torch.ones(model.Dout)).log().type(model.dtype)
    model.learn(X, XStar, T, Y, Z, n, N, Nmc, batch_size, 0, 0, 0, 0,
            verbose=2,learning_rate=10**(-3),optimisationDescription=optimisationDescription,
            saveStates=None,displayZlearn=True,displayYlearn=True,displayDef=True)
    perf = float(model.nelbo[0])
    results[j,0] = perf
    results[j,1:(D2+1)] = model.layers[0].q_theta_m.data.clone()
    results[j,(D2+1):(2*D2+1)] = model.layers[0].q_theta_logv.exp().sqrt().data.clone()
    if perf < bestPerf:
        print("New record!")
        bestPerf=perf
        saveBestTheta_m = model.layers[0].q_theta_m.data.clone()
        saveBestTheta_logv = model.layers[0].q_theta_logv.data.clone()
        saveBestMWd = model.layers[0].q_W_delta_m.clone()
        saveBestVWd = model.layers[0].q_W_delta_logv.clone()
        saveBestAni = model.layers[0].etaAnisotropy.data.clone()
        saveBestRd = model.layers[0].log_delta_lengthscale.data.clone()
        
#model.layers[0].q_theta_m.data = saveBestTheta_m
#model.layers[0].q_theta_logv.data = saveBestTheta_logv
if saveModelObj:
        with open(experimentFolder+'/results/bestModel.pkl', 'wb') as output:
                pickle.dump(model, output)
#model.log_Z_noise_var.data = (noiseZsaveValue**2*torch.ones(model.Dout)).log().type(model.dtype)


#%%
print(results)
model.layers[0].q_theta_m.data = saveBestTheta_m
model.layers[0].q_theta_logv.data = saveBestTheta_logv
model.layers[0].q_W_delta_m.data = saveBestMWd
model.layers[0].q_W_delta_logv.data = saveBestVWd
model.layers[0].etaAnisotropy.data = saveBestAni
model.layers[0].log_delta_lengthscale.data = saveBestRd


#%%
end = time.time()

print((end - start)/60)

ym,yv,zm,zv = model.layers[0].predictGivenTheta(X,XStar,T,Nmc,None,forcePhiCompute=True)

#model.layers[0].q_theta_m.data = saveBestTheta_m

plt.scatter(zm.squeeze().data.cpu().numpy(),Z.squeeze().cpu().numpy())
plt.scatter(ym.data.mean(0).squeeze().cpu().numpy(),Y.data.squeeze().cpu().numpy())
plt.show()


#%%
lowerDef = torch.tensor([Z.min(),XStar.min()]).type(dtype)
upperDef = torch.tensor([Z.max(),XStar.max()]).type(dtype)
axialPre = 80
actualGrid = (giveGrid(axialPre,1).type(model.dtype)*(upperDef[1]-lowerDef[1]).unsqueeze(0)+lowerDef[1].unsqueeze(0))
ym,yv,zm,zv = model.layers[0].predictGivenTheta(actualGrid,XStar,T,Nmc,None,forcePhiCompute=True)
plt.scatter(actualGrid.data.cpu().numpy(),ym.mean(0).squeeze().data.cpu().numpy())


#%%
#model.layers[0].etaAnisotropy.data[0] = torch.ones_like(model.layers[0].etaAnisotropy).type(dtype)
model.layers[0].etaAnisotropy.data[0].exp()


#%%
Z.min()


#%%
lowerDef = torch.tensor([Z.min(),XStar.min()]).type(dtype)
#upperDef = torch.tensor([Z.max(),XStar.max()]).type(dtype)
upperDef = torch.tensor([0,XStar.max()]).type(dtype)
axialPre = 80
actualGrid = (giveGrid(axialPre,D1+1).type(model.dtype)*(upperDef-lowerDef).unsqueeze(0)+lowerDef.unsqueeze(0)).squeeze()
Omega_delta_sample = torch.exp(-model.layers[0].log_delta_lengthscale)*2.**0.5*model.layers[0].epsilon_for_Omega_delta_sample
Phi_delta_before_activation=torch.matmul(actualGrid, Omega_delta_sample)
Phi_delta_noStd = torch.cat((torch.sin(Phi_delta_before_activation), torch.cos(Phi_delta_before_activation)), 1)
Phi_delta = Phi_delta_noStd * torch.sqrt(torch.exp(model.layers[0].log_delta_sigma2) / model.layers[0].NRFs)
mean_F_delta = torch.matmul(Phi_delta, model.layers[0].q_W_delta_m)
var_F_delta = torch.matmul(Phi_delta**2, torch.exp(model.layers[0].q_W_delta_logv))
defo = mean_F_delta#+actualGrid[:,0].unsqueeze(1)
CS = plt.contourf(np.linspace(lowerDef[1].item(),upperDef[1].item(),axialPre),np.linspace(lowerDef[0].item(),upperDef[0].item(),axialPre),defo.data.cpu().view(axialPre,axialPre))
plt.clabel(CS,fontsize=9, inline=1,colors="black")


#%%
x1 =(torch.ones(1)*(-1.5)).type(dtype)
x2 =(torch.ones(1)*(-.5)).type(dtype)
x3 =(torch.ones(1)*(2.5)).type(dtype)

XX = torch.cat((x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0)),0)
pb1,pb2,zmXX,zvXX = model.layers[0].predictGivenTheta(XX,XX,model.layers[0].q_theta_m.unsqueeze(0).expand(3,-1),Nmc,None,forcePhiCompute=True)

print(XX)
print(zmXX)
pb1.mean(0)


#%%
width=1
etaGrid = (giveGrid(axialPre,1).type(model.dtype)*(upperDef[0]-lowerDef[0]).unsqueeze(0)+lowerDef[0].unsqueeze(0))
etaGrid1 = etaGrid#((giveGrid(axialPre,1).type(model.dtype)-.5)*width)+zmXX[0].item()
etaGrid2 = etaGrid#((giveGrid(axialPre,1).type(model.dtype)-.5)*width)+zmXX[1].item()
etaGrid3 = etaGrid#((giveGrid(axialPre,1).type(model.dtype)-.5)*width)+zmXX[2].item()


#%%
actualGrid = torch.cat((torch.cat((etaGrid1,x1.unsqueeze(0).expand(axialPre,-1)),1),torch.cat((etaGrid2,x2.unsqueeze(0).expand(axialPre,-1)),1),torch.cat((etaGrid3,x3.unsqueeze(0).expand(axialPre,-1)),1)),0)

Omega_delta_sample = torch.exp(-model.layers[0].log_delta_lengthscale)*2.**0.5*model.layers[0].epsilon_for_Omega_delta_sample
Phi_delta_before_activation=torch.matmul(actualGrid, Omega_delta_sample)
Phi_delta_noStd = torch.cat((torch.sin(Phi_delta_before_activation), torch.cos(Phi_delta_before_activation)), 1)
Phi_delta = Phi_delta_noStd * torch.sqrt(torch.exp(model.layers[0].log_delta_sigma2) / model.layers[0].NRFs)
mean_F_delta = torch.matmul(Phi_delta, model.layers[0].q_W_delta_m)
var_F_delta = torch.matmul(Phi_delta**2, torch.exp(model.layers[0].q_W_delta_logv))
plt.plot((giveGrid(axialPre,1).type(model.dtype)*width).squeeze().cpu().numpy(),(mean_F_delta[0:axialPre,:]).data.squeeze().cpu().numpy())
plt.plot((giveGrid(axialPre,1).type(model.dtype)*width).squeeze().cpu().numpy(),(mean_F_delta[axialPre:(2*axialPre),:]).data.squeeze().cpu().numpy())
plt.plot((giveGrid(axialPre,1).type(model.dtype)*width).squeeze().cpu().numpy(),(mean_F_delta[(2*axialPre):(3*axialPre),:]).data.squeeze().cpu().numpy())


#%%
#np.savetxt(experimentFolder+"/results/warping.csv",mean_F_delta.data.cpu().numpy(),delimiter=";")


#%%
lower1 = torch.tensor([XStar.min(),XStar.min()]).type(model.dtype)
upper1 = torch.tensor([XStar.max(),XStar.max()]).type(model.dtype)
axialPre = 80
x = (giveGrid(axialPre,1).type(model.dtype)*(upper1[0]-lower1[0]).unsqueeze(0)+lower1[0].unsqueeze(0))
meanYGivenTheta,varYGivenTheta,meanZGivenTheta,varZGivenTheta =  model.forward(x, XStar[0:2,:], T[0:2,:], 50)

plt.plot(x.squeeze().cpu().numpy(),meanYGivenTheta.data[:,:,0].t().squeeze().cpu().numpy())
plt.scatter(XStar.squeeze().cpu().numpy(),Z.squeeze().cpu().numpy(),c="gray")
plt.scatter(X.squeeze().cpu().numpy(),Y.squeeze().cpu().numpy())
plt.show()
if model.deepness>0:
    print(model.layers[1].log_lengthscale.data.exp())
    if model.deepness>1:
        print(model.layers[2].log_lengthscale.data.exp())
indices_batchplt = np.random.choice(N, np.min([N,1000]), replace=False)
yy, yyv, zz, zzv = model.forward(X[1:2,:],XStar[indices_batchplt,:],T[indices_batchplt,:],10)
if torch.cuda.is_available():
    if len(zz.squeeze().size())==1:
        plt.scatter(zz.squeeze().data.cpu().numpy(),Z[indices_batchplt,:].squeeze().data.cpu().numpy())
    else:
        plt.scatter(zz.squeeze().mean(0).data.cpu().numpy(),Z[indices_batchplt,:].squeeze().data.cpu().numpy())
else:
    if len(zz.squeeze().size())==1:
        plt.scatter(zz.squeeze().data.numpy(),Z[indices_batchplt,:].squeeze().data.numpy())
    else:
        plt.scatter(zz.squeeze().mean(0).data.numpy(),Z[indices_batchplt,:].squeeze().data.cpu().numpy())
plt.show()


#%%




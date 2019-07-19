from abc import ABCMeta, abstractmethod
import torch

from .base import BaseNet
from torch.distributions import kl_divergence


class CalibrationNet(BaseNet,metaclass=ABCMeta):
    def __init__(self,computer_model,discrepancy,calib_prior,calib_posterior,true_calib=None):
        super(CalibrationNet, self).__init__()
        self.computer_model = computer_model
        self.discrepancy = discrepancy
        self.calib_prior = calib_prior
        self.calib_posterior = calib_posterior
        self.nell_computers = [self.discrepancy.compute_nell,self.computer_model.compute_nell]
        self.true_calib = true_calib
        self.layers = discrepancy.layers

    @abstractmethod
    def phenomenon(self,observationable_inputs,calibration_inputs):
        NotImplemented

    @property
    def likelihood(self):
        return torch.nn.ModuleDict({"discrepancy":self.discrepancy.likelihood,
                                    "computer_model":self.computer_model.likelihood})

    def compute_error(self, Output_pred, Output_true,n_over_m):
        # This function is not used for learning
        name_phenomenon_error = "phenomenon error"
        name_computer_model_error = "computer model error"
        Y_pred = Output_pred[0]
        Z_pred = Output_pred[1]
        Y_true = Output_true[0]
        Z_true = Output_true[1]
        n_over_m_Y = n_over_m[0]
        n_over_m_Z = n_over_m[1]
        if Y_true is not None:
            phenomenon_error = self.discrepancy.compute_error(Y_pred, Y_true,n_over_m_Y)
        else:
            phenomenon_error = torch.zeros(1)
        computer_model_error = self.computer_model.compute_error(Z_pred, Z_true,n_over_m_Z)
        if self.true_calib is not None:
            calib_error = ((self.true_calib - self.calib_posterior.loc)**2).sum().sqrt().item()
            return torch.tensor([phenomenon_error.item(),computer_model_error.item(),calib_error])
        else:
            return torch.tensor([phenomenon_error.item(),computer_model_error.item()])

    def kl_divergence(self): # overwrite the base method to add the kl of the calibration parameter
        total_dkl = super().kl_divergence()
        total_dkl += kl_divergence(self.calib_posterior,self.calib_prior)
        return total_dkl

    def compute_nell(self, Output_pred, Output_true, n_over_m):# accept None 
        Y_pred = Output_pred[0]
        Y_true = Output_true[0]
        n_over_m_Y = n_over_m[0]
        if Y_pred is None:
            nell_Y = 0
        else:
            nell_Y = self.discrepancy.compute_nell(Y_pred, Y_true,n_over_m_Y)
        Z_pred = Output_pred[1]
        Z_true = Output_true[1]
        n_over_m_Z = n_over_m[1]
        if Z_pred is None:
            nell_Z = 0
        else:
            nell_Z = self.computer_model.compute_nell(Z_pred, Z_true,n_over_m_Z)
        return nell_Y + nell_Z

    def forward(self, input,input_nmc_rep=True):
        nmc = self.computer_model.nmc
        X_raw      = input[0]
        X_star_raw = input[1]
        T_raw      = input[2]
        if X_raw is None:
            Y = None
        else:
            if input_nmc_rep:
                X = X_raw.expand(torch.Size([nmc])+X_raw.size())
            else:
                X = X_raw
            theta = self.calib_posterior.rsample(torch.Size([nmc]))
            theta_ex = theta.unsqueeze(-2).expand([nmc,X.size(1),theta.size(-1)]) # batchpoint dim extention
            Y = self.phenomenon(  X   ,theta_ex) # expand because one replication of Y is always evaluated at one 
        if X_star_raw is None or T_raw is None:
            Z = None
        else:
            if input_nmc_rep:
                X_star = X_star_raw.expand(torch.Size([nmc])+X_star_raw.size())
                T      = T_raw.expand(torch.Size([nmc])+T_raw.size())
            else:
                X_star,T = X_star_raw,T_raw
            Z = self.computer_model(torch.cat((X_star,T),-1),input_nmc_rep=False)
        return [Y,Z]
    

    def give_discrepancy_input_output(self, X,Y):
        self.eval()
        nmc = self.discrepancy.nmc
        with torch.no_grad():
            theta = self.calib_posterior.sample([nmc])
            theta_ex = theta.unsqueeze(-2).expand([nmc,X.size(0),theta.size(-1)])
            X_ex = X.unsqueeze(0).expand(torch.Size([theta.size(0)])+X.size())
            Z_Xtheta = self.computer_model(torch.cat((X_ex,theta_ex),-1),input_nmc_rep=False)
            return self.give_discrepancy_child_input_output(X,Z_Xtheta.mean(0),Y)
    

class AdditiveDiscrepancy(CalibrationNet):
        def __init__(self,*args,**kwargs):
            super(AdditiveDiscrepancy, self).__init__(*args,**kwargs)

        def phenomenon(self,X,T):
            delta = self.discrepancy(X,input_nmc_rep=False)
            eta = self.computer_model(torch.cat((X,T),-1),input_nmc_rep=False)
            return eta + delta
        
        def give_discrepancy_child_input_output(self,X,Z_Xtheta,Y):
            return X,Y-Z_Xtheta

        #def phenomenon_initialize(self,X,Z_Xtheta,Y):
        #    self.discrepancy.initialize(X,Y-Z_Xtheta)


class GeneralDiscrepancy(CalibrationNet):
        def __init__(self,computer_model,discrepancy,calib_prior,calib_posterior):
            super(GeneralDiscrepancy, self).__init__(computer_model,discrepancy,calib_prior,calib_posterior)

        def phenomenon(self,X,T):
            eta = self.computer_model(torch.cat((X,T),-1),input_nmc_rep=False)
            return self.discrepancy(torch.cat((eta,X),-1),input_nmc_rep=False)
        
        #def phenomenon_initialize(self,X,Z_Xtheta,Y):
        #    self.discrepancy.initialize(torch.cat((Z_Xtheta,X),-1),Y)
        def give_discrepancy_child_input_output(self,X,Z_Xtheta,Y):
            return torch.cat((Z_Xtheta,X),-1),Y
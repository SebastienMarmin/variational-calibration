import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import vcal
from vcal.utilities import regress_linear
import matplotlib.pyplot as plt
import matplotlib.lines as mli
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

if __name__ == "__main__":

    # test all combinations of arguments
    N=5
    d=1
    Y = torch.randn(N)
    X = torch.linspace(0, 1,N).unsqueeze(1)   # points

    sigma = np.sqrt(1)                        # écart-type à priori
    NRFs  = 200                               # nombre de features
    theta = 0.05                              # portée radiale
    tau  = np.sqrt(1.0/10.0)                      # écart-type du bruit


    Omega  = np.sqrt(2)/theta*torch.randn(d,NRFs)
    sp = torch.mm(X, Omega)
    Phi = sigma/np.sqrt(NRFs)*torch.cat((sp.sin(),sp.cos()),1)

    Lambda_inversed_list = [True,False]
    Lambda_rooted_list = [True,False]
    Gamma_inversed_list = [True,False]
    Gamma_rooted_list = [True,False]
    Lambda_dim_list = [-1,0,1,2]
    Gamma_dim_list = [-1,0,1,2]
    mu_dim_list = [-1,0,1]

    for Lambda_dim in Lambda_dim_list:
        if Lambda_dim == -1:
            Lambda=None
        elif Lambda_dim == 0:
            Lambda = tau**2*torch.tensor([[1.0]]).type(X.type())
        elif Lambda_dim == 1:
            Lambda = tau**2*torch.ones(N,1).type(X.type())
        elif Lambda_dim == 2:
            Lambda = tau**2*torch.eye(N).type(X.type())
        for Gamma_dim in Gamma_dim_list:
            if Gamma_dim == -1:
                Gamma=None
            elif Gamma_dim == 0:
                Gamma = torch.tensor([[1.0]]).type(X.type())
            elif Gamma_dim == 1:
                Gamma = torch.ones(2*NRFs,1).type(X.type())
            elif Gamma_dim == 2:
                Gamma = torch.eye(2*NRFs).type(X.type())
            for mu_dim in mu_dim_list:
                if mu_dim == -1:
                    mu=None
                elif mu_dim == 0:
                    mu = torch.tensor([0]).type(X.type())
                elif mu_dim == 1:
                    mu = torch.zeros([2*NRFs]).type(X.type())
                for Lambda_inversed in Lambda_inversed_list:
                    if Lambda_inversed and Lambda is not None:
                        Lambda_i = Lambda.inverse() if Lambda_dim == 2 else 1/Lambda
                    else:
                        Lambda_i = Lambda
                    for Lambda_rooted in Lambda_rooted_list:
                        if Lambda_rooted and Lambda is not None:
                            Lambda_s = Lambda_i.sqrt()
                        else:
                            Lambda_s = Lambda_i
                        for Gamma_inversed in Gamma_inversed_list:
                            if Gamma_inversed and Gamma is not None:
                                Gamma_i = Gamma.inverse() if Gamma_dim == 2 else 1/Gamma
                            else:
                                Gamma_i = Gamma
                            for Gamma_rooted in Gamma_rooted_list:
                                if Gamma_rooted and Gamma is not None:
                                    Gamma_s = Gamma_i.sqrt()
                                else:
                                    Gamma_s = Gamma_i

                                combination = "mu_dim=%d,Lambda_dim=%d,Gamma_dim=%d,Lambda_inversed=%d,Gamma_inversed=%d,Lambda_rooted=%d,Gamma_rooted=%d" % (int(mu_dim),int(Lambda_dim),int(Gamma_dim),int(Lambda_inversed),int(Gamma_inversed),int(Lambda_rooted),int(Gamma_rooted))
                                print(combination)
                                meanW,varW,_=regress_linear(Phi,Y,Lambda=Lambda_s,Gamma=Gamma,mu=mu,Lambda_inversed=Lambda_inversed,Gamma_inversed=Gamma_inversed,Lambda_rooted=Lambda_rooted,Gamma_rooted=Gamma_rooted)

                                axialPre=40
                                x = torch.linspace(0,1,axialPre).unsqueeze(1)
                                spTest = torch.mm(x, Omega)
                                PhiTest= sigma/np.sqrt(NRFs)*torch.cat((spTest.sin(),spTest.cos()),1)
                                y_mean = PhiTest.mv(meanW)
                                y_var  = PhiTest.mm(varW.mm(PhiTest.t()))+torch.diag(tau**2*torch.ones(axialPre))
                                y_95 = +2*torch.diag(y_var).sqrt()+y_mean
                                y_05 = -2*torch.diag(y_var).sqrt()+y_mean

                                plt.plot(x.numpy(),y_mean.numpy())
                                plt.plot(x.numpy(),y_05.numpy(),c="gray")
                                plt.plot(x.numpy(),y_95.numpy(),c="gray")
                                plt.scatter(X,Y)
    plt.title("Everything should perfectly overlay")
    plt.show()




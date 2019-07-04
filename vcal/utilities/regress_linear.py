import torch

def regress_linear(X,Y,Lambda=None,Gamma=None,mu=None,Lambda_inversed=False,Gamma_inversed=False,Lambda_rooted=False,Gamma_rooted=False):
    # X: input
    # Y: output vector (one-column matrix accepted)
    # Lambda: prior covariance matrix of the noise (accept vector if diagonal, scalar if homoscedastic, defaut [.1])
    # Gamma: prior covariance of the weigths beta (accept vector if diagonal, scalar if homoscedastic, default: [1])
    # mu: prior mean vector of the weigths. Scalar accepted for constant mean (default: [0]). One-column matrix accepted.
    # Lambda_inversed/Gamma_inversed: boolean telling if Lambda and Gamma are already inversed (i.e. as precision matrix)
    # Lambda_rooted/Gamma_rooted: boolean telling if Lambda and Gamma are provided as a lower triangular covariance/precision root (or (inverse) standard deviations in diagonal/homoscedastic cases)
    if Lambda is None:
        Lambda = torch.tensor([.1]).type(X.type())
        Lambda_inversed = False
        Lambda_rooted = False
    if isinstance(Lambda,float):
        Lambda = Lambda * torch.ones(1).type(X.type())
    if Gamma is None:
        Gamma = torch.tensor([1.0]).type(X.type())
        Gamma_inversed = False
        Gamma_rooted = False

    Ysh = Y.shape
    if len(Ysh)==2 and Ysh[1]==1:
        Y = Y.squeeze()
    if mu is not None:
        if len(mu.size()) == 0:
            mu = mu.unsqueeze(-1)
        if mu.size(-1) == 1 or mu.size(0) == 1:
            mu = mu.view(-1,1).squeeze(-1)
            lemu = mu.shape[-1]
        else:
            raise NotImplemented
    p = X.shape[1]
    Xt = X.t()
    Ls = Lambda.shape
    if len(Ls)==1:# if provide only variances as a vector or scalar
        if Lambda_rooted:
            Lambda_r = Lambda # *_M1 means "inverse of *"
            if Lambda_inversed:
                Lambda_r_M1 = Lambda_r
            else:
                Lambda_r_M1 = 1/Lambda_r
            Phit = Xt*Lambda_r_M1 #Xt_LambdaM1_X = LambdaM1*Xt.mm(X)
            Lambda_r_MY = Lambda_r_M1*Y #Xt_LambdaM1_X = LambdaM1*Xt.mm(X)
        else:
            if Lambda_inversed:
                Lambda_M1 = Lambda
            else:
                Lambda_M1 = 1/Lambda
            Psit = Xt*Lambda_M1
    else:
        if Lambda_rooted:
            if Lambda_inversed:
                Lambda_r_M1 = Lambda
                Phit = Xt.mm(Lambda_r_M1.t())
                Lambda_r_MY = Lambda.mv(Y)
            if not Lambda_inversed:
                Phit = torch.triangular_solve(X,Lambda,upper=False, transpose=False).t() # TODO handle upper
                Lambda_r_MY = torch.triangular_solve(Y,Lambda,upper=False, transpose=False) # TODO handle upper
        else:
            if Lambda_inversed:
                Psit = Xt.mm(Lambda)#Xt_LambdaM1_X = Xt.mm(Lambda.mm(X))
            else: # these case should almost never happen as one usually knows the inverse of the prior covariance of the noise
                Lambda_r = torch.cholesky(Lambda,upper=False)
                Psit = (torch.cholesky_solve(X,Lambda_r,upper=False)).t()
            #Lambda_r_M1 = Lambda
            #Phit = Xt.mm(Lambda_r_M1.t())
    if Lambda_rooted:
        Xt_LambdaM1_X = Phit.mm(Phi)
        Xt_LambdaM1_Y = Phit.mv(Lambda_r_MY)
    else:
        Xt_LambdaM1_X = Psit.mm(X)
        Xt_LambdaM1_Y = Psit.mv(Y)
    Gs = Gamma.shape
    if len(Gs)==1:# if cov is diagonal provided in a vector or scalar
        if Gamma_inversed:
            GammaM1 = Gamma
        else:
            GammaM1 = 1/Gamma
        if mu is None:
            GammaM1_mu = 0
        else:
            GammaM1_mu = GammaM1*mu
        if Gs[0]==1:
            GammaM1_mat = GammaM1*torch.eye(p).type(X.type())
        else:
            GammaM1_mat = torch.diag(GammaM1)
    else:
        if Gamma_inversed:
            GammaM1_mat = Gamma
        else:
            Gamma_r = torch.cholesky(Gamma,upper=False)# these case should almost never happen as one usually knows the inverse of the prior covariance of the weigths
            GammaM1_mat =  torch.cholesky_inverse(Gamma_r,upper=False)
        if mu is None:
            GammaM1_mu = 0
        else:
            if lemu==1:
                GammaM1_mu = GammaM1_mat.sum(1)*mu
            else:
                GammaM1_mu = GammaM1_mat.mv(mu)
    
    precBeta = Xt_LambdaM1_X+GammaM1_mat
    # Main inversion (computation of the posterior covariance of the weigths)
    precBeta_r = torch.cholesky(precBeta,upper=False)
    covBeta = torch.cholesky_inverse(precBeta_r,upper=False)        
    meanBeta = covBeta.mv(Xt_LambdaM1_Y+GammaM1_mu)
    return meanBeta,covBeta,precBeta_r


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.lines as mli
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import numpy as np
    # test all combinations of arguments
    N=5
    d=1
    Y = torch.randn(N)
    X = torch.linspace(0, 1,N).unsqueeze(1)   # points

    sigma = np.sqrt(1)                        # écart-type à priori
    NRFs  = 200                               # nombre de features
    theta = 0.05                              # portée radiale
    tau  = np.sqrt(1/10)                      # écart-type du bruit


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
            Lambda = tau**2*torch.tensor([1.0]).type(X.type())
        elif Lambda_dim == 1:
            Lambda = tau**2*torch.ones(2*NRFs).type(X.type())
        elif Lambda_dim == 2:
            Lambda = tau**2*torch.eye(2*NRFs).type(X.type())
        for Gamma_dim in Gamma_dim_list:
            if Gamma_dim == -1:
                Gamma=None
            elif Gamma_dim == 0:
                Gamma = torch.tensor([1.0]).type(X.type())
            elif Gamma_dim == 1:
                Gamma = torch.ones(2*NRFs).type(X.type())
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
                        Lambda = 1/Lambda
                    for Lambda_rooted in Lambda_rooted_list:
                        if Lambda_rooted and Lambda is not None:
                            Lambda = Lambda.sqrt()
                        for Gamma_inversed in Gamma_inversed_list:
                            if Gamma_inversed and Gamma is not None:
                                Gamma = 1/Gamma
                            for Gamma_rooted in Gamma_rooted_list:
                                if Gamma_rooted and Gamma is not None:
                                    Gamma = Gamma.sqrt()

                                combination = "mu_dim=%d,Lambda_dim=%d,Gamma_dim=%d,Lambda_inversed=%d,Gamma_inversed=%d,Lambda_rooted=%d,Gamma_rooted=%d" % (int(mu_dim),int(Lambda_dim),int(Gamma_dim),int(Lambda_inversed),int(Gamma_inversed),int(Lambda_rooted),int(Gamma_rooted))
                                print(combination)
                                meanW,varW,_=regress_linear(Phi,Y,Lambda=None,Gamma=None,mu=mu,Lambda_inversed=Lambda_inversed,Gamma_inversed=Gamma_inversed,Lambda_rooted=Lambda_rooted,Gamma_rooted=Gamma_rooted)

                                axialPre=100
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

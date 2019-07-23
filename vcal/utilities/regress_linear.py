import torch


def regress_linear(X,Y,Lambda=None,Gamma=None,mu=None,Lambda_inversed=False,Gamma_inversed=False,Lambda_rooted=False,Gamma_rooted=False): # TODO batch version
    # X: input [...,n,d]
    # Y: output vector (no one-column matrix) [...,n]
    # Lambda: prior covariance matrix of the noise (default [[.1]])
    #           - [...,n,n] or
    #           - [...,n,1] if diagonal, or
    #           - [...,1,1] if diagonal and homoscedastic.
    #           - float will be undersood as diagonal and homoscedastic.
    # Gamma: prior covariance [d,d] of the weigths beta (again, one column if diagonal, one element if homoscedastic)
    #         (default [[1]])
    # mu: prior mean vector of the weigths [...,n] or [...,1] for constant mean or None if centered weights distribution.
    # Lambda_inversed/Gamma_inversed: boolean telling if Lambda and Gamma are inversed (i.e. as precision matrix)
    # Lambda_rooted/Gamma_rooted: boolean telling if Lambda and Gamma are provided as a lower triangular covariance/precision root (or (inverse) standard deviations in diagonal/homoscedastic cases)
    if Lambda is None:
        Lambda = torch.tensor([[.1]]).type(X.type())
        Lambda_inversed = False
        Lambda_rooted = False
    elif isinstance(Lambda,float) or isinstance(Lambda,int):
        Lambda = Lambda * torch.ones(1,1).type(X.type())
    if Gamma is None:
        Gamma = torch.tensor([[1.0]]).type(X.type())
        Gamma_inversed = False
        Gamma_rooted = False
    elif isinstance(Gamma,float) or isinstance(Gamma,int):
        Gamma = Gamma * torch.ones(1,1).type(X.type())

    
    if mu is not None:
        lemu = mu.shape[-1]
        
    p = X.shape[-1]
    Xt = X.transpose(-1,-2)
    if Lambda.size(-1)==1:# Lambda detected as diagonal
        Lambda_sq = Lambda.squeeze(-1)
        if Lambda_rooted:
            Lambda_r = Lambda_sq # *_M1 means "inverse of *"
            if Lambda_inversed:
                Lambda_r_M1 = Lambda_r
            else:
                Lambda_r_M1 = 1/Lambda_r
            Phit = Xt*Lambda_r_M1 #Xt_LambdaM1_X = LambdaM1*Xt.mm(X)
            Lambda_r_MY = Lambda_r_M1*Y #Xt_LambdaM1_X = LambdaM1*Xt.mm(X)
        else:
            if Lambda_inversed:
                Lambda_M1 = Lambda_sq
            else:
                Lambda_M1 = 1/Lambda_sq
            Psit = Xt*Lambda_M1
    else:# Lambda not diagonal
        if Lambda_rooted:
            if Lambda_inversed:
                Lambda_r_M1 = Lambda
                Phit = Xt.mm(Lambda_r_M1.t(-1,-2))
                Lambda_r_MY = Lambda.mv(Y)
            if not Lambda_inversed:
                Phit = torch.triangular_solve(X,Lambda,upper=False, transpose=False).transpose(-1,-2) # TODO handle upper
                Lambda_r_MY = torch.triangular_solve(Y,Lambda,upper=False, transpose=False) # TODO handle upper
        else:
            if Lambda_inversed:
                Psit = Xt.mm(Lambda)
            else: # these case should almost never happen as one usually knows the inverse of the prior covariance of the noise
                Lambda_r = torch.cholesky(Lambda,upper=False)
                Psit = (torch.cholesky_solve(X,Lambda_r,upper=False)).transpose(-1,-2)
    if Lambda_rooted:
        Xt_LambdaM1_X = Phit.mm(Phi)
        Xt_LambdaM1_Y = Phit.mv(Lambda_r_MY)
    else:
        Xt_LambdaM1_X = Psit.mm(X)
        Xt_LambdaM1_Y = Psit.mv(Y)
    if Gamma.size(-1)==1:# if cov is diagonal provided in a vector or scalar
        Gamma_sq = Gamma.squeeze(-1)
        if Gamma_inversed:
            GammaM1 = Gamma_sq
        else:
            GammaM1 = 1/Gamma_sq
        if mu is None:
            GammaM1_mu = 0
        else:
            GammaM1_mu = GammaM1*mu
        if Gamma.size(-2)==1:
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
















"""

def regress_linear(X,Y,Lambda=None,Gamma=None,mu=None,Lambda_inversed=False,Gamma_inversed=False,Lambda_rooted=False,Gamma_rooted=False):# TODO batch version
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
"""


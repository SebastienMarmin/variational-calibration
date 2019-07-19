import math

import torch
from torch import tril, matmul
from .import constraints
from .distribution import Distribution
from .utils import _standard_normal, lazy_property

from numpy import log as np_log, pi as np_pi
log2pi = np_log(2*np_pi)

class BlockCovarianceMatrix(torch.nn.Module):
    def __init__(self, d1,d2,batch_dims = (),identic_row=False,identic_col=False,dependent_row=False,dependent_col=False,initial_stddev=1.0,requires_grad=True,is_param=True):
        self._identic_row = identic_row
        self._dependent_row = dependent_row
        self._identic_col = identic_col
        self._dependent_col = dependent_col
        self.event_shape = torch.Size(batch_dims+(d1,d2))
        self._has_tril = dependent_row or dependent_col
        if (identic_row and dependent_row) or (identic_col and dependent_col) :
            print("erre")
            raise NotImplementedError("Correlated components but with identic distrib are not supported.")

        self._iid_row = identic_row and not dependent_row
        self._iid_col = identic_col and not dependent_col

        self.id_row  = not identic_row and not dependent_row
        self.id_col  = not identic_col and not dependent_col
        gen_row = not identic_row and dependent_row # Equals to dependent_row, treat homoscedastic case?
        gen_col = not identic_col and dependent_col
        if self._iid_row and self._iid_col:
            param_size = (1)    # common stddev
        elif id_row and id_row: #idependent but not identic
            param_size = (d1,d2) # component wise stddev
        elif gen_row and gen_col: # general covariance matrix 
            param_size = (d1*d2,d1*d2)
        elif self._iid_row and id_col:
            param_size = (d2)
        elif self._iid_row and gen_col:
            param_size = (d2,d2) # common covariance matrix root
        elif id_row and self._iid_col:
            param_size = (d1)
        elif id_row and gen_col:
            param_size = (d1,d2,d2) # batch matrix root
        elif gen_row and self._iid_col:
            param_size = (d1,d1)
        else :# gen_row and id_col
            param_size = (d2,d1,d1)

        data = torch.zero(*(batch_dims+param_size))
        self.param = torch.nn.Parameter(data,requires_grad)
        self.set_diagonal(initial_stddev)

    def get_diagonal(self):
        if self._has_tril:
            return torch.diagonal(self.param,dim1=-2,dim2=-1)
        else:
            return self.param
    def set_diagonal(value):
        if self._has_tril:
            self.diagonal(self.param,offset=0,dim1=-2,dim2=-1).fill_(value)
        else:
            self.param.fill_(value)
    def get_variances(self):
        if self._has_tril:
            return (tril(self.param)**2).sum(-1)
        else:
            return self.param**2
    def get_stddevs(self):
        if self._has_tril:
            return (tril(self.param)**2).sum(-1).sqrt()
        else:
            return self.param
    def set_stddev(value):
        if self._has_tril:
            stddevs = self.get_stddevs()
            self.param *= value/stddevs.unsqueeze(-1)
        else:
            self.param.fill_(value)
    
    def set_covariances(C):
        if C._iid_row and C._iid_col:
            self.param.zeros_()
            self.set_diagonal(C.param.item())
        elif C._dependent_row and C._dependent_col:
            if self._dependent_row and self._dependent_col:
                self.param = C.param.detach().clone()
            elif not self._has_tril:
                va = C.get_variances().view(self.d1,self.d2).detach().clone()
                if not self._identic_col and not self._identic_row:
                    self.param = va.sqrt()
                elif not self._identic_col and self._identic_row:
                    self.param = va.mean(-2).sqrt()
                elif self._identic_col and not self._identic_row:
                    self.param = va.mean(-1).sqrt()
                else:
                    self.param = va.mean(-2).mean(-1).sqrt()
            else:
                raise NotImplemented("")# TODO
    #def set_tril(L):
    def L_times(self,X):
        if self._diagonal:
            return self.parameter.squeeze(-2).unsqueeze(-1)*X
        else:
            if self._iid_row and self._iid_col:
                return self.param*X
            elif self._id_row and self._iid_col:
                1+1

                
        #if C._iid_row and C._iid_col:

class CovarianceMatrix(torch.nn.Module):
    def __init__(self, d=None,data=None,initial_stddev=1.0,requires_grad=True,is_param=True):
        if d is None:
            if data is None:
                d = 1
            else:
                d = data.size(-1)
        self.d = d
        if data is None:
            data = initial_stddev*torch.ones(1,d) # diagonal heteroscedastic
        self._diagonal, self._homoscedastic = infer_state_from_data(d,data.size())
        super(CovarianceMatrix, self).__init__()
        if is_param:
            self.parameter=torch.nn.Parameter(data,requires_grad)
        else:
            self.parameter=data
            self.parameter.requires_grad = requires_grad
    
    def __repr__(self):
        if self._diagonal:
            add_diag="Diagonal "
        else:
            add_diag="Full "
        if self._homoscedastic:
            add_homosc = "homoscedastic "
        else:
            add_homosc = ""
        return add_diag+add_homosc+"covariance root, dimension "+ str(self.d)+". " + self.parameter.__repr__()



    @property
    def data(self):
        return self.parameter.data
    @data.setter
    def data(self,data):
        self._diagonal, self._homoscedastic = infer_state_from_data(self.d,data.size())
        self.parameter.data = data

    def detach(self):
        self.parameter.detach()

    @property
    def shape(self):
        return self.parameter.shape

    def size(self,*args):
        return self.parameter.size(*args)

    @property
    def diagonal(self):
        return self._diagonal
    @diagonal.setter
    def diagonal(self, newly_diagonal:bool):
        if self._diagonal:
            if not newly_diagonal:
                scale = give_inner_data(self.d,self.data,True,diagonal=False,homoscedastic=self._homoscedastic,init_with_homosc=self._homoscedastic)
            else:
                return
        else:
            if newly_diagonal:
                scale = give_inner_data(self.d,self.data,False,diagonal=True,homoscedastic=self._homoscedastic,init_with_homosc=self._homoscedastic)
            else:
                return
        self._diagonal = newly_diagonal
        self.data = scale.detach()
    @property
    def homoscedastic(self):
        return self._homoscedastic
    @homoscedastic.setter
    def homoscedastic(self, newly_homoscedastic:bool):
        if self._homoscedastic:
            if not newly_homoscedastic:
                scale = give_inner_data(self.d,self.data,self._diagonal,self._diagonal,False,init_with_homosc=self._homoscedastic)
            else:
                return
        else:
            if newly_homoscedastic:
                scale = give_inner_data(self.d,self.data,self._diagonal,self._diagonal,True,init_with_homosc=self._homoscedastic)
            else:
                return
        self._homoscedastic = newly_homoscedastic
        self.data = scale.detach()

    def _get_diags(self):
        if self._diagonal:
            if self._homoscedastic:
                    batch_shape = self.parameter.squeeze(-2).squeeze(-1).size()
                    diags_shape  = batch_shape+torch.Size([1,self.d])
                    diags = self.parameter.expand(diags_shape)
            else: # most cases
                    diags = self.parameter
        else:
            diags = torch.diagonal(self.parameter,dim1=-2,dim2=-1).unsqueeze(-2)
        return diags

    @property
    def tril(self): # for pre or post multiply, use times_L and L_times, faster for diagonal cases.
        parameter=self.parameter
        if self._diagonal:
            diags = self._get_diags()
            return torch.diag_embed(diags.squeeze(-2),offset=0,dim1=-2,dim2=-1)
        else:
            scale = torch.tril(parameter)
            if self._homoscedastic: # rare usage (forcing constant variances but with full covariance)
                variances = (scale**2).sum(-1)
                stds = variances.sqrt()
                common_std = variances.mean(-1).sqrt() # reduced to batch dims
                return scale/(stds/common_std.unsqueeze(-1)).unsqueeze(-1)
            else:
                return scale
    
    @tril.setter
    def tril(self,X):
        if self._diagonal:
            self.diagonal = False
            target_shape = self.parameter.shape
            scale = X.expand(target_shape)
            self.parameter.data = scale
            self.diagonal = True # compute and keep only the variances
            pass
        else:
            target_shape = self.parameter.shape
            scale = X.expand(target_shape)
            self.parameter.data = scale




    def times_L(self,X):
        if self._diagonal:
            return X*self.parameter
        else:
            return torch.matmul(X,self.tril)

    def L_times(self,X):
        if self._diagonal:
            return self.parameter.squeeze(-2).unsqueeze(-1)*X
        else:
            L = self.tril
            return torch.matmul(L,X)

    def L_inverse_times(self,X,transpose=False):# return L^-1 X or L^-1T X
        if self._diagonal:
            return 1/self.parameter.squeeze(-2).unsqueeze(-1)*X
        else:
            if self._homoscedastic:# rare case
                tril = self.tril
                common_std = tril[...,0,0] # for benefiting from the unitriangular option
                return 1/common_std*torch.triangular_solve(X,tril/common_std,upper=False,transpose=transpose,unitriangular=True)[0]
            else:
                return torch.triangular_solve(X,self.parameter,upper=False,transpose=transpose)[0]


    @property
    def covariance(self):
        scale = self.tril
        if self._diagonal:
            return scale**2
        else:
            return torch.matmul(scale,scale.transpose(-1, -2))


    @property
    def adjoint_covariance(self): # L^T L and not L L^T
        scale = self.tril
        if self._diagonal:
            return scale**2
        else:
            return torch.matmul(scale.transpose(-1, -2),scale)

    @property
    def variance(self):
        if self._diagonal:
            return (self._get_diags()**2).squeeze(-2)
        else:
            return (torch.tril(self.parameter)**2).sum(-1)

    @property
    def stddev(self):
        if self._diagonal:
            return self._get_diags().squeeze(-2)
        else:
            return (torch.tril(self.parameter)**2).sum(-1).sqrt()
    @stddev.setter
    def stddev(self,sigma):
        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = torch.Tensor([sigma])
        elif len(sigma.size())==0:
            sigma = sigma.unsqueeze(0)
        if self._diagonal and self._homoscedastic and len(sigma)>1:
            raise print("error") # TODO
        target_shape = self.parameter.shape
        scale_factor = sigma.unsqueeze(-2).expand(target_shape)/self.stddev.unsqueeze(-2)
        self.parameter.data = self.parameter.data.clone().detach()* scale_factor.detach()

    @property
    def adjoint_variance(self):  # diag(L^T L) and not diag(L L^T)
        if self._diagonal:
            return (self._get_diags()**2).squeeze(-2)
        else:
            return (torch.tril(self.parameter)**2).sum(-2)

    @property
    def adjoint_stddev(self):  # diag(L^T L) and not diag(L L^T)
        if self._diagonal:
            return self._get_diags().squeeze(-2)
        else:
            return (torch.tril(self.parameter)**2).sum(-2).sqrt()

    def set_iid(self,std=1.0):
        save_diag = self.diagonal
        save_homoscedastic = self.homoscedastic
        self.diagonal = True
        self.homoscedastic = True
        self.parameter.data[...,0,0] = std
        self.diagonal = save_diag
        self.homoscedastic = save_homoscedastic

    @property
    def log_det(self):
        return 2*self._get_diags().abs().log().sum(-1).squeeze(-1)


    def detach_clone(self,C,keep_iid=True,keep_homoscedastic=True,keep_batch_shape=True):
        if keep_iid:
            new_iid = self._diagonal
        else:
            new_iid = C._diagonal
        if keep_homoscedastic:
            new_homoscedastic = self._homoscedastic
        else:
            new_homoscedastic = C._homoscedastic
        self._homoscedastic = new_homoscedastic
        self._diagonal = new_iid
        if keep_batch_shape:
            batch_shape = self.parameter.data.shape[:-2]
            target_shape = batch_shape + C.parameter.shape[-2:]
            input_data = C.parameter.expand(target_shape).detach().clone()
        else:
            input_data = C.parameter.detach().clone()
        new_data = give_inner_data(self.d,input_data,C._diagonal,new_iid,new_homoscedastic,C._homoscedastic)
        self.parameter.data = new_data.detach()


def old_block_mm(W1,W2,inv=False,A_inv=None,fro=False,W2_deterministic_matrix=False,B_force_row_order=True):
    A = W1.scale #if W1.all_independent else tril(W1.scale)
    w2_de = W2_deterministic_matrix 
    B = W2 if w2_de else (W2.scale if W2.all_independent else tril(W2.scale))
    d1 = W1.nrow
    d2 = W1.ncol# TODO check W2 ?
    A_full_indep = W1.all_independent
    B_full_indep = False if w2_de else W2.all_independent
    A_only_col_dep = W1.only_col_dep
    B_only_col_dep = False if w2_de else W2.only_col_dep
    A_only_row_dep = W1.only_row_dep
    B_only_row_dep = False if w2_de else W2.only_row_dep
    if fro and A_full_indep:
        if A.shape[-1]==1 and A.shape[-2]==1:
            sumB = frobe_Norm(B,d1,d2,B_full_indep,B_only_col_dep,B_only_row_dep)# to avoid adding multiple time the same quantity in cases of same variances or same covariances
            A2 = A.squeeze(-1).squeeze(-1)**2
            return sumB/A2 if inv else sumB*A2
        if B_full_indep and B.shape[-1]==1 and B.shape[-2]==1:
            #print("her")
            Atri = 1/A if inv else A
            sumA = (Atri.expand(*Atri.shape[:-2],d1,d2)**2).sum((-2,-1))
            return B.squeeze(-1).squeeze(-1)**2*sumA
    if A_full_indep:
        Ar = A.expand(*A.shape[:-2],d1,d2).contiguous()
        if B_full_indep:
            Au = Ar
            if fro:
                sumDim = (-2,-1)
        elif B_only_col_dep:
            Au = Ar.unsqueeze(-1)
            if fro:
                sumDim = (-3,-2,-1)
        elif B_only_row_dep:
            Au = Ar.transpose(-2,-1).unsqueeze(-1)
            if fro:
                sumDim = (-3,-2,-1)
        else:# full correlation for B
            Au = Ar.view(*A.shape[:-2],d1*d2,1)
            if fro:
                sumDim = (-2,-1)
        AB = B/Au if inv else B*Au
    elif A_only_col_dep:
        if B_full_indep:
            Ai = (W1.get_tril_inverse() if A_inv is None else A_inv) if inv else A
            Br = B.expand(*B.shape[:-2],d1,d2)
            Bu = Br.unsqueeze(-2)
            AB = Bu*Ai
            if fro:
                sumDim = (-3,-2,-1)
        elif B_only_col_dep:
            AB = torch.triangular_solve(B,tril(A),upper=False,transpose=False)[0]
            #AB = matmul(Ai,B)
            if fro:
                sumDim = (-3,-2,-1)
        elif B_only_row_dep:
            Ai = (W1.get_tril_inverse() if A_inv is None else A_inv) if inv else A
            Au = Ai.unsqueeze(-1)
            Bu = B.transpose(-3,-2).unsqueeze(-3)
            AB = Bu*Au
            if fro:
                sumDim = (-4,-3,-2,-1)
        else:
            Bu = B.view(*B.shape[:-2]+(d1,d2,-1))# d1*d2 most of the time
            AB = torch.triangular_solve(Bu,tril(A),upper=False,transpose=False)[0]
            #AB = matmul(Ai,Bu)
            if fro:
                sumDim = (-3,-2,-1)
    elif A_only_row_dep:
        if B_full_indep:
            Ai = (W1.get_tril_inverse() if A_inv is None else A_inv) if inv else A
            Br = B.expand(*B.shape[:-2],d1,d2)
            Bu = Br.transpose(-2,-1).unsqueeze(-2)
            AB = Bu*Ai
            if fro:
                sumDim = (-3,-2,-1)
        elif B_only_col_dep:
            Ai = (W1.get_tril_inverse() if A_inv is None else A_inv) if inv else A
            Au = Ai.unsqueeze(-1)
            Bu = B.transpose(-3,-2).unsqueeze(-3)
            AB = Bu*Au
            if fro:
                sumDim = (-4,-3,-2,-1)
        elif B_only_row_dep:
            #AB = matmul(Ai,B)
            AB = torch.triangular_solve(B,tril(A),upper=False,transpose=False)[0]
            if fro:
                sumDim = (-3,-2,-1)
        else:
            if B_force_row_order:
                Br = B.view(*B.shape[:-2],d1,d2,-1)
                Bu = Br.transpose(-2,-3)
            else:
                Bu = B.view(*B.shape[:-2],d2,d1,-1)
            #AB = matmul(Ai,Bu)
            AB = torch.triangular_solve(Bu,tril(A),upper=False,transpose=False)[0]
            if fro:
                sumDim = (-3,-2,-1)
    else:
        if B_full_indep:
            Br = B.expand(*B.shape[:-2],d1,d2).contiguous()
            Bu = Br.view(*B.shape[:-2],1,d1*d2)
            Ai = (W1.get_tril_inverse() if A_inv is None else A_inv) if inv else A
            AB = Bu*Ai
            if fro:
                sumDim = (-2,-1)
        elif B_only_col_dep:
            Ai = (W1.get_tril_inverse() if A_inv is None else A_inv) if inv else A
            Ar = Ai.view(*Ai.shape[:-2],d1*d2,d1,d2)
            Au = Ar.unsqueeze(-2)
            Bu = B.unsqueeze(-4)
            AB = matmul(Au,Bu).squeeze(-2)# TODO block tril_inv?
            if fro:
                sumDim = (-3,-2,-1)
        elif B_only_row_dep:
            Ai = (W1.get_tril_inverse() if A_inv is None else A_inv) if inv else A
            Ar = Ai.view(*A.shape[:-2],d1*d2,d1,d2).transpose(-1,-2)
            Au = Ar.unsqueeze(-2)
            Bu = B.unsqueeze(-4)
            AB = matmul(Au,Bu).squeeze(-2)# TODO block tril_inv?
            if fro:
                sumDim = (-3,-2,-1)
        else:
            #AB = matmul(Ai,B)
            AB = torch.triangular_solve(B,tril(A),upper=False,transpose=False)[0]
            if fro:
                sumDim = (-2,-1)
    if fro:
        return (AB**2).sum(sumDim)
    else:
        return AB


def L1_inv_L2_frob(W1,W2):
    B = W2.scale if W2.all_independent else tril(W2.scale)
    d1 = W1.nrow
    d2 = W1.ncol# TODO check W2 ?
    if W2.all_independent:
        batch_le = len(W1.batch_shape)
        m = blockfrob(W1,B,inv=True,diag=True,X_event_shape=True)
        return m.sum(tuple(range(batch_le,len(m.shape))))
    elif W2.all_dependent:
        return blockfrob(W1,B,inv=True,diag=False,X_event_shape=False).sum(-1)
    elif W1.all_dependent:
        return blockfrob(W1,W2.full_tril,inv=True,diag=False,X_event_shape=False).sum(-1)
    elif W1.all_independent:
        A = W1.scale #if W1.all_independent else tril(W1.scale)
        if A.shape[-1]==1 and A.shape[-2]==1:
            sumBs = blockfrob(W2,torch.ones(1,1,device=B.device,dtype=B.dtype),inv=False,diag=True,X_event_shape=False)
            batch_le = len(W1.batch_shape)
            sumB = sumBs.sum(tuple(range(batch_le,len(sumBs.shape))))
            # to avoid adding multiple time the same quantity in cases of same variances or same covariances
            A2 = A.squeeze(-1).squeeze(-1)**2
            return sumB/A2
        Ar = A.expand(*A.shape[:-2],d1,d2).contiguous()
        if W2.only_col_dep:
            Au = Ar.unsqueeze(-1)
            sumDim = (-3,-2,-1)
        else :#B_only_row_dep:
            Au = Ar.transpose(-2,-1).unsqueeze(-1)
            sumDim = (-3,-2,-1)
        AB = B/Au
        #return block_mm(W2,1/A,inv=False,frob=True,diag=True,X_event_shape=True).sum((-1,-2))
    elif W1.only_col_dep:
        A = W1.scale #if W1.all_independent else tril(W1.scale)
        if W2.only_col_dep:
            AB = torch.triangular_solve(B,tril(A),upper=False,transpose=False)[0]
            sumDim = (-3,-2,-1)
        else:# B_only_row_dep:
            Ai = W1.get_tril_inverse()
            Au = Ai.unsqueeze(-1)
            Bu = B.transpose(-3,-2).unsqueeze(-3)
            AB = Bu*Au
            sumDim = (-4,-3,-2,-1)
    else:# W1.only_row_dep:
        if W2.only_col_dep:
            Ai = W1.get_tril_inverse()
            Au = Ai.unsqueeze(-1)
            Bu = B.transpose(-3,-2).unsqueeze(-3)
            AB = Bu*Au
            sumDim = (-4,-3,-2,-1)
        else:# B_only_row_dep:
            A = W1.scale #if W1.all_independent else tril(W1.scale)
            AB = torch.triangular_solve(B,tril(A),upper=False,transpose=False)[0]
            sumDim = (-3,-2,-1)
    return (AB**2).sum(sumDim)

def tril_op(L,X,inv,frob,diag):
    Ltril = tril(L)
    if X.shape[-2]==1:
        if not inv:
            if diag:
                LX = (Ltril*X)
                if frob:
                    return (LX**2).sum(-2)
                else:
                    return LX
            else:
                sumL = Ltril.sum(-1).unsqueeze(-1)
                if frob:
                    return (sumL**2).sum(-2)*X.squeeze(-2)**2
                else:
                    return sumL*X
        else:
            if diag:
                X_exp = torch.diag_embed(X.expand(*X.shape[:-2],1,L.size(-1)).squeeze(-2), offset=0, dim1=-2, dim2=-1)
            else:
                X_exp = X.expand(*X.shape[:-2],L.size(-1),X.size(-1))
    else:
        X_exp = X
    if inv:
        LX = torch.triangular_solve(X_exp,Ltril,upper=False,transpose=False)[0]
    else:
        if diag:
            print(("error! with diag, dim[-2] must be one"))#TODO
        LX = matmul(Ltril,X_exp)
    if frob:
        return (LX**2).sum(-2)
    else:
        return LX


def torch_distrib(q):
    d1 = q.nrow
    d2 = q.ncol
    m = q.loc.expand(d1,d2).view(-1,1).squeeze(-1)
    L = q.full_tril
    return torch.distributions.MultivariateNormal(m,scale_tril=L)

def blockfrob(W,X,inv=False,diag=False,X_event_shape=True):
    A = W.scale # if W.all_independent else tril(W.scale)
    d1 = W.nrow
    d2 = W.ncol
    all_independent = W.all_independent
    all_dependent = W.all_dependent
    only_col_dep    = W.only_col_dep
    only_row_dep    = W.only_row_dep
    if diag and X_event_shape and not all_independent:
        if X.shape[-1]!=1 or X.shape[-2]!=1:
            Xex = X.expand(*X.shape[:-2],d1,d2)
            if all_dependent:
                B = Xex.contiguous().view(*X.shape[:-2],1,d1*d2)
            else:
                B = Xex
        else:
            B = X
        B_event_shape=False
    else:
        B = X
        B_event_shape=X_event_shape
    if all_independent:
        if B_event_shape:
            if not diag:
                fro_fac = d1*d2/(max(A.shape[-1],B.shape[-1])*max(A.shape[-2],B.shape[-2]))
                if (A.shape[-1]==1 and B.shape[-2]==1) or (B.shape[-1]==1 and A.shape[-2]==1):
                    #fro_fac = d1*d2/(A.shape[-1]*A.shape[-2]*B.shape[-1]*B.shape[-2])
                    Asum = (1/A**2).sum((-1,-2)) if inv else (A**2).sum((-1,-2))
                    Asum_u = Asum.unsqueeze(-1).unsqueeze(-1)
                    return (Asum_u*(B**2).sum((-1,-2)).unsqueeze(-1).unsqueeze(-1)*fro_fac).squeeze(-1)
                else:
                    AB = B/A if inv else A*B
                    return (AB**2).sum((-2,-1)).unsqueeze(-1)*fro_fac
            else:
                ABs = (B/A if inv else A*B)
                ABs2 = ABs**2
                AB = ABs2.expand(*ABs.shape[:-2],d1,d2)
        else:
            if not diag and B.shape[-2]==1:
                Asum = (1/A**2).sum((-1,-2)).unsqueeze(-1) if inv else (A**2).sum((-1,-2)).unsqueeze(-1)
                return Asum*(B**2).sum(-2)*W._fro_fac
            else:
                if not diag and A.shape[-1]==1 and A.shape[-2]==1:
                    A2 = 1/A**2 if inv else A**2
                    As = A2.squeeze(-1)
                    return As*(B**2).sum(-2)
                if diag:
                    Ar = A.expand(*A.shape[:-2],d1,d2).contiguous().view(*A.shape[:-2],1,-1)
                else:
                    Ar = A.expand(*A.shape[:-2],d1,d2).contiguous().view(*A.shape[:-2],-1,1)
                AB = B/Ar if inv else Ar*B
                return (AB**2).sum(-2)
        return AB
    elif all_dependent:
        if B_event_shape and not (B.size(-1)==1 and B.size(-2)==1):
            Brshaped = B.expand(*B.shape[:-2],d1,d2).contiguous().view(*B.shape[:-2],d1*d2,1)
        else:
            Brshaped = B
        return tril_op(A,Brshaped,inv,True,diag)
    else:
        if diag:
            if B.size(-1)==1:
                di = d1 if only_row_dep else d2
                Bu = B.unsqueeze(-2).expand(*B.shape[:-2],1,1,di)
            else:
                Bav = B.view(*B.shape[:-2],d1,1,d2)
                Bu = Bav.transpose(-1,-3) if only_row_dep else Bav
            comp = tril_op(A,Bu,inv,False,diag)
            if only_row_dep:
                compt = comp.transpose(-2,-1).contiguous().view(*comp.shape[:-3],d2,d1,d1) 
            else:
                compt = comp.contiguous().view(*comp.shape[:-3],d1,d2,d2)
            return (compt**2).sum(-1).transpose(-2,-1) if only_row_dep else (compt**2).sum(-2)
        else:
            if B_event_shape:
                Bt = B.transpose(-1,-2) if only_row_dep else B
                Bu = Bt.unsqueeze(-1)
            else:
                if B.size(-2)==1:
                    Bu = B.unsqueeze(-3)
                else:
                    Bav = B.view(*B.shape[:-2],d1,d2,-1)
                    Bu = Bav.transpose(-3,-2) if only_row_dep else Bav
            comp = tril_op(A,Bu,inv,True,diag)
            return comp.sum(-2)*W._fro_fac


def blockmatmul(W,X,inv=False,X_event_shape=True):
    A = W.scale # if W.all_independent else tril(W.scale)
    d1 = W.nrow
    d2 = W.ncol
    all_independent = W.all_independent
    all_dependent = W.all_dependent
    only_col_dep    = W.only_col_dep
    only_row_dep    = W.only_row_dep
    B = X
    B_event_shape=X_event_shape
    if all_independent:
        if B_event_shape:
            ABs = (B/A if inv else A*B)
            ABs2 = ABs**2 if frob else ABs
            AB = ABs2.expand(*ABs2.shape[:-2],d1,d2)
        else:
            Ar = A.expand(*A.shape[:-2],d1,d2).contiguous().view(*A.shape[:-2],-1,1)
            AB = B/Ar if inv else Ar*B
        return AB
    elif all_dependent:
        if B_event_shape and not (B.size(-1)==1 and B.size(-2)==1):
            Brshaped = B.expand(*B.shape[:-2],d1,d2).contiguous().view(*B.shape[:-2],d1*d2,1)
        else:
            Brshaped = B
        AB = tril_op(A,Brshaped,inv,False,False)
        if B_event_shape:
            return A.view(*AB.shape[:-2],d1,d2)
        else:
            return AB
    else:
        if B_event_shape:
            Bt = B.transpose(-1,-2) if only_row_dep else B
            Bu = Bt.unsqueeze(-1)
            comp = tril_op(A,Bu,inv,False,False).squeeze(-1)
            compt = comp.transpose(-1,-2) if only_row_dep else comp
            return compt#.contiguous().view(*compt.shape[:-2],d1,d2)
        else:
            if B.size(-2)==1:
                Bu = B.unsqueeze(-3)
            else:
                Bav = B.view(*B.shape[:-2],d1,d2,-1)
                Bu = Bav.transpose(-3,-2) if only_row_dep else Bav
            comp = tril_op(A,Bu,inv,False,False)
            compt = comp.transpose(-3,-2) if only_row_dep else comp
            return compt.contiguous().view(*comp.shape[:-3],d1*d2,-1)


def frobe_Norm(B,d1,d2,all_independent,only_col_dep,only_row_dep):
    ef_dim = (-3,-2,-1) if only_row_dep or only_col_dep else (-2,-1)
    all_dependent = not all_independent and not only_row_dep and not only_col_dep
    if all_independent:
        f1 = d1 if B.size(-2)== 1 else 1
        f2 = d2 if B.size(-1)== 1 else 1
    elif only_col_dep:
        f1 = d1 if B.size(-3)== 1 else 1
        f2 = 1
    elif only_row_dep:
        f1 = 1
        f2 = d2 if B.size(-3)== 1 else 1
    return (B**2).sum(ef_dim) if all_dependent else f1*f2*(B**2).sum(ef_dim)



class GaussianMatrix(torch.nn.Module):
    def __init__(self,*dim,dependent_rows = False,dependent_cols = False,same_row_cov=False,same_col_cov=False,constant_mean=None,centered = False ,stddev=1.0):
        super(GaussianMatrix, self).__init__()
        # delfault is independent rows and columns (but not equally distributed)
        self.nrow = dim[-2]
        self.ncol = dim[-1]
        self.batch_shape = torch.Size(dim[:-2])
        self.event_shape = torch.Size(dim)
        self.dependent_rows = dependent_rows
        self.dependent_cols = dependent_cols
        self.same_row_cov = same_row_cov
        self.same_col_cov = same_col_cov
        self.centered = centered
        #super(GaussianMatrix, self).__init__(batch_shape, dim[-2:], validate_args=None)

        if centered:
            self.loc=torch.zeros(1,1)
        elif constant_mean is not None:
            self.loc = constant_mean*torch.ones(self.batch_shape).unsqueeze(-1).unsqueeze(-1)
        else:
            self.loc = torch.zeros(self.event_shape)
        
#        if dependent_rows and not dependent_cols:
#            self.only_col_dep = True
#            self.only_row_dep = False
            #self._tr = True
            #self._dep_cols = True
            #self._n = self.ncol
            #self._m = self.nrow
            #self._same_r = same_col_cov
            #self._same_c = same_row_cov
#        elif dependent_cols and not dependent_rows:
#            self.only_col_dep = False
#            self.only_row_dep = True
            #self._tr = False
            #self._dep_cols = dependent_cols
            #self._n = self.nrow
            #self._m = self.ncol
            #self._same_c = same_col_cov
            #self._same_r = same_row_cov

        self.all_dependent = dependent_cols and dependent_rows
        self.only_col_dep = dependent_cols and not dependent_rows
        self.only_row_dep = dependent_rows and not dependent_cols
        self.all_independent = not dependent_rows and not dependent_cols


        # TODO : warnings for not treated cases
        if not self.all_independent:
            if self.all_dependent:
                scale_dims = (self.nrow*self.ncol,self.nrow*self.ncol)
            elif self.only_col_dep:
                if self.same_col_cov:
                    scale_dims = (1,self.ncol,self.ncol)
                    self._fro_fac = self.row
                else:
                    scale_dims = (self.nrow,self.ncol,self.ncol)
                    self._fro_fac = 1
            else:
                if self.same_row_cov:
                    scale_dims = (1,self.nrow,self.nrow)
                    self._fro_fac = self.ncol
                else:
                    scale_dims = (self.ncol,self.nrow,self.nrow)
                    self._fro_fac = 1
        else:
            if self.same_row_cov:
                if self.same_col_cov:
                    scale_dims = (1,1)
                    self._fro_fac = self.ncol*self.nrow
                else:
                    scale_dims = (1,self.ncol)
                    self._fro_fac = self.nrow
            else:
                if self.same_col_cov:
                    scale_dims = (self.nrow,1)
                    self._fro_fac = self.ncol
                else:
                    scale_dims = (self.nrow,self.ncol)
                    self._fro_fac = 1

        self.scale = torch.zeros(*(self.batch_shape+scale_dims))
        self.set_diagonal(stddev)
        self.scale_dims = (-(i+1) for i in range(len(scale_dims)))

    """
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianMatrix, _instance)
        #new.__init__(loc=torch.zeros(1,1), row_scale_tril=None, col_scale_tril=None, validate_args=None)

        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        super(GaussianMatrix, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)

        new.loc = torch.nn.Parameter(self.loc.expand(loc_shape).detach())
        row_cov_tensor, _ = torch.broadcast_tensors(self.row_cov.parameter, new.loc[...,:,0].unsqueeze(-1))
        col_cov_tensor, _ = torch.broadcast_tensors(self.col_cov.parameter, new.loc[...,0,:].unsqueeze(-2))
        new.row_cov = CovarianceMatrix(d=self.nrow,data=row_cov_tensor.detach(),requires_grad=self.row_cov.parameter.requires_grad)
        new.col_cov = CovarianceMatrix(d=self.ncol,data=col_cov_tensor.detach(),requires_grad=self.col_cov.parameter.requires_grad)
        new.nrow = new.loc.size(-2)
        new.ncol = new.loc.size(-1)
        return new
        """

    @property
    def full_tril(self):
        d1 = self.nrow
        d2 = self.ncol
        batch_shape = self.batch_shape
        if self.all_independent:
            L = torch.diag_embed(self.scale.expand(batch_shape+(d1,d2)).contiguous().view(*batch_shape,d1*d2,1).squeeze(-1))
        elif self.only_col_dep:
            L_blocks = torch.tril(self.scale).expand(*batch_shape,d1,d2,d2)
            L = torch.diag_embed(L_blocks.unsqueeze(-1).transpose(-1,-4).squeeze(0)).unsqueeze(-5).transpose(-2,-5).squeeze(-2).transpose(-1,-2).contiguous().view(*self.batch_shape,d1*d2,d1*d2)
        elif self.only_row_dep:
            L_blocks = torch.tril(self.scale).expand(*batch_shape,d2,d1,d1)
            L = torch.diag_embed(L_blocks.unsqueeze(-1).transpose(-1,-4).squeeze(-4)).transpose(-2,-3).contiguous().view(*batch_shape,d1*d2,d1*d2)
        else:
            L = torch.tril(self.scale)
        return L

    def to_torch_distrib(self):# TODO handle low rank
        d1 = self.nrow
        d2 = self.ncol
        batch_shape = self.batch_shape
        m = self.loc.expand(*batch_shape,d1,d2).view(*batch_shape,-1,1).squeeze(-1)
        L = self.full_tril
        return torch.distributions.MultivariateNormal(m,scale_tril=L)

    def _extended_shape(self,sample_shape):
        return sample_shape + self.event_shape

    def get_diagonal(self,formated=True):
        if self.all_independent:
            s = self.scale
            return s.expand(self.event_shape) if formated else s
        else:
            s = torch.diagonal(self.scale,dim1=-2,dim2=-1)
            if formated:
                if self.only_col_dep:
                    return s.expand(self.event_shape)
                elif self.only_row_dep:
                    return s.transpose(-1,-2).expand(self.event_shape)
                else:
                    return s.view(self.event_shape)
            else:
                return s

    def set_diagonal(self,value):
        if self.all_independent:
            self.scale.fill_(value)
        else:
            torch.diagonal(self.scale,offset=0,dim1=-2,dim2=-1).fill_(value)
    def get_variances(self,expand_rows=False,expand_cols=False):
        if self.all_independent:
            V = self.scale**2
        else:
            gv = (tril(self.scale)**2).sum(-1)
            gvs = gv.shape[:-2]
            if self.only_row_dep:
                V = gv.transpose(-1,-2)
            elif self.all_dependent:
                return gv.view(*gvs,self.nrow,self.ncol)
        if expand_rows and expand_cols:
            return V.expand(*gvs,self.nrow,self.ncol)
        elif expand_rows and not expand_cols:
            return V.expand(*gvs,self.nrow,-1)
        elif not expand_rows and expand_cols:
            return V.expand(*gvs,-1,self.ncol)
        else:
            return V

    @property
    def variances(self):
        return self.get_variances(expand_rows=True,expand_cols=True)


    def get_stddevs(self):
        if self.all_independent:
            return self.param
        else:
            return (tril(self.scale)**2).sum(-1).sqrt()
    def set_stddevs(self,value):# float or torch scalar or matrix (batches times nrow times ncol)
        # TODO value check when same_row_dist e.g.
        if self.all_independent:
            if isinstance(value, (int, float)) or value.numel() == 1:
                self.scale.fill_(value.squeeze())
            else:
                self.scale = value.expand(self.scale.size())
        else:
            stddevs = self.get_stddevs()
            if isinstance(value, (int, float)) or value.numel() == 1:
                self.scale *= value/stddevs.unsqueeze(-1)
            else:
                if self.all_dependent:
                    upd_fac = value.view(*value.shape[:-2],self.ncol*self.nrow,1)/stddevs.unsqueeze(-1)
                    self.scale *= upd_fac
                else:
                    valuet = value.transpose(-2,-1) if self.only_row_dep else value
                    self.scale *= ((valuet/stddevs).unsqueeze(-1)).expand(self.scale.size())# expand is just a check

    def get_tril_inverse(self):# computed when L^-1X has to be computed for multiple X's
        if self.all_independent:
            return 1/self.scale
        else:
            I = torch.eye(self.scale.size(-1), dtype=self.scale.dtype,device=self.scale.device)
            return  torch.triangular_solve(I,tril(self.scale),upper=False,transpose=False)[0]

    def LvectX(self,X,is_vectorised=False,out_reshape=True,inverse=False,pre_comp_inv=None):
        if inverse:
            sc = pre_comp_inv if pre_comp_inv is not None else self.get_tril_inverse()
        else:
            if self._dep_cols:
                sc = tril(self.scale)
            else:
                sc = self.scale
        # performs (L vect(X)).view(nrow,ncol), batchwise, with L the (virtual) covariance root
        if self._dep_cols:
            if self.all_dependent: # L is full
                if not is_vectorised:# transpose for vectorize by column for consistency with row_cov block diagonal case
                    X_vect = X.view(*X.shape[:-2],X.size(-2)*X.size(-1),1)
                else:
                    X_vect = X.unsqueeze(-1)
                LX = matmul(sc,X_vect)
                if out_reshape:
                    return LX.view(*LX.shape[:-2],self._n,self._m)
                else:
                    return LX.squeeze(-1)
            else:# L is block diagonal
                Xt = X.transpose(-2,-1) if self._tr else X
                batch_X = Xt.unsqueeze(-1) # rows are treated independent repetitions
                batch_LX = matmul(sc,batch_X)
                res = batch_LX.squeeze(-1)#   return batch_LX.view(*batch_LX.shape[:-2],Xt.size(-1)*self._n)
                return res.transpose(-2,-1) if self._tr else res
        else:# L is diagonal
            return X*sc.expand(self.event_shape) # expand do nothing if not same_col_dist
        
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        if self.all_independent:
            eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
            return self.loc + eps*self.scale.expand(self.event_shape) # expand do nothing if not same_col_dist
        elif self.all_dependent:
            eps = _standard_normal(shape[:-2]+torch.Size((self.ncol*self.nrow,1)), dtype=self.loc.dtype, device=self.loc.device)
            Leps = blockmatmul(self,eps,inv=False,X_event_shape=False)
            print(Leps.size())
            return self.loc + Leps.view(shape)#self.LvectX(eps,is_vectorised=True)
        else:
            d1 = self.nrow
            d2 = self.ncol
            #eps_sh = sample_shape + torch.Size((d1,d2,1)) if self.only_col_dep else sample_shape + torch.Size((d2,d1,1))
            #eps = _standard_normal(eps_sh, dtype=self.loc.dtype, device=self.loc.device)
            eps = _standard_normal(shape[:-2]+torch.Size((self.nrow,self.ncol)), dtype=self.loc.dtype, device=self.loc.device)
            Leps = blockmatmul(self,eps,inv=False,X_event_shape=True)
            #Lepst = Leps.transpose(-2,-1) if self.only_row_dep else Leps
            return self.loc + Leps # self.LvectX(eps)
        #self.row_cov.L_times(eps)
        #return self.loc + self.col_cov.times_L(self.row_cov.L_times(eps))


    def lrsample(self, X,sample_shape,ignore_dependence_rows=False,ignore_dependence_cols=False): # samples with independant rows that have mean and covariance as matmul(X,self.rsample())
        dep_rows = self.dependent_rows
        dep_cols = self.dependent_cols
        if ignore_dependence_rows and dep_rows:
            dep_rows = False
        if ignore_dependence_cols and dep_cols:
            dep_cols = False
        d1 = self.nrow
        d2 = self.ncol
        if self.centered:
            Y_mean = 0
        elif self.loc.shape[-2] ==1:
            Y_mean = X.sum(-1).unsqueeze(-1)*self.loc
        else:
            Y_mean = torch.matmul(X,self.loc)
        if not (dep_rows or dep_cols):
            eps = _standard_normal(sample_shape+Y_mean.size(), dtype=self.loc.dtype, device=self.loc.device)
            V = self.get_variances()
            if V.size(-2) == 1:
                X2V = ((X**2).sum(-1).unsqueeze(-1)*V)
            else:
                X2V = matmul(X**2,V)
            Y_centered = X2V.sqrt()*eps
        elif dep_rows and dep_cols:
            Xrep = X.unsqueeze(-1).expand(*(X.shape[:-1]+(d1,d2)))
            Xrsh = Xrep.contiguous().view(*(X.shape[:-1]+(d1*d2,1)))
            Luns = tril(self.scale).unsqueeze(-3)
            XL = Xrsh*Luns
            XLrsh = XL.view(*(X.shape[:-1]+(d1,d2,d1*d2)))
            U = XLrsh.sum(-3) # cov root of XW
            eps = _standard_normal(sample_shape+torch.Size(Y_mean.shape[:-2]+(X.size(-2),d1*d2,1)), dtype=self.loc.dtype, device=self.loc.device)
            Y_centered = matmul(U,eps).squeeze(-1)
            #Y_centered = Y_c_vect.view(Y_mean.size())
        elif dep_rows and not dep_cols:
            if not self.all_dependent:
                Y_V = (matmul(X.unsqueeze(-2).unsqueeze(-2),tril(self.scale).unsqueeze(-4)).squeeze(-2)**2).sum(-1)
            else:
                L_rs = tril(self.scale).view(*self.scale.shape[:-2],d1,d2,d1*d2)
                XL = X.unsqueeze(-1).unsqueeze(-1)*L_rs.unsqueeze(-4)
                C = (XL.unsqueeze(-4)*XL.unsqueeze(-3)).sum(-1)
                Y_V = C.sum((-3,-2))
            eps = _standard_normal(sample_shape+Y_mean.size(), dtype=self.loc.dtype, device=self.loc.device)
            Y_centered = Y_V.sqrt()*eps
        else: #not dep_rows and dep_cols
            if not self.all_dependent:
                X_uns = X.unsqueeze(-1).unsqueeze(-1)
                L_uns = tril(self.scale).unsqueeze(-4)
                XL = X_uns * L_uns
                eps = _standard_normal(sample_shape+torch.Size(Y_mean.shape[:-1]+(d1,d2,1)), dtype=self.loc.dtype, device=self.loc.device)
                Y_centered = matmul(XL,eps).squeeze(-1).sum(-2)
            else:
                X_uns = X.unsqueeze(-2).unsqueeze(-2)
                L_rs =  tril(self.scale).view(*self.scale.shape[:-2],d1,d2,d1*d2)
                L_uns = L_rs.unsqueeze(-4).unsqueeze(-4).transpose(-4,-1).squeeze(-1)
                XL = matmul(X_uns,L_uns).transpose(-1,-3)
                eps = _standard_normal(sample_shape+torch.Size(Y_mean.shape[:-2]+[d1*d2,1]), dtype=self.loc.dtype, device=self.loc.device)
                Y_centered = matmul(XL,eps).squeeze(-1)
        return Y_mean + Y_centered

    @property
    def log_det(self):
        di = self.get_diagonal(formated=False).abs().log()
        if self.all_independent:
            f1 = self.nrow if self.scale.size(-2)== 1 else 1
            f2 = self.ncol if self.scale.size(-1)== 1 else 1
        elif self.only_col_dep:
            f1 = self.nrow if self.scale.size(-3)== 1 else 1
            f2 = 1
        elif self.only_row_dep:
            f1 = 1
            f2 = self.col if self.scale.size(-3)== 1 else 1
        sum_log_di = di.sum((-1)) if self.all_dependent else f1*f2*di.sum((-2,-1))
        return 2*sum_log_di


    def log_prob(self, value):
        diff = value - self.loc
        # No unecessary triangular inversion here:
        dist = blockfrob(self,diff,inv=True,diag=False,X_event_shape=True).squeeze(-1)
        d = self.nrow*self.ncol
        return -.5*(d*log2pi+self.log_det+dist)

    def optimize(self, train: bool = True):
        for param in self.parameters():
            param.requires_grad = train


    """ 
    def entropy(self):
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
 """


class GaussianVector(GaussianMatrix):
    support = constraints.real
    has_rsample = True
    has_lrsample = True

    def __init__(self,*dim,independent = True,const_mean = False, homoscedastic = False,centered=False,stddev=1.0):
        self.d = dim[-1]
        super(GaussianVector, self).__init__(*dim,1,independent_rows = independent,independent_cols = True,const_row_mean = const_mean,const_col_mean = True, homoscedastic_rows = homoscedastic,homoscedastic_cols = False ,centered=centered,stddev=stddev)

    def rsample(self, sample_shape=torch.Size()):
        return super().rsample(sample_shape).squeeze(-1)

    def lrsample(self, X): 
        return super().lrsample(X).squeeze(-1)

    @property
    def stddev(self):
        return self.row_cov.stddev
    @stddev.setter
    def stddev(self,X):
        self.row_cov.stddev=X

    @property
    def cov(self):
        return self.row_cov

    @property
    def tril(self):
        return self.row_cov.tril
    @tril.setter
    def tril(self,X):
        self.row_cov.tril = X

    def log_prob(self, value):
        return super().log_prob(value.unsqueeze(-1)).squeeze(-1)
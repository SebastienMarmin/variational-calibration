import torch
from torch import tril, matmul
from .utilities import _standard_normal

log2pi = 1.83787706640934533918

def L1_inv_L2_frob(W1, W2):
    B = W2.scale if W2.all_independent else tril(W2.scale)
    d1 = W1.nrow
    d2 = W1.ncol  # TODO check W2 ?
    if W2.all_independent:
        batch_le = len(W1.batch_shape)
        m = blockfrob(W1, B, inv=True, diag=True, X_event_shape=True)
        return m.sum(tuple(range(batch_le, len(m.shape))))
    elif W2.all_dependent:
        return blockfrob(W1, B, inv=True, diag=False, X_event_shape=False).sum(-1)
    elif W1.all_dependent:
        return blockfrob(W1, W2.full_tril, inv=True, diag=False, X_event_shape=False).sum(-1)
    elif W1.all_independent:
        A = W1.scale  # if W1.all_independent else tril(W1.scale)
        if A.shape[-1] == 1 and A.shape[-2] == 1:
            sumBs = blockfrob(W2, torch.ones(1, 1, device=B.device, dtype=B.dtype), inv=False, diag=True, X_event_shape=False)
            batch_le = len(W1.batch_shape)
            sumB = sumBs.sum(tuple(range(batch_le, len(sumBs.shape))))
            # to avoid adding multiple time the same quantity in cases of same
            # variances or same covariances
            A2 = A.squeeze(-1).squeeze(-1)**2
            return sumB/A2
        Ar = A.expand(*A.shape[:-2], d1, d2).contiguous()
        if W2.only_col_dep:
            Au = Ar.unsqueeze(-1)
            sumDim = (-3, -2, -1)
        else:  # B_only_row_dep:
            Au = Ar.transpose(-2, -1).unsqueeze(-1)
            sumDim = (-3, -2, -1)
        AB = B/Au
    elif W1.only_col_dep:
        A = W1.scale  # if W1.all_independent else tril(W1.scale)
        if W2.only_col_dep:
            AB = torch.triangular_solve(B, tril(A), upper=False, transpose=False)[0]
            sumDim = (-3, -2, -1)
        else:  # B_only_row_dep:
            Ai = W1.get_tril_inverse()
            Au = Ai.unsqueeze(-1)
            Bu = B.transpose(-3, -2).unsqueeze(-3)
            AB = Bu*Au
            sumDim = (-4, -3, -2, -1)
    else:  # W1.only_row_dep:
        if W2.only_col_dep:
            Ai = W1.get_tril_inverse()
            Au = Ai.unsqueeze(-1)
            Bu = B.transpose(-3, -2).unsqueeze(-3)
            AB = Bu*Au
            sumDim = (-4, -3, -2, -1)
        else:  # B_only_row_dep:
            A = W1.scale # if W1.all_independent else tril(W1.scale)
            AB = torch.triangular_solve(B, tril(A), upper=False, transpose=False)[0]
            sumDim = (-3, -2, -1)
    return (AB**2).sum(sumDim)


def tril_op(L, X, inv, frob, diag):
    Ltril = tril(L)
    if X.shape[-2] == 1:
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
                X_exp = torch.diag_embed(X.expand(*X.shape[:-2], 1, L.size(-1)).squeeze(-2), offset=0, dim1=-2, dim2=-1)
            else:
                X_exp = X.expand(*X.shape[:-2], L.size(-1), X.size(-1))
    else:
        X_exp = X
    if inv:
        LX = torch.triangular_solve(X_exp, Ltril, upper=False, transpose=False)[0]
    else:
        if diag:
            print(("error! with diag, dim[-2] must be one"))  # TODO
        LX = matmul(Ltril, X_exp)
    if frob:
        return (LX**2).sum(-2)
    else:
        return LX


def torch_distrib(q):
    d1 = q.nrow
    d2 = q.ncol
    m = q.loc.expand(d1, d2).view(-1, 1).squeeze(-1)
    L = q.full_tril
    return torch.distributions.MultivariateNormal(m, scale_tril=L)


def blockfrob(W, X, inv=False, diag=False, X_event_shape=True):
    A = W.scale  # if W.all_independent else tril(W.scale)
    d1 = W.nrow
    d2 = W.ncol
    all_independent = W.all_independent
    all_dependent = W.all_dependent
    only_row_dep = W.only_row_dep
    if diag and X_event_shape and not all_independent:
        if X.shape[-1] != 1 or X.shape[-2] != 1:
            Xex = X.expand(*X.shape[:-2], d1, d2)
            if all_dependent:
                B = Xex.contiguous().view(*X.shape[:-2], 1, d1*d2)
            else:
                B = Xex
        else:
            B = X
        B_event_shape = False
    else:
        B = X
        B_event_shape = X_event_shape
    if all_independent:
        if B_event_shape:
            if not diag:
                fro_fac = d1*d2/(max(A.shape[-1], B.shape[-1])*max(A.shape[-2], B.shape[-2]))
                if (A.shape[-1] == 1 and B.shape[-2] == 1) or (B.shape[-1] == 1 and A.shape[-2] == 1):
                    Asum = (1/A**2).sum((-1, -2)) if inv else (A**2).sum((-1, -2))
                    Asum_u = Asum.unsqueeze(-1).unsqueeze(-1)
                    return (Asum_u*(B**2).sum((-1, -2)).unsqueeze(-1).unsqueeze(-1)*fro_fac).squeeze(-1)
                else:
                    AB = B/A if inv else A*B
                    return (AB**2).sum((-2, -1)).unsqueeze(-1)*fro_fac
            else:
                ABs = (B/A if inv else A*B)
                ABs2 = ABs**2
                AB = ABs2.expand(*ABs.shape[:-2], d1, d2)
        else:
            if not diag and B.shape[-2] == 1:
                Asum = (1/A**2).sum((-1, -2)).unsqueeze(-1) if inv else (A**2).sum((-1, -2)).unsqueeze(-1)
                return Asum*(B**2).sum(-2)*W._fro_fac
            else:
                if not diag and A.shape[-1] == 1 and A.shape[-2] == 1:
                    A2 = 1/A**2 if inv else A**2
                    As = A2.squeeze(-1)
                    return As*(B**2).sum(-2)
                if diag:
                    Ar = A.expand(*A.shape[:-2], d1, d2).contiguous().view(*A.shape[:-2], 1, -1)
                else:
                    Ar = A.expand(*A.shape[:-2], d1, d2).contiguous().view(*A.shape[:-2], -1, 1)
                AB = B/Ar if inv else Ar*B
                return (AB**2).sum(-2)
        return AB
    elif all_dependent:
        if B_event_shape and not (B.size(-1) == 1 and B.size(-2) == 1):
            Brshaped = B.expand(*B.shape[:-2], d1, d2).contiguous().view(*B.shape[:-2], d1*d2,1)
        else:
            Brshaped = B
        return tril_op(A, Brshaped, inv, True, diag)
    else:
        if diag:
            if B.size(-1) == 1:
                di = d1 if only_row_dep else d2
                Bu = B.unsqueeze(-2).expand(*B.shape[:-2], 1, 1, di)
            else:
                Bav = B.view(*B.shape[:-2], d1, 1, d2)
                Bu = Bav.transpose(-1, -3) if only_row_dep else Bav
            comp = tril_op(A, Bu, inv, False, diag)
            if only_row_dep:
                compt = comp.transpose(-2, -1).contiguous().view(*comp.shape[:-3], d2, d1, d1)
            else:
                compt = comp.contiguous().view(*comp.shape[:-3], d1, d2, d2)
            return (compt**2).sum(-1).transpose(-2, -1) if only_row_dep else (compt**2).sum(-2)
        else:
            if B_event_shape:
                Bt = B.transpose(-1, -2) if only_row_dep else B
                Bu = Bt.unsqueeze(-1)
            else:
                if B.size(-2) == 1:
                    Bu = B.unsqueeze(-3)
                else:
                    Bav = B.view(*B.shape[:-2], d1, d2, -1)
                    Bu = Bav.transpose(-3, -2) if only_row_dep else Bav
            comp = tril_op(A, Bu, inv, True, diag)
            return comp.sum(-2)*W._fro_fac


def blockmatmul(W,X,inv=False,X_event_shape=True):
    A = W.scale # if W.all_independent else tril(W.scale)
    d1 = W.nrow
    d2 = W.ncol
    all_independent = W.all_independent
    all_dependent = W.all_dependent
    only_col_dep    = W.only_col_dep
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
            Bt = B.transpose(-1,-2) if W.only_row_dep else B
            Bu = Bt.unsqueeze(-1)
            comp = tril_op(A,Bu,inv,False,False).squeeze(-1)
            compt = comp.transpose(-1,-2) if W.only_row_dep else comp
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




class GaussianMatrix(torch.nn.Module):
    def __init__(self,*dim,dependent_rows = False,dependent_cols = False,same_row_cov=False,same_col_cov=False,constant_mean=None,centered = False ,stddev=1.0,parameter=True):
        super(GaussianMatrix, self).__init__()
        # delfault is independent rows and columns (but not equally distributed)
        self.nrow = dim[-2]
        self.ncol = dim[-1]
        self.batch_shape = torch.Size(dim[:-2])
        self.event_shape = torch.Size(dim)
        self.dependent_rows = dependent_rows
        self.dependent_cols = dependent_cols
        self.centered = centered
        #super(GaussianMatrix, self).__init__(batch_shape, dim[-2:], validate_args=None)

        if centered:
            self.loc=torch.zeros(1,1)
        elif constant_mean is not None:
            self.loc = constant_mean*torch.ones(*self.batch_shape,1,1)
        else:
            self.loc = torch.zeros(self.event_shape)

        self.all_dependent = dependent_cols and dependent_rows
        self.only_col_dep = dependent_cols and not dependent_rows
        self.only_row_dep = dependent_rows and not dependent_cols
        self.all_independent = not dependent_rows and not dependent_cols


        # TODO : warnings for not treated cases
        if not self.all_independent:
            if self.all_dependent:
                scale_dims = (self.nrow*self.ncol,self.nrow*self.ncol)
            elif self.only_col_dep:
                if same_col_cov:
                    scale_dims = (1,self.ncol,self.ncol)
                    self._fro_fac = self.row
                else:
                    scale_dims = (self.nrow,self.ncol,self.ncol)
                    self._fro_fac = 1
            else:
                if same_row_cov:
                    scale_dims = (1,self.nrow,self.nrow)
                    self._fro_fac = self.ncol
                else:
                    scale_dims = (self.ncol,self.nrow,self.nrow)
                    self._fro_fac = 1
        else:
            if same_row_cov:
                if same_col_cov:
                    scale_dims = (1,1)
                    self._fro_fac = self.ncol*self.nrow
                else:
                    scale_dims = (1,self.ncol)
                    self._fro_fac = self.nrow
            else:
                if same_col_cov:
                    scale_dims = (self.nrow,1)
                    self._fro_fac = self.ncol
                else:
                    scale_dims = (self.nrow,self.ncol)
                    self._fro_fac = 1

        self.scale = torch.zeros(*(self.batch_shape+scale_dims))
        self.set_diagonal(stddev)
        self.scale_dims = (-(i+1) for i in range(len(scale_dims)))

        if parameter:
            if not self.centered:
                self.loc = torch.nn.Parameter(self.loc.detach())
            self.scale = torch.nn.Parameter(self.scale.detach())

    def expand(self, *shape):
        q = self
        d1 = q.nrow
        d2 = q.ncol
        if d1!=shape[-2] or d2!=shape[-1]:
            print("erro")#TODO
        dependent_rows=q.dependent_rows
        dependent_cols=q.dependent_cols
        constant_mean=q.loc[0,0] if q.loc.shape[-1]==1 and q.loc.shape[-2]==1 else None
        centered = q.centered
        same_row_cov,same_col_cov = self._infere_same_cov()
        qT = GaussianMatrix(*shape,dependent_rows=dependent_rows,dependent_cols=dependent_cols,same_row_cov=same_row_cov,same_col_cov=same_col_cov,constant_mean=constant_mean,centered=centered,parameter=False)
        qT.loc = q.loc.expand(*qT.batch_shape,q.loc.size(-2),q.loc.size(-1))
        if qT.all_independent:
            qT.scale = q.scale.expand(*qT.batch_shape,q.scale.size(-2),q.scale.size(-1))
        elif qT.all_dependent:
            qT.scale = q.scale.expand(*qT.batch_shape,q.scale.size(-2),q.scale.size(-1))
        else:
            qT.scale = q.scale.expand(*qT.batch_shape,q.scale.size(-3),q.scale.size(-2),q.scale.size(-1))
        return qT


    def to_torch_distrib(self):# TODO handle low rank
        d1 = self.nrow
        d2 = self.ncol
        batch_shape = self.batch_shape
        m = self.loc.expand(*batch_shape,d1,d2).view(*batch_shape,-1,1).squeeze(-1)
        L = self.full_tril
        return torch.distributions.MultivariateNormal(m,scale_tril=L)

    def _extended_shape(self,sample_shape):
        return sample_shape + self.event_shape


    def _infere_same_cov(self):
        q = self
        if q.all_independent:
            same_row_cov = q.scale.shape[-2]==1
            same_col_cov = q.scale.shape[-2]==1
        elif q.all_dependent:
            same_row_cov = False
            same_col_cov = False
        elif q.only_row_dep:
            same_row_cov = False
            same_col_cov = q.scale.shape[-3]==1
        else:
            same_row_cov = q.scale.shape[-3]==1
            same_col_cov = False
        return same_row_cov,same_col_cov

    def transpose(self):
        q = self
        batch_shape = q.batch_shape
        d1 = q.ncol
        d2 = q.nrow
        dependent_rows=q.dependent_cols
        dependent_cols=q.dependent_rows
        constant_mean=q.loc[0,0] if q.loc.shape[-1]==1 and q.loc.shape[-2]==1 else None
        centered = q.centered
        same_row_cov,same_col_cov = self._infere_same_cov()
        qT = GaussianMatrix(*batch_shape,d1,d2,dependent_rows=dependent_rows,dependent_cols=dependent_cols,same_row_cov=same_col_cov,same_col_cov=same_row_cov,constant_mean=constant_mean,centered=centered,parameter=False)
        qT.loc = q.loc.transpose(-1,-2).contiguous()
        if qT.all_independent:
            qT.scale = q.scale.transpose(-1,-2).contiguous()
        elif qT.all_dependent:
            tril = torch.tril(q.scale)
            covRoot = tril.view(*tril.shape[:-2],d2,d1,d2*d1).transpose(-3,-2).contiguous().view(*tril.shape[:-2],d2*d1,d2*d1)
            qT.scale = torch.cholesky(torch.matmul(covRoot,covRoot.transpose(-1,-2)))
        else:
            qT.scale = q.scale
        return qT

    @property
    def mean(self):
        return self.loc.expand(*self.batch_shape,self.nrow,self.ncol)
    @mean.setter
    def mean(self,M):
        loc = M.expand(*self.loc.shape)
        try:
            self.loc = loc
        except TypeError:
            self.loc = torch.nn.Parameter(loc,requires_grad=self.loc.requires_grad)
    
    def set_covariance(self,q,detach=True):
        d1, d2 = self.nrow, self.ncol
        same_row_cov,same_col_cov = self._infere_same_cov()
        if self.all_independent:
            self.set_diagonal(q.get_variances(expand_rows=False,expand_cols=False,root=True).detach())
        elif (q.only_row_dep and self.only_row_dep) or (q.only_row_dep and self.only_row_dep):
            scale = q.scale.expand(*self.batch_shape,d1,d2,d2) if self.only_col_dep else  q.scale.expand(*self.batch_shape,d2,d1,d1)
            if (self.only_row_dep and same_col_cov) or (self.only_col_dep and same_row_cov):
                scale_res = (scale**2).mean(-3).sqrt().unsqueeze(-3)
            else:
                scale_res = scale
            try:
                self.scale = scale_res.detach()
            except TypeError:
                self.scale = torch.nn.Parameter(scale_res.detach(),requires_grad=self.scale.requires_grad)# check if detach needed
        elif q.all_independent or (q.only_row_dep and self.only_col_dep) or (q.only_col_dep and self.only_row_dep):
            try:
                self.scale = torch.zeros_like(self.scale)
            except TypeError:
                self.scale = torch.nn.Parameter(torch.zeros_like(self.scale),requires_grad=self.scale.requires_grad)
                # kill all correlations as expected behavior
            self.set_diagonal(q.get_variances(root=True).detach())
        else: # q.all_dependent or self.all_dependent,
            self.full_tril = q.full_tril.detach()

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
    @full_tril.setter
    def full_tril(self,L):
        self.set_scale_from_full_tril(L,is_cov=False)
    


    @property
    def covariance_matrix(self):## TODO matmul earlier
        L = self.full_tril
        return matmul(L,L.transpose(-1,-2))
    @covariance_matrix.setter
    def covariance_matrix(self,C):
        n = C.size(-2)
        m = C.size(-1)
        d1d2 = d1*d2
        if n!= d1d2 or m!= d1d2:
            print("error")# TODO
        self.set_scale_from_full_tril(C,is_cov=True)

    def set_scale_from_full_tril(self,L,is_cov=False):
        d1 = self.nrow
        d2 = self.ncol
        batch_shape = self.batch_shape
        if self.all_independent:# keep only the variances
            if is_cov:
                V = torch.diagonal(L,dim1=-1,dim2 = -2)
            else:
                V = (tril(L)**2).sum(-1)
            scale = V.expand(*batch_shape,d1*d2).view(*batch_shape,d1,d2)
        elif self.only_col_dep:
            if d1>1:# keep only the col cov
                if is_cov:
                    full_cov = L
                else:
                    full_cov = matmul(L,L.transpose(-2,-1))
                col_cov = torch.diagonal(full_cov.view(d1,d2,d1,d2).transpose(-3,-2),dim1=-3,dim2 = -4).transpose(-1,-3).transpose(-1,-2)
                col_L = torch.cholesky(col_cov)
                scale = col_L.expand(*batch_shape,d1,d2,d2)
            else:
                if is_cov:
                    lt = torch.cholesky(L)
                else:
                    lt = L
                scale = lt.unsqueeze(-3).expand(*batch_shape,1,d2,d2)
        elif self.only_row_dep:
            if d2>1:# keep only the row cov
                if is_cov:
                    full_cov = L
                else:
                    full_cov = matmul(L,L.transpose(-2,-1))
                row_cov = torch.diagonal(full_cov.view(d1,d2,d1,d2).transpose(-3,-2),dim1=-2,dim2 = -1).transpose(-1,-3).transpose(-1,-2)
                row_L = torch.cholesky(row_cov)
                scale = row_L.expand(*batch_shape,d2,d1,d1)
            else:
                if is_cov:
                    lt = torch.cholesky(L)
                else:
                    lt = L
                scale = lt.unsqueeze(-3).expand(*batch_shape,1,d1,d1)
        else:
            if is_cov:
                lt = torch.cholesky(L)
            else:
                lt = L
            scale = lt.expand(*batch_shape,d1*d2,d1*d2)
        try:
            self.scale = scale
        except TypeError:
            self.scale = torch.nn.Parameter(scale,requires_grad=self.scale.requires_grad)

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
        if isinstance(value, (int, float)):
            if self.all_independent:
                scale = torch.ones_like(self.scale)*value
            else:
                scale = self.scale.detach()
                torch.diagonal(scale,offset=0,dim1=-2,dim2=-1).fill_(value)
        else:
            d1 = self.nrow
            d2 = self.ncol
            same_row_cov,same_col_cov = self._infere_same_cov()
            scale_exp = value.expand(*self.batch_shape,d1,d2)
            scale_res = scale_exp
            if same_row_cov:
                scale_res = (scale_res**2).mean(-2).sqrt().unsqueeze(-2)
            if same_col_cov:
                scale_res = (scale_res**2).mean(-1).sqrt().unsqueeze(-1)

            if self.all_independent:
                scale = scale_res
            else:
                scale = self.scale.detach()
                torch.diagonal(scale,offset=0,dim1=-2,dim2=-1).fill_(0.0)
                if self.only_col_dep:
                    scale += torch.diag_embed(scale_res,dim1=-2,dim2=-1)
                elif self.only_row_dep:
                    scale += torch.diag_embed(scale_res.transpose(-1,-2),dim1=-2,dim2=-1)
                else:
                    scale += torch.diag_embed(scale_exp.contiguous().view(*scale_exp.shape[:-2],d1*d2),dim1=-2,dim2=-1)
        try:
            self.scale = scale.detach()
        except TypeError:
            self.scale = torch.nn.Parameter(scale,requires_grad=self.scale.requires_grad)


    def get_variances(self,expand_rows=False,expand_cols=False,root=False):
        if self.all_independent:
            V = self.scale.abs() if root else self.scale**2 
            gvs = V.shape[:-2]
        else:
            v = (tril(self.scale)**2).sum(-1)
            gv = v.sqrt() if root else v
            gvs = gv.shape[:-2]
            if self.only_row_dep:
                V = gv.transpose(-1,-2)
            elif self.all_dependent:
                return gv.view(*gvs,self.nrow,self.ncol)
            else:
                V = gv
        if expand_rows and expand_cols:
            return V.expand(*gvs,self.nrow,self.ncol)
        elif expand_rows and not expand_cols:
            return V.expand(*gvs,self.nrow,-1)
        elif not expand_rows and expand_cols:
            return V.expand(*gvs,-1,self.ncol)
        else:
            return V

    def get_column_covariances(self,index,root):
        if self.all_independent:
            index_c = 0 if self.scale.size(-1)==1 else index
            sc = self.scale[...,:,index_c].unsqueeze(-1)
            C =  sc.abs() if root else sc**2
        elif self.only_row_dep:
            index_c = 0 if self.scale.size(-3)==1 else index
            L = tril(self.scale[...,index_c,:,:])
            if root:
                C = L
            else:
                C = matmul(L,L.transpose(-2,-1))
        elif self.only_col_dep:
            sc = (self.scale[...,:,index,0:(index+1)]**2).sum(-1).unsqueeze(-1)
            C = sc.sqrt if root else sc**2
        else:
            raise NotImplementedError # TODO
        return C # dxd or dx1 if diagonal or 1x1 if diagonal and homoscedastic


    def set_column_covariances(self,L,index,root):# dxd or dx1 if diagonal or 1x1 if diagonal and homoscedastic
        with torch.no_grad():
            scale = self.scale
            L_diag = L.size(-1)==1
            if L_diag:
                s = L.squeeze(-1) if root else L.squeeze(-1).sqrt()
            if self.all_independent:
                if not L_diag:
                    v = (L**2).sum(-1) if root else torch.diagonal(L,dim1=-2,dim2=-1)
                    s = v.sqrt()
                scale[...,:,index] = s
            elif self.only_row_dep:
                if L_diag:
                    S = torch.diag_embed(s,dim1=-2,dim2=-1)
                else:
                    S = L if root else torch.cholesky(L,lower=True,transpose=False)
                index_c = 0 if scale.size(-3)==1 else index
                scale[...,index_c,:,:] = S
            elif self.only_col_dep:
                if not L_diag:
                    v = (L**2).sum(-1) if root else torch.diagonal(L,dim1=-2,dim2=-1)
                    s = v.sqrt()
                if scale.size(-3)==1 and not s.size(-1)==1:# take average variance
                    s = (s**2).mean(-1).sqrt()
                scale[...,:,index,index] = s
            else:
                raise NotImplementedError # TODO
            try:
                self.scale = scale.detach()
            except TypeError:
                self.scale = torch.nn.Parameter(scale,requires_grad=self.scale.requires_grad)
            # TODO unify self.scale setting in one method with device handling


    @property
    def variances(self):
        return self.get_variances(expand_rows=True,expand_cols=True,root=False)
    @property
    def stddevs(self):
        return self.get_variances(expand_rows=True,expand_cols=True,root=True)


    def set_stddevs(self,value):# float or torch scalar or matrix (batches times nrow times ncol)
        # TODO value check when same_row_dist e.g.
        if self.all_independent:
            self.set_diagonal(value)
        else:
            stddevs = self.param if self.all_independent else (tril(self.scale)**2).sum(-1).sqrt()
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
            return self.loc + eps*self.scale.expand(self.event_shape) # non need to apply .abs() # expand do nothing if not same_col_dist
        elif self.all_dependent:
            eps = _standard_normal(shape[:-2]+torch.Size((self.ncol*self.nrow,1)), dtype=self.loc.dtype, device=self.loc.device)
            Leps = blockmatmul(self,eps,inv=False,X_event_shape=False)
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


    def lrsample(self, X, sample_shape=torch.Size(),ignore_dependence_rows=False,ignore_dependence_cols=False): # samples with independant rows that have mean and covariance as matmul(X,self.rsample())
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
        # because scale may be unbroadcasted (constant variance or contant cov),
        # instead of computing on scale.expand() we multiply the result with
        # the corresponding integer factor.
        if self.all_independent:
            f1 = self.nrow if self.scale.size(-2)== 1 else 1
            f2 = self.ncol if self.scale.size(-1)== 1 else 1
        elif self.only_col_dep:
            f1 = self.nrow if self.scale.size(-3)== 1 else 1
            f2 = 1
        elif self.only_row_dep:
            f1 = 1
            f2 = self.ncol if self.scale.size(-3)== 1 else 1
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


    def entropy(self): # TODO test
        log_det = self.log_det
        H = 0.5 * self.nrow*self.col * (1.0 + log2) + .5*log_det
        return H

    def tensors_to(self,*args,**kwargs):# put to device tensors that are not nn.Parameter
        if not isinstance(self.loc,torch.nn.Parameter):
            self.loc = self.loc.to(*args,**kwargs)
        if not isinstance(self.scale,torch.nn.Parameter):
            self.scale = self.scale.to(*args,**kwargs)
    # What is bette? This function or put a nn.Parameter status to tensors who should never be optimized (the flag requires_grad must be always false) 
                    

class GaussianVector(GaussianMatrix):
    def __init__(self,*dim,dependent = False,iid=False,constant_mean=None,centered = False,stddev=1.0,parameter=True):
        if iid and dependent:
            print("warning set dependent to false")
            dependent = False
        self.d = dim[-1]
        if self.d==1:# also work without this block, just for performance
            iid = True
            dependent = False

        super(GaussianVector, self).__init__(*dim,1,dependent_rows = dependent,dependent_cols = False,same_row_cov=iid,same_col_cov=True,constant_mean=constant_mean,centered = centered ,stddev=stddev,parameter=parameter)

    def rsample(self, sample_shape=torch.Size()):
        return super().rsample(sample_shape).squeeze(-1)

    def lrsample(self, X, sample_shape=torch.Size()): 
        return super().lrsample(X,sample_shape).squeeze(-1)


    @property
    def variances(self):
        return self.get_variances(expand_rows=True,expand_cols=False,root=False).squeeze(-1)

    @property
    def stddevs(self):
        return self.get_variances(expand_rows=True,expand_cols=False,root=True).squeeze(-1)
    @stddevs.setter
    def stddevs(self,X):# float or vector (with size [...,self.d])
        try:
            super().set_stddevs(X.unsqueeze(-1))
        except AttributeError:
            super().set_stddevs(X)

    @property
    def mean(self):
        return super().mean.squeeze(-1)
    @mean.setter
    def mean(self,m):
        super().mean = m.unsqueeze(-1)

    @property
    def tril(self):
        return self.full_tril
    @tril.setter
    def tril(self,L):
        self.full_tril = L

    def log_prob(self, value):
        return super().log_prob(value.unsqueeze(-1)).squeeze(-1)
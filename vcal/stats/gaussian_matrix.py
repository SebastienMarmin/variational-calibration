import math

import torch
from .import constraints
from .distribution import Distribution
from .utils import _standard_normal, lazy_property

from numpy import log as np_log, pi as np_pi
log2pi = np_log(2*np_pi)

def give_inner_data(d,data,init_with_diag,diagonal,homoscedastic,init_with_homosc):
    if 0 in data.size():
        return data
    if init_with_homosc and init_with_diag:
        data_size_list = list(data.size())
        data_size_list[-1] = d
        data = data.expand(torch.Size(data_size_list))
    if diagonal:
        if init_with_diag:
            scale = data
        else:
            scale = (torch.tril(data)**2).sum(-1).sqrt().unsqueeze(-2)
            # keep variances instead of brutally extracting the diagonal
        if homoscedastic:
            scale = (scale**2).mean(-1).sqrt().unsqueeze(-1) # sqrt of average variance
    else:
        if init_with_diag:
            scale = torch.diag_embed(data.squeeze(-2), offset=0, dim1=-2, dim2=-1)
        else:
            scale = data
        if homoscedastic and d>1:
            variances = (torch.tril(scale)**2).sum(-1)
            stds = variances.sqrt()
            common_std = variances.mean(-1).sqrt() # reduced to batch dims
            scale = scale/(stds/common_std.unsqueeze(-1)).unsqueeze(-1)
    return scale

def infer_state_from_data(d,data_size):
    if len(data_size) < 2:
        raise ValueError("data matrix must be at least two-dimensional, "
                            "with optional leading batch dimensions. It can be a [[scalar]] or "
                                "a [vector].")
    if d == 1:
        diagonal = True
        homoscedastic = True
        if data_size[-2]!=1 or data_size[-1]!=1:
            raise ValueError("in dimension one, the two las dimension of data must be singleton (for consistency).")
    else:
        if data_size[-2]==1:
            diagonal = True
            if data_size[-1]==1:
                homoscedastic = True
            elif data_size[-1]==d:
                homoscedastic = False
            else:
                raise ValueError("in diagonal case, the two last dimensions of data must be 1x1 (homoscedastic) or 1xd.")
        elif data_size[-2]==d:
            diagonal = False
            homoscedastic = False
            if data_size[-1]!=d:
                raise ValueError("in non diagonal case, the last dimension of data must be of length d.")
            # can be non-diagonal homoscedastic setting self.homoscedastic = True, but impossible at initialization to keep the number of arbuments low.
        else:
            raise ValueError("the dimension before the last one of data must be either length 1 (diagonal case), or length d (full covariance case). Use additional front dimensions for independent batch points.")
    return diagonal, homoscedastic

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

class GaussianMatrix(Distribution):
    r"""
    Creates a matrix normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precision matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal

    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.

        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """
    #arg_constraints = {'loc': constraints.real_vector,
    #                   'row_scale_tril': constraints.lower_cholesky,
    #                   'col_scale_tril': constraints.lower_cholesky
    #                   }
    support = constraints.real
    has_rsample = True
    has_lrsample = True

    def __init__(self,*dim,independent_rows = True,independent_cols = True,const_row_mean = False,const_col_mean = False, homoscedastic_rows = False,homoscedastic_cols = False ,centered=False,stddev=1.0):
        # delfault is independent rows and columns (but not equally distributed)
        self.nrow = dim[-2]
        self.ncol = dim[-1]
        batch_shape = dim[:-2]
        self.centered = centered
        super(GaussianMatrix, self).__init__(batch_shape, dim[-2:], validate_args=None)
        if centered:
            self.loc = torch.zeros(1).squeeze(0)
        elif const_row_mean and const_col_mean:
            self.loc = torch.nn.Parameter(torch.zeros(*batch_shape,1, 1))
        elif const_row_mean:
            self.loc = torch.nn.Parameter(torch.zeros(*batch_shape,1, self.ncol))
        elif const_col_mean:
            self.loc = torch.nn.Parameter(torch.zeros(*batch_shape,self.nrow, 1))
        else:
            self.loc = torch.nn.Parameter(torch.zeros(*batch_shape,self.nrow, self.ncol))
    
        if self.ncol==1:
            col_param = False
            col_stddev = 1
        else:
            col_param = True
            col_stddev = stddev
        if self.nrow == 1 and self.ncol>1:
            row_param = False
            row_stddev = 1
        else:
            row_param = True
            row_stddev = stddev
        if row_param and col_param:
            row_stddev = stddev**.5
            col_stddev = row_stddev

        self.row_cov = CovarianceMatrix(self.nrow,row_stddev*torch.ones(*batch_shape,1,self.nrow),is_param=row_param)
        self.col_cov = CovarianceMatrix(self.ncol,col_stddev*torch.ones(*batch_shape,1,self.ncol),is_param=col_param)
        
        self.row_cov.diagonal = independent_rows
        self.col_cov.diagonal = independent_cols
        self.row_cov.homoscedastic = homoscedastic_rows
        self.col_cov.homoscedastic = homoscedastic_cols


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


    def set_to(self,W):# but keep specificities (like constant mean, idependenties etc) # handles batch extensions
        if not self.centered:
            if self.loc.size(-1)==1:
                self.loc.data = W.loc.mean(-1).unsqueeze(-1).expand(self.loc.shape).detach().clone()
            elif self.loc.size(-2)==1:
                self.loc.data = W.loc.mean(-2).unsqueeze(-2).expand(self.loc.shape).detach().clone()
            else:
                self.loc.data = W.loc.expand(self.loc.shape).detach().clone()
        self.row_cov.detach_clone(W.row_cov)
        self.col_cov.detach_clone(W.col_cov)

    @property
    def row_scale_tril(self):# row covariance is L L^T
        return self.row_cov.tril

    @property
    def col_scale_tril(self):# column covariance is L^T L
        return self.col_cov.tril

    
    @property
    def row_covariance(self):
        return self.row_cov.covariance

    @property
    def col_covariance(self):
        return self.col_cov.adjoint_covariance # "adjoint" because column covariance is L^T L and not L L^T

    @property
    def mean(self):
        return self.loc

    @property
    def row_variance(self):
        return self.row_cov.variance

    @property
    def col_variance(self): # "adjoint" because column covariance is L^T L and not L L^T
        return self.col_cov.adjoint_variance

    @property
    def row_stddev(self):
        return self.row_cov.stddev
    @row_stddev.setter
    def row_stddev(self,X):
        self.row_cov.stddev=X
    
    @property
    def col_stddev(self): 
        return self.col_cov.adjoint_stddev
    @col_stddev.setter
    def col_stddev(self,X):
        self.col_cov.stddev=X


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        #self.row_cov.L_times(eps)
        return self.loc + self.col_cov.times_L(self.row_cov.L_times(eps))


    def lrsample(self, X): # samples with independant rows that have mean and covariance as matmul(X,self.rsample())
        if self.centered:
            Y_mean = 0
        else:
            Y_mean = torch.matmul(X,self.loc)
        eps = _standard_normal(Y_mean.size(), dtype=self.loc.dtype, device=self.loc.device)
        indep_rows = self.col_cov.times_L(eps)
        XL = self.row_cov.times_L(X)
        Y_centered = (XL**2).sum(-1).sqrt().unsqueeze(-1)*indep_rows
        return Y_mean + Y_centered

    def log_prob(self, value):
        diff = value - self.loc
        U = self.row_cov
        V = self.col_cov
        # No unecessary triangular inversion here:
        exponent_root = V.L_inverse_times(U.L_inverse_times(diff).transpose(-2,-1),transpose=True)
        n = U.d
        m = V.d
        return -.5*(n*m*log2pi+m*U.log_det+n*V.log_det+(exponent_root**2).sum((-1,-2)))

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
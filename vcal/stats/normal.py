import math
from numbers import Number

import torch
from .import constraints
from .exp_family import ExponentialFamily
from .utils import _standard_normal, broadcast_all


class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    has_lrsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.log_scale.exp()
    
    @stddev.setter
    def stddev(self, value):
        self.log_scale.data = value.detach().log()

    @property
    def scale(self):
        return self.log_scale.exp()
    
    @scale.setter
    def scale(self, value):
        self.log_scale.data = value.detach().log()
    
    @property
    def variance(self):
        return (2*self.log_scale).exp()

    def __init__(self, loc, scale, validate_args=None):
        loc_tensor, scale_tensor = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = loc_tensor.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)
        self.loc = torch.nn.Parameter(loc_tensor)
        self.log_scale = torch.nn.Parameter(scale_tensor)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        #new.loc = self.loc.expand(batch_shape)
        #new.log_scale = self.log_scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        new.loc = torch.nn.Parameter(self.loc.expand(batch_shape))
        new.log_scale = torch.nn.Parameter(self.log_scale.expand(batch_shape))
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.stddev.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.stddev


    def lrsample(self, X):
        Y_mean = torch.matmul(X,self.loc)#m_e = self.loc[(None,)*len(X_shape)]
        eps = _standard_normal(Y_mean.size(), dtype=self.loc.dtype, device=self.loc.device)
        Y_sd = torch.matmul(X**2,self.variance).sqrt()
        return Y_mean + eps*Y_sd

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (2*self.log_scale).exp() #(self.scale ** 2)
        log_scale = self.log_scale# if isinstance(self.stddev, Number) else self.log_scale
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.stddev.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.stddev * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + self.log_scale

    @property
    def _natural_params(self):
        var = self.variance()
        return (self.loc / var, -0.5 * var.reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)

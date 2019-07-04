import abc

class CovarianceStructure(metaclass=abc.ABCMeta):
    def __init__(self,dimension):
        super(CovarianceStructure, self).__init__()
        self.dimension = dimension
        NotImplementedError("Subclass of CovarianceStructure should implement __init__().")
    @abc.abstractmethod
    def correlation(self,X1,X2=None, stability=False):
        NotImplementedError("Subclass of CovarianceStructure should implement covariance().")
    @abc.abstractmethod
    def sample_spectrum(self,n_samples):
        NotImplementedError("Subclass of CovarianceStructure should implement sample_spectrum().")


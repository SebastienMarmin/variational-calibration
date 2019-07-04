import abc

class UnivariateInjection(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self,X):
        NotImplementedError("Subclass of should implement.")
    @abc.abstractmethod
    def i(self,X):
        NotImplementedError("Subclass of should implement.")


class log(UnivariateInjection):
    def __call__(self,X):
        return X.log()
    def i(self,X):
        return X.exp()

class root(UnivariateInjection):
    def __call__(self,X):
        return X.sqrt()
    def i(self,X):
        return X**2
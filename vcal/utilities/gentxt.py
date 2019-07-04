import torch
import numpy as np

def gentxt(file,delimiter=";",dtype=None):
    X_np = torch.from_numpy(np.genfromtxt(file,delimiter=delimiter))
    if dtype is None:
        dtype = torch.get_default_dtype()
    X = X_np.type(dtype)
    if len(X.size())==1 or len(X.size())==0:
        X = X.unsqueeze(-1)
    return X

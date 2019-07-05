import torch
import numpy as np
from .distance_matrix import dist_matrix_sq

def lhs_maximin(n,d,rep=10,existingDesign=None) : # 
    with torch.no_grad():
        mask = torch.eye(n, n).byte()
        array1  = torch.linspace(0.,1.-1./n,n)
        criterionMin = 0
        for i in range(rep) :
            currentDesign = array1.unsqueeze(1)
            if d>1:
                for j in range(d-1):
                    randominixes  = torch.randperm(n)
                    vec1Sampled = array1.clone()[randominixes]
                    currentDesign = torch.cat((currentDesign,vec1Sampled.unsqueeze(1)),1)
            currentDesign += torch.rand(n,d)*(4/(5.0*n)-1/(5.0*n))+1/(5.0*n)
            #currentDesign = currentDesign.T
            if (rep>1):
                if existingDesign is None:
                    distMat = dist_matrix_sq(currentDesign)
                else:
                    distMat = dist_matrix_sq(torch.cat((currentDesign,existingDesign),0))
                #torch.fill_diagonal(distMat,np.inf)
                distMat.masked_fill_(mask, np.inf)
                currentCriterion = torch.min(distMat)
            else:
                currentCriterion = np.inf
            if (currentCriterion>criterionMin) :
                result = currentDesign
                criterionMin = currentCriterion
        return result


def factorial(pre,dim,lower=None,upper=None,dtype=None) :
    if dtype is None:
        dtype = torch.get_default_dtype()
    res = (torch.from_numpy(np.float32(np.meshgrid(*[(np.array(range(pre)).astype(float))/(pre-1) for indice in range(dim)])).reshape(dim, pre**dim).T)).type(dtype)
    if dim>1 : res[:,[0,1]] = res[:,[1,0]]
    if lower is None and upper is None:
        return res
    if lower is None:
        lower = torch.zeros(dim).type(dtype)
    if upper is None:
        lower = torch.ones(dim).type(dtype)
    return res*(upper-lower).unsqueeze(0)+lower.unsqueeze(0)
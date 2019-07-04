import torch

def dist_matrix_sq(self, X1, X2=None, stability=False):
    x1 = X1
    if X2 is None:
        if self.stability:
            mu = torch.mean(x1, 0)
            x1.sub_(mu) #inline; subtracting the mean for stability
        D = -2 * torch.mm(x1, x1.t())
        sum_x1x1 = torch.sum(x1*x1, 1).unsqueeze(1).expand_as(D)
        D.add_(sum_x1x1)
        D.add_(sum_x1x1.t())
    else:
        x2 = X2
        if self.stability:
            n = x1.shape[0]
            m = x2.shape[0]
            mu = (m/(n+m))*torch.mean(x2, 0) + (n/(n+m))*torch.mean(x1, 0)
            x1.sub_(mu) #inline; subtracting the mean for stability
            x2.sub_(mu) #inline; subtracting the mean for stability
        D = -2 * torch.mm(x1, x2.t())
        sum_x1x1 = torch.sum(x1*x1, 1).unsqueeze(1).expand_as(D)
        sum_x2x2 = torch.sum(x2*x2, 1).unsqueeze(0).expand_as(D)
        D.add_(sum_x1x1)
        D.add_(sum_x2x2)
    return D



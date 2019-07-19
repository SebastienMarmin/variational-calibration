import torch
from torch.distributions.kl import register_kl
from .gaussian_matrix import GaussianMatrix, L1_inv_L2_frob, blockfrob



@register_kl(GaussianMatrix, GaussianMatrix)
def _kl_matnorm_matnorm(p, q):
    #if p.all_dependent:
    #    term1 = torch.triangular_solve(p.scale,tril(q.scale),upper=False,transpose=False)[0]
    d1 = p.nrow
    d2 = p.ncol
    
    term1 =  L1_inv_L2_frob(q,p)
    
    if p.centered and q.centered:
        term2 = 0
    else:
        diff = p.loc-q.loc
        term2 =  blockfrob(q,diff,inv=True,diag=False,X_event_shape=True).squeeze(-1)
        #term2 =  block_mm(q,diff_column,inv=True,A_inv=Lq_i,fro=True,W2_deterministic_matrix=True)
    term3 = q.log_det - p.log_det

    return .5*(term1+term2-d1*d2+term3)
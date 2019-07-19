import torch
import torch.distributions as distributions
from torch.distributions import kl_divergence
from torch.distributions.kl import register_kl
from torch import matmul
#import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import vcal
from vcal.stats import GaussianMatrix, blockfrob, L1_inv_L2_frob
#from vcal.layers import FourierFeaturesGaussianProcess as GP
#from torch.utils import hooks as hooks
#import matplotlib
#from matplotlib import pyplot as plt


def transpose(q):
    batch_shape = q.batch_shape
    d1 = q.ncol
    d2 = q.nrow
    dependent_rows=q.dependent_cols
    dependent_cols=q.dependent_rows
    same_row_cov = q.same_col_cov
    same_col_cov = q.same_row_cov
    constant_mean=q.loc[0,0] if q.loc.shape[-1]==1 and q.loc.shape[-2]==1 else None
    centered = q.centered

    qT = GaussianMatrix(*batch_shape,d1,d2,dependent_rows=dependent_rows,dependent_cols=dependent_cols,same_row_cov=same_row_cov,same_col_cov=same_col_cov,constant_mean=constant_mean,centered=centered)
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

def LX_t(W,B,inv=False,frob=False,diag=False,X_event_shape=True):
    Ltril = torch.tril(torch_distrib(W).scale_tril)
    if B_event_shape:
        if diag:
            B = B.expand(*B.shape[:-2],d1,d2).contiguous().view(*B.shape[:-2],1,d1*d2).squeeze(-2)
            B = torch.diag_embed(B)
            print("lol")
        else:
            B= B.expand(*B.shape[:-2],d1,d2).contiguous().view(*B.shape[:-2],d1*d2,1)
    else:
        if diag:
            B = torch.diag_embed(B.expand(*B.shape[:-2],1,d1*d2).squeeze(-2))
        else:
            B = B.expand(*B.shape[:-2],d1*d2,-1)
    if inv:
        LX = torch.triangular_solve(B,Ltril,upper=False,transpose=False)[0]
    else:
        LX = matmul(Ltril,B)

    if frob:
        return (LX**2).sum(-2)
    else:
        return LX



if __name__ == '__main__':

    #d1 = 2
    #d2 = 3
    #L = torch.tril(torch.rand(d2,d1,d1))
    #print(L)
    #print(torch.diag_embed(L.unsqueeze(-1).transpose(-1,0).squeeze(0)).size())
    #print(torch.diag_embed(L.unsqueeze(-1).transpose(-1,0).squeeze(0)).transpose(-2,-3).contiguous().view(d1*d2,d1*d2))
    #print(torch.diag_embed(L.unsqueeze(-1).transpose(-1,0).squeeze(0)).unsqueeze(0).transpose(-2,0).squeeze(-2).transpose(-1,-2).transpose(-3,-4).transpose(-1,-2).contiguous().view(d1*d2,d1*d2))
    d1 = 2
    d2 = 3
    n = 2
    batch = [2]
    nmc = [10000]
    shape = torch.Size(batch)+torch.Size((d1,d2))
    X = torch.rand(*nmc,*batch,n,d1)+1
    tru=None
    stds = torch.randn(d1,d2)
    cL=[]
    cL+=[{"dependent_rows":True,"dependent_cols":True,"same_row_cov":False,"same_col_cov":False,"constant_mean":None}]
    
    cL+=[{"dependent_rows":False,"dependent_cols":True,"same_row_cov":False,"same_col_cov":False,"constant_mean":1}]
    cL+=[{"dependent_rows":True,"dependent_cols":False,"same_row_cov":False,"same_col_cov":False,"constant_mean":None}]
    cL+=[{"dependent_rows":False,"dependent_cols":True,"same_row_cov":True,"same_col_cov":False,"constant_mean":None}]
    cL+=[{"dependent_rows":True,"dependent_cols":False,"same_row_cov":False,"same_col_cov":True,"constant_mean":1}]
    
    cL+=[{"dependent_rows":False,"dependent_cols":False,"same_row_cov":True,"same_col_cov":True,"constant_mean":None}]
    cL+=[{"dependent_rows":False,"dependent_cols":False,"same_row_cov":False,"same_col_cov":True,"constant_mean":None}]
    cL+=[{"dependent_rows":False,"dependent_cols":False,"same_row_cov":True,"same_col_cov":False,"constant_mean":1}]
    cL+=[{"dependent_rows":False,"dependent_cols":False,"same_row_cov":False,"same_col_cov":False,"constant_mean":None}]


    """
    def comp(shape):
        l1 = LX_t(q,X,inv=inv,frob=fro,diag=diag,X_event_shape=B_event_shape)
        #if len(l1.shape) > 1:
        #    l1 = l1.squeeze(-1)
        l2 = block_mm(q,X,inv=inv,frob=fro,diag=diag,X_event_shape=B_event_shape)
        if shape:
            print(l2.shape)
        elif False:
            print(torch.round(l1*1000)/1000)
            print(torch.round(l2*1000)/1000)
        else:
            if (l1.shape[0] == l2.shape[0] and len(l1.shape) == 1) or (len(l1.shape) > 1 and len(l2.shape) > 1 and l1.shape[0] == l2.shape[0] and l1.shape[1] == l2.shape[1]):
                print((l1-l2).abs().sum())
            elif len(l1.shape) == 1 and len(l2.shape) == 2 and l2.shape[0]*l2.shape[1]==l1.shape[0]:
                print((l1.view(d1,d2)-l2).abs().sum())
            else:
                print(torch.round(l1*1000)/1000)
                print(torch.round(l2*1000)/1000)
            
    shap=False
    with torch.no_grad():
        for case_q in cL:
            for inv in (True,False):
                for B_event_shape in (True,False):
                    for diag in (False,True):
                        for fro in (True,):
                            print("B_ev:"+str(int(B_event_shape))+",diag:"+str(int(diag))+",fro:"+str(int(fro))+",inv:"+str(int(inv))+",q:"+str((case_q)))
                            q = GaussianMatrix(*shape,**case_q)
                            q.loc = 2*torch.randn_like(q.loc)
                            q.scale = torch.rand_like(q.scale)
                            if B_event_shape:
                                X = torch.rand(*batch,1,d2)
                                comp(shap)
                                X = torch.rand(*batch,d1,1)
                                comp(shap)
                                X = torch.rand(*batch,d1,d2)
                                comp(shap)
                            elif not B_event_shape:
                                if not diag:
                                    X = torch.rand(*batch,1,3)
                                    comp(shap)
                                    X = torch.rand(*batch,d1*d2,3)
                                    comp(shap)
                                else:
                                    X = torch.rand(*batch,1,d1*d2)
                                    comp(shap)
                                    X = torch.rand(*batch,1,1)
                                    comp(shap)

    """



    with torch.no_grad():
        for case_p in cL:
            for case_q in cL:
                print("p: "+str(case_p))
                print("q: "+str(case_q))
                q = GaussianMatrix(*shape,**case_q)
                p = GaussianMatrix(*shape,**case_p)

                p.loc = 2*torch.randn_like(p.loc)
                p.scale = torch.rand_like(p.scale)+1
                q.loc = 2*torch.zeros_like(q.loc)
                q.scale = torch.rand_like(q.scale)+1

                qT = transpose(q)
                pT = transpose(p)

                p_t = p.to_torch_distrib()
                q_t = q.to_torch_distrib()
                pT_t = pT.to_torch_distrib()
                qT_t = qT.to_torch_distrib()
                

                
                dd1 = ((kl_divergence(p,q)-kl_divergence(pT,qT)).abs().sum(-1))
                dd2 = ((kl_divergence(p_t,q_t)-kl_divergence(pT,qT)).abs().sum(-1))
                dd3 = ((kl_divergence(pT_t,qT_t)-kl_divergence(pT,qT)).abs().sum(-1))

                if max(max(dd1,dd2),max(dd2,dd3))>0.00001:
                    print("DKL error :"+str(max(max(dd1,dd2),max(dd2,dd3))))

            X = p.loc+0.002*torch.rand(*batch,d1,d2)
            dd1 = p.log_prob(X).sum(-1)-p_t.log_prob(X.view(*batch,d1*d2)).sum(-1)#
            if dd1 > 0.:
                print("log_prob error :"+str(dd1.item()))


                #print(kl_divergence(p,q))
                #print(kl_divergence(pT,qT))
                #print(kl_divergence(p_t,q_t))
                #print(kl_divergence(pT_t,qT_t))



    with torch.no_grad():
        for case_q in cL:
            X = torch.randn(*batch,n,d1)
            print("q: "+str(case_q))
            q = GaussianMatrix(*shape,**case_q)
            q.loc = 2*torch.randn_like(q.loc)
            q.scale = torch.rand_like(q.scale)
            W = q.sample(torch.Size(nmc))
            print(X.shape)
            print(W.shape)
            V = q.to_torch_distrib().sample(torch.Size(nmc)).view(*nmc,*batch,d1,d2)
            print(("d1"))
            print(torch.matmul(X,W).mean(0))
            print(torch.matmul(X,V).mean(0))
            print(q.lrsample(X,torch.Size(nmc)).mean(0))
            print("\n")
            print(torch.matmul(X,W).var(0))
            print(torch.matmul(X,V).var(0))
            print(q.lrsample(X,torch.Size(nmc)).var(0))





    print("_____________________________")
    with torch.no_grad():
        for case_p in cL:
            for case_q in cL:
                print("p: "+str(case_p))
                print("q: "+str(case_q))
                d1 = 50
                d2 = 50
                p = GaussianMatrix(*batch,d1,d2,**case_p)
                q = GaussianMatrix(*batch,d1,d2,**case_q)


                p.loc = 2*torch.randn_like(p.loc)
                p.scale = torch.rand_like(p.scale)+1
                q.loc = 2*torch.zeros_like(q.loc)
                q.scale = torch.rand_like(q.scale)+1


                pt = p.to_torch_distrib()
                qt = q.to_torch_distrib()
                    
                import time

                start = time.time()
                print(kl_divergence(p,q))
                end = time.time()
                print("tme: "+str(end - start))


                start = time.time()
                print(kl_divergence(pt,qt))
                end = time.time()
                print("the: "+str(end - start))

                #L = torch.tril(torch.randn(d1,d2))
                #print(L)
                #print(L.new_zeros(2, 2))
                #def col_dep_to_full(L):#d1,d2,d2 -> d1*d2,d1*d2

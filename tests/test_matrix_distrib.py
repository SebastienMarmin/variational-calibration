import torch
import torch.distributions as distributions
from torch.distributions import kl_divergence
from torch.distributions.kl import register_kl
from torch import matmul
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import vcal
from vcal.stats import GaussianMatrix
import time

def LX_t(W,B,inv=False,frob=False,diag=False,X_event_shape=True):
    # Naive coding of blockmatmul and blockfrob in vcal.stats.gaussian_matrix
    Ltril = torch.tril(torch_distrib(W).scale_tril)
    if B_event_shape:
        if diag:
            B = B.expand(*B.shape[:-2],d1,d2).contiguous().view(*B.shape[:-2],1,d1*d2).squeeze(-2)
            B = torch.diag_embed(B)
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

    d1 = 2
    d2 = 3
    n = 2
    batch = [] # type: List[int]
    nmc = [10000]
    shape = torch.Size(batch)+torch.Size((d1,d2))
    X = torch.rand(*nmc,*batch,n,d1)+1
    tru=None
    stds = torch.randn(d1,d2)
    cL=[] # type: List[Dict]
    cL+=[{"dependent_rows":True,"dependent_cols":True,"same_row_cov":False,"same_col_cov":False,"constant_mean":None}]
    
    cL+=[{"dependent_rows":False,"dependent_cols":True,"same_row_cov":False,"same_col_cov":False,"constant_mean":1}]
    cL+=[{"dependent_rows":True,"dependent_cols":False,"same_row_cov":False,"same_col_cov":False,"constant_mean":None}]
    cL+=[{"dependent_rows":False,"dependent_cols":True,"same_row_cov":True,"same_col_cov":False,"constant_mean":None}]
    cL+=[{"dependent_rows":True,"dependent_cols":False,"same_row_cov":False,"same_col_cov":True,"constant_mean":1}]
    
    cL+=[{"dependent_rows":False,"dependent_cols":False,"same_row_cov":True,"same_col_cov":True,"constant_mean":None}]
    cL+=[{"dependent_rows":False,"dependent_cols":False,"same_row_cov":False,"same_col_cov":True,"constant_mean":None}]
    cL+=[{"dependent_rows":False,"dependent_cols":False,"same_row_cov":True,"same_col_cov":False,"constant_mean":1}]
    cL+=[{"dependent_rows":False,"dependent_cols":False,"same_row_cov":False,"same_col_cov":False,"constant_mean":None}]

    with torch.no_grad():
        for case_p in cL:
            for case_q in cL:
                print("p: "+str(case_p))
                print("q: "+str(case_q))
                q = GaussianMatrix(*shape,**case_q,parameter=False)
                p = GaussianMatrix(*shape,**case_p,parameter=False)

                p.loc = 2*torch.randn_like(p.loc)
                p.scale = torch.rand_like(p.scale)+1
                q.loc = 2*torch.zeros_like(q.loc)
                q.scale = torch.rand_like(q.scale)+1

                qT = q.transpose()
                pT = p.transpose()

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



    with torch.no_grad():
        for case_q in cL:
            X = torch.randn(*batch,n,d1)
            print("q: "+str(case_q))
            q = GaussianMatrix(*shape,**case_q,parameter=False)
            q.loc = 2*torch.randn_like(q.loc)
            q.scale = torch.rand_like(q.scale)
            W = q.sample(torch.Size(nmc))
            V = q.to_torch_distrib().sample(torch.Size(nmc)).view(*nmc,*batch,d1,d2)
            print(("Samples from different methods"))
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
                d1 = 30
                d2 = 30
                p = GaussianMatrix(*batch,d1,d2,**case_p,parameter=False)
                q = GaussianMatrix(*batch,d1,d2,**case_q,parameter=False)

                p.loc = 2*torch.randn_like(p.loc)
                p.scale = torch.rand_like(p.scale)+1
                q.loc = 2*torch.zeros_like(q.loc)
                q.scale = torch.rand_like(q.scale)+1

                pt = p.to_torch_distrib()
                qt = q.to_torch_distrib()
                    
                start = time.time()
                print(kl_divergence(p,q))
                end = time.time()
                print("time GaussianMatrix: "+str(end - start))

                start = time.time()
                print(kl_divergence(pt,qt))
                end = time.time()
                print("time Naive:          "+str(end - start))

    print("test expand...")
    with torch.no_grad():
        for case_q in cL:
            #print("q: "+str(case_q))
            q = GaussianMatrix(*shape,**case_q,parameter=False)
            q.loc = 2*torch.randn_like(q.loc)
            q.scale = torch.rand_like(q.scale)
            q = q.expand(4,2,*shape)
            #print(q.loc.size())
            #print(q.scale.size())
    print("... no error got raised.")



    with torch.no_grad():
        for case_p in cL:
            for case_q in cL:
                print("p: "+str(case_p))
                print("q: "+str(case_q))
                q = GaussianMatrix(*shape,**case_q,parameter=False)
                p = GaussianMatrix(*shape,**case_p,parameter=False)

                p.loc = 2*torch.randn_like(p.loc)
                p.scale = torch.rand_like(p.scale)+1
                q.loc = 2*torch.zeros_like(q.loc)
                q.scale = torch.rand_like(q.scale)+1

                p.set_covariance(q)
                print(p.scale)
                print(q.scale)
import torch
from torch.autograd import Variable
import numpy as np
from core.utilities import giveGrid 
import matplotlib.pyplot as plt
import matplotlib.lines as mli
import matplotlib.colors as colors
import matplotlib.cm as cmx


##### DISPLAY
def univariateNormalPDF(x,m,v):
    Q_solve = 1/(np.sqrt(v))*(x-m)
    return 1/(np.sqrt(v*2*np.pi))*np.exp(-0.5*(Q_solve**2))

def normalPDF(x,m,v,isRoot=False):
    D2 = v.size()[1]
    q_theta_m = m
    if isRoot:
        q_theta_vroot = v
    else:
        q_theta_vroot = torch.potrf(v)
    Q_solve = torch.inverse(q_theta_vroot).mm((x-q_theta_m.unsqueeze(0)).t())
    qTheta = float(1/((2*np.pi)**(D2/2)*(q_theta_vroot.diag().abs().prod())))*(-0.5*(Q_solve**2).sum(0)).exp()
    return qTheta


def realYPosteriorGivenHyperParam(x,priorMean,priorCovRoot,X, XStar, T, Y, Z, model,theta=None,returnNumpy=True):
    if theta is None:
        epsilon_for_theta_sample = Variable(torch.randn(3, model.layers[0].D2).type(model.dtype), requires_grad=False)
        if model.layers[0].factorized:
                theta = torch.add(torch.mul(epsilon_for_theta_sample.data, torch.exp(model.layers[0].q_theta_logv.data / 2.0)), model.layers[0].q_theta_m.data)
    nGrid = x.size()[0]
    nGridTheta = theta.size()[0]
    n = X.size()[0]
    N = XStar.size()[0]
    D2 = theta.size()[1]
    thetaMean  = priorMean
    thetaCovRoot  = priorCovRoot
    Z_noise_var  = model.log_Z_noise_var.exp().data
    Y_noise_var  = model.log_Y_noise_var.exp().data
    
    XthetaRep = torch.cat(((X.unsqueeze(0)).expand(nGridTheta,-1,-1),theta.unsqueeze(1).expand(-1,n,-1)),2)
    xthetaRep = torch.cat(((x.unsqueeze(0)).expand(nGridTheta,-1,-1),theta.unsqueeze(1).expand(-1,nGrid,-1)),2)
    
    
    Omega_eta_sample = (torch.exp(-model.layers[0].log_eta_lengthscale)*2.**0.5*model.layers[0].epsilon_for_Omega_eta_sample).data
    Omega_delta_sample = (torch.exp(-model.layers[0].log_delta_lengthscale)*2.**0.5*model.layers[0].epsilon_for_Omega_delta_sample).data
    
    Phi_etaX_before_activation = torch.matmul(XthetaRep, Omega_eta_sample)
    Phi_etax_before_activation = torch.matmul(xthetaRep, Omega_eta_sample)
    Phi_deltaX_before_activation = torch.mm(X, Omega_delta_sample)
    Phi_deltax_before_activation = torch.mm(x, Omega_delta_sample)
    Phi_eta_before_activation_computerCode = torch.mm(torch.cat((XStar,T),1),Omega_eta_sample)
    
    if model.kernel == "rbf":
        Phi_etaX = torch.cat((torch.sin(Phi_etaX_before_activation), 
            torch.cos(Phi_etaX_before_activation)), 2) * torch.sqrt(torch.exp(model.layers[0].log_eta_sigma2.data) / model.NRFs)
        Phi_etax = torch.cat((torch.sin(Phi_etax_before_activation), 
            torch.cos(Phi_etax_before_activation)), 2) * torch.sqrt(torch.exp(model.layers[0].log_eta_sigma2.data) / model.NRFs)
        Phi_deltaX = torch.cat((torch.sin(Phi_deltaX_before_activation), 
            torch.cos(Phi_deltaX_before_activation)), 1) * torch.sqrt(torch.exp(model.layers[0].log_delta_sigma2.data) / model.NRFs)
        Phi_deltax = torch.cat((torch.sin(Phi_deltax_before_activation), 
            torch.cos(Phi_deltax_before_activation)), 1) * torch.sqrt(torch.exp(model.layers[0].log_delta_sigma2.data) / model.NRFs)
        Phi_eta_computerCode = torch.cat((torch.sin(Phi_eta_before_activation_computerCode),
            torch.cos(Phi_eta_before_activation_computerCode)), 1) * torch.sqrt(torch.exp(model.layers[0].log_eta_sigma2.data) / model.NRFs)
    
    S11 = (torch.matmul(Phi_etax, Phi_etax.transpose(1,2))+(torch.matmul(Phi_deltax, Phi_deltax.t())))+torch.diag(torch.ones(nGrid).type(model.dtype)*Y_noise_var)
    S2211 = (torch.matmul(Phi_etaX, Phi_etaX.transpose(1,2))+(torch.matmul(Phi_deltaX, Phi_deltaX.t())))+torch.diag(torch.ones(n).type(model.dtype)*Y_noise_var)
    S2212 = torch.matmul(Phi_etaX, (Phi_eta_computerCode.t()))
    S2221 = S2212.transpose(1,2)
    S2222 = torch.matmul(Phi_eta_computerCode,Phi_eta_computerCode.t())+torch.diag(torch.ones(N).type(model.dtype)*Z_noise_var)
    S121 = (torch.matmul(Phi_etax, Phi_etaX.transpose(1,2))+(torch.matmul(Phi_deltax, Phi_deltaX.t())))
    S122 = torch.matmul(Phi_etax, Phi_eta_computerCode.t())
    S12  = torch.cat((S121,S122),2)
    S221 = torch.cat((S2211,S2212),2)
    S222 = torch.cat((S2221,S2222.unsqueeze(0).expand(nGridTheta,-1,-1)),2)
    S22  = torch.cat((S221,S222),1)
    meansYx = torch.zeros(nGridTheta,nGrid).type(model.dtype)
    covsYx = torch.zeros(nGridTheta,nGrid,nGrid).type(model.dtype)
    for i in range(nGridTheta):
        s22m1 = torch.inverse(S22[i,:,:])
        s12TimesS22m1 = torch.matmul(S12[i,:,:],s22m1)
        meansYx[i,:] = torch.matmul(s12TimesS22m1,torch.cat((Y,Z),0)).squeeze()
        covsYx[i,:,:] = S11[i,:,:] - torch.matmul(s12TimesS22m1,S12.transpose(1,2)[i,:,:])
    return meansYx, covsYx


def realThetaPosteriorGivenHyperParam(t,priorMean,priorCovRoot,X, XStar, T, Y, Z, model,returnNumpy=True,log=False):
    nGrid = t.size()[0]
    n = X.size()[0]
    N = XStar.size()[0]
    D2 = t.size()[1]
    thetaMean  = priorMean
    thetaCovRoot  = priorCovRoot
    Z_noise_var  = model.log_Z_noise_var.exp().data
    Y_noise_var  = model.log_Y_noise_var.exp().data
    model.layers[0].forward(Variable(X),Variable(XStar),Variable(T), t.size()[0],Variable(t))
    PhiEta_noStd = model.layers[0].Phi_eta_noStd.data
    PhiDelta_noStd = model.layers[0].Phi_delta_noStd.data
    PhiEtaStar_noStd = model.layers[0].Phi_eta_computerCode_noStd.data
    PhiEta = PhiEta_noStd * torch.sqrt(torch.exp(model.layers[0].log_eta_sigma2.data) / model.layers[0].NRFs)
    PhiDelta = PhiDelta_noStd * torch.sqrt(torch.exp(model.layers[0].log_delta_sigma2.data) / model.layers[0].NRFs)
    PhiEtaStar = PhiEtaStar_noStd * torch.sqrt(torch.exp(model.layers[0].log_eta_sigma2.data) / model.layers[0].NRFs) 
    S11 = ((torch.matmul(PhiDelta, PhiDelta.t()))) + torch.diag(torch.ones(n).type(model.dtype)*Y_noise_var)  + torch.matmul(PhiEta, PhiEta.transpose(1,2))#+ torch.zeros(PhiEta.size()[0],PhiEta.size()[1],PhiEta.size()[1])#
    S22 =torch.matmul(PhiEtaStar,PhiEtaStar.t())+torch.diag(torch.ones(N).type(model.dtype)*Z_noise_var)
    
    S22M1 = torch.inverse(S22)
    S12 = torch.matmul(PhiEta, (PhiEtaStar.t()))
    meanY_givenEta = torch.matmul(torch.matmul(S12,S22M1),Z)
    varY_givenEta = S11 - torch.matmul(torch.matmul(S12,S22M1),S12.transpose(1,2))
    detVarY_givenEtaUnstack = [torch.Tensor([float(np.linalg.det(matr.numpy()))]) for matr in torch.unbind(varY_givenEta)]
    detVarY_givenEta = (torch.stack(detVarY_givenEtaUnstack)).squeeze()
    b = [matr.inverse() for matr in torch.unbind(varY_givenEta)]
    varY_givenEtaM1 = torch.stack(b)
    if log:
        pY_givenEta =(np.log(1/(float(2*np.pi)**(n/2)*(detVarY_givenEta)**(0.5))))+ (-0.5*(torch.matmul(torch.matmul((Y-meanY_givenEta).transpose(1,2),varY_givenEtaM1),(((Y-meanY_givenEta)))).squeeze()))
    else:
        pY_givenEta =(1/(float(2*np.pi)**(n/2)*(detVarY_givenEta)**(0.5))*
                  (-0.5*(torch.matmul(torch.matmul((Y-meanY_givenEta).transpose(1,2),varY_givenEtaM1),
                                      (((Y-meanY_givenEta)))).squeeze())).exp())
    if D2==1:
        elementaryVolume = float((t.max()-t.min())/nGrid)
    else:
        elementaryVolume = float(np.prod([vec.max()-vec.min() for vec in torch.unbind(t.t())])/nGrid)
    (c1ixc0,cloneInutile) = torch.trtrs((t-thetaMean).t(),thetaCovRoot)
    if log:
        pTheta_givenDataTrue = ((c1ixc0**2).sum(0)*(-0.5))+pY_givenEta
        res = pTheta_givenDataTrue#-(pTheta_givenDataTrue.exp().sum(0)*elementaryVolume).log()
    else:
        pTheta_givenDataTrue = ((c1ixc0**2).sum(0)*(-0.5)).exp()*pY_givenEta
        res = pTheta_givenDataTrue/(pTheta_givenDataTrue.sum()*elementaryVolume)
    if returnNumpy:
        return  res.numpy()
    return  res

def plotCalibDomain(X, XStar, T, Y, Z, model,lower2,upper2,priorMean,priorCovRoot,trueTheta=None,axialPre=None,outputFile=None,log=False,subplot=None,returnPlot = False):
    dtype=X.type()
    D2 = lower2.size()[0]
    if  axialPre is None:
        axialPre = 20
    tGrid  = giveGrid(axialPre,D2).type(dtype)*(upper2-lower2).unsqueeze(0)+lower2.unsqueeze(0)
    qThetaExact = realThetaPosteriorGivenHyperParam(tGrid,priorMean,priorCovRoot,X, XStar, T, Y, Z, model,returnNumpy=False,log=log)
    qTheta = normalPDF(tGrid,model.layers[0].q_theta_m.data,torch.diag((model.layers[0].q_theta_logv.data/2).exp()),isRoot=True)
    pTheta = normalPDF(tGrid,model.layers[0].prior_theta_m.data,torch.diag((model.layers[0].prior_theta_logv.data/2).exp()),isRoot=True)
    output = plt.figure()
    if subplot is not None:
        outputPlt = subplot
    else:
        outputPlt = plt
    if (D2==1):
        t = np.linspace(lower2[0],upper2[0],axialPre)
        outputPlt.plot(t,qThetaExact.numpy(),c="green")
        outputPlt.plot(t,qTheta.numpy(),c="blue")
        outputPlt.plot(t,pTheta.numpy(),c="black")
        if trueTheta is not None:
            outputPlt.plot([trueTheta[0],trueTheta[0]],
                 [0,torch.cat((qThetaExact,qTheta,pTheta),0).max()],color="red")
    if (D2==2):
        t1 = np.linspace(lower2[0],upper2[0],axialPre)
        t2 = np.linspace(lower2[1],upper2[1],axialPre)
        outputPlt.contourf(t1, t2, qThetaExact.view(axialPre,axialPre).t().numpy())#
        if trueTheta is not None:
            outputPlt.scatter(trueTheta[0],trueTheta[1],c="red")
        outputPlt.contour(t1, t2, qTheta.view(axialPre,axialPre).t().numpy(),colors='black')
    if subplot is None:
        outputPlt.show()
    if outputFile is not None:
        output.savefig(outputFile, bbox_inches='tight')
    if returnPlot:
        return tGrid, qThetaExact,output
    return tGrid, qThetaExact
        
def plotVariableDomain(X, XStar, T, Y, Z, model,lower1,upper1,priorMean,priorCovRoot,thetaNum,thetaAnaly,axialPre=None,outputFile=None):
    x = giveGrid(axialPre,1).type(model.dtype)*(upper1-lower1).unsqueeze(0)+lower1.unsqueeze(0)
    meanYx,covYx = realYPosteriorGivenHyperParam(x,model.layers[0].prior_theta_m.data,torch.diag(model.layers[0].prior_theta_logv.data.exp()),
                              X, XStar, T, Y, Z, model,thetaAnaly,returnNumpy=True)
    for i in range(meanYx.size()[0]):
        plt.plot(x.numpy(),meanYx[i,:].numpy(),c="black")
    plt.scatter(XStar.numpy(),Z.numpy(),c="grey")#*0+min(q05Yx)
    for i in range(meanYx.size()[0]):
        stdYx = torch.diag(covYx[i,:,:]).sqrt()
        q95Yx = meanYx[i,:]+2*stdYx
        q05Yx = meanYx[i,:]-2*stdYx
        plt.plot(x.numpy(),q95Yx.numpy(),c="red")
        plt.plot(x.numpy(),q05Yx.numpy(),c="red")
    plt.scatter(X.numpy(),Y.numpy())


    meanYx_vi, varYx_vi,meanZx_vi,varZx_vi = model.layers[0].predictGivenTheta(
        Variable(x,requires_grad=False),
        Variable(XStar,requires_grad=False),
        Variable(T,requires_grad=False),
        thetaNum.size()[0],
        Variable(thetaNum,requires_grad=False),
        forcePhiCompute=True)
    for i in range(meanYx_vi.size()[0]):
        plt.plot(x.numpy(),meanYx_vi[i,:,0].data.numpy(),c="green")
    for i in range(meanYx_vi.size()[0]):
        stdYx = varYx_vi[i,:,0].sqrt().data
        q95Yx = meanYx_vi[i,:,0].data+2*stdYx
        q05Yx = meanYx_vi[i,:,0].data-2*stdYx
        plt.plot(x.numpy(),q95Yx.numpy(),c="blue")
        plt.plot(x.numpy(),q05Yx.numpy(),c="blue")
    plt.show()
    return x,meanYx_vi,varYx_vi,meanZx_vi,varZx_vi
    
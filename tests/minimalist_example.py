import os, sys
from os.path import isfile, join
from os import listdir
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
#from collections import OrderedDict

import timeit
#import intertools
import vcal
from vcal.nets import AdditiveDiscrepancy, RegressionNet
from vcal.layers import FourierFeaturesGaussianProcess as GP
from vcal.stats import GaussianVector
from vcal.utilities import MultiSpaceBatchLoader,SingleSpaceBatchLoader, gentxt, VcalException
from vcal.vardl_utils.initializers import IBLMInitializer

import json

import matplotlib
from matplotlib import pyplot as plt
import timeit


from vcal.utilities.designs_of_experiments import lhs_maximin, factorial




def real_computer_experiment(x,t):
    return (       (  3  *x + 5*t).cos()
            -      (-1/10*x -   t).cos()
            +  4/3*(  3  *x - 4*t).cos()
            -  2/3*(  3/2*x - 6*t).cos())
def real_discrepancy(x):
    return (   1/5*(  5/2*x).cos()
            - 1/10*(  5/2*x).sin()
            +  1/5*(  2/6*x).cos())

# TODO description of the data



def display_model(XX,YY_pred,YY=None,X=None,Y=None,i=0,sample_paths=10,plt_std=False):
    input_dim = XX.shape[1]
    nmc_test = YY_pred.shape[0]
    if input_dim==1:
        if YY is not None:
            plt.plot(XX[:,0].numpy(),YY[:,i].numpy())
        if sample_paths>0:
            n_samples = min(sample_paths,nmc_test)
            for j in range(n_samples):
                plt.plot(XX[:,0].numpy(),YY_pred[j,:,i].detach().numpy(),c="orange")
        fmean = YY_pred[:,:,i].detach().mean(0)
        fsd = YY_pred[:,:,i].detach().var(0).sqrt()
        plt.plot(XX[:,0].numpy(),fmean.numpy(),c="black")
        plt.plot(XX[:,0].numpy(),(fmean+2*fsd).numpy(),c="grey")
        plt.plot(XX[:,0].numpy(),(fmean-2*fsd).numpy(),c="grey")
        if X is not None and Y is not None:
            plt.scatter(X[:,0].numpy(),Y[:,i].numpy(),zorder=10)
    if input_dim==2:
        if plt_std:
            fdispl = YY_pred[:,:,i].detach().var(0).sqrt()
        else:
            fdispl = YY_pred[:,:,i].detach().mean(0)
        plt.tricontourf(XX[:,0].squeeze().numpy(), XX[:,1].squeeze().numpy(), fdispl.squeeze().numpy())
        CS = plt.tricontour(XX[:,0].squeeze().numpy(), XX[:,1].squeeze().numpy(), fdispl.squeeze().numpy(),colors="black")
        #plt.clabel(CS, fontsize=9, inline=1)
        if X is not None:
            plt.scatter(X[:,0].squeeze().numpy(),X[:,1].squeeze().numpy(),color="black",zorder=10)
        



def display_calibration(X,Y,XX,YY,XX_star,TT,ZZ,model):
 

    #XGridTorch = utilities.giveGrid(axialPre,D1).type(dtype)
    #XTGridTorch = utilities.giveGrid(axialPre,D1+D2).type(dtype)
    #XXTGridTorch = XTGridTorch[,0:D1]
    #XTTGridTorch = XTGridTorch[,D1:(D1+D2)]
    #predict = model.layers[0].forward(Variable(XGridTorch),Variable(XXTGridTorch),Variable(XTTGridTorch), nMC_test)
    #print(predict)
    if False :
        marginRatio = 0.05
        margin_t=(T.max()-T.min())*marginRatio
        margin_x=(X.max()-X.min())*marginRatio
        tAxis = np.linspace(T.min()-margin_t,T.max()+margin_t,axialPre)
        xAxis = np.linspace(0-margin_x,1+margin_x,axialPre)
        tAxisTorch = torch.from_numpy(tAxis).type(dtype).unsqueeze(1)
        xAxisTorch = torch.from_numpy(xAxis).type(dtype).unsqueeze(1)
        xtgridTorch = utilities.giveGrid(axialPre,2).type(dtype)
        xGridTorch = xtgridTorch[:,0].unsqueeze(1)
        tGridTorch  = ((xtgridTorch[:,1])*(tAxisTorch.max()-tAxisTorch.min())+tAxisTorch.min()).unsqueeze(1)
        qTheta = univariateNormalPDF(tAxis,q_theta_m.squeeze().numpy(),q_theta_v.numpy())
        pTheta = univariateNormalPDF(tAxis,dataPrior.thetaMean.numpy(),dataPrior.thetaCovRoot.squeeze().numpy()**2)
        qThetaExact = realThetaPosteriorGivenHyperParam(tAxisTorch,dataPrior,model)
        dataLikely = np.interp(T.squeeze().numpy(),tAxis,qThetaExact)
        dataLikelyNorm = dataLikely/np.max(qThetaExact)
        plt.subplot(221)
        for i in range(N) :
            plt.scatter(T[i,0],0,c=[float(dataLikelyNorm[i]),float(dataLikelyNorm[i]),float(dataLikelyNorm[i])])
            plt.scatter(T[i,0],0,facecolors='none', edgecolors='black')
        plt.plot([float(trueTheta),float(trueTheta)],[0,np.max([np.max([np.max(qThetaExact),np.max(qTheta)]),np.max(pTheta)])],color="red")
        plt.plot(tAxis,qTheta,color="blue")
        plt.plot(tAxis,pTheta,color="black")
        plt.plot(tAxis,qThetaExact,color="green")
        plt.margins(x=0)
        plt.ylabel("Probability density")

        predict = model.layers[0].forward(Variable(xAxisTorch),Variable(xGridTorch),Variable(tGridTorch), nMC_test)
        Ypredict = predict[:,0:(axialPre),]
        Zpredict = predict[:,axialPre:predict.size(1),:]

        plt.subplot(223)
        plt.contourf(tAxis, xAxis, (torch.from_numpy(qThetaExact).unsqueeze(1).expand(axialPre,axialPre)).t().numpy(), cmap=plt.get_cmap('gray'))
        plt.margins(y=0)
        rf = dataPrior.computerCode(xGridTorch,tGridTorch)
        jet = cm = plt.get_cmap('rainbow')
        cNorm  = colors.Normalize(vmin=rf.min(), vmax=rf.max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        for i in range(n) :
            plt.plot((torch.Tensor([tAxisTorch.min(),tAxisTorch.max()]).unsqueeze(0).expand(n,2).t()).numpy()[:,i],X.expand(n,2).t().numpy()[:,i],color=scalarMap.to_rgba(Y.squeeze().numpy()[i]))
        for i in range(N) :
            plt.scatter(T[i], XStar[i],color=scalarMap.to_rgba(Z.squeeze().numpy()[i]))
        plt.plot([float(trueTheta),float(trueTheta)],[np.min(xAxis),np.max(xAxis)],color="red")
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$y$')

        plt.subplot(222)
        plt.contourf(tAxis,xAxis, rf.view(axialPre,axialPre).numpy(), vmin=rf.min(),vmax=rf.max(), cmap=cm)
        CS = plt.contour(tAxis,xAxis, rf.view(axialPre,axialPre).numpy(),colors="black")
        plt.clabel(CS, fontsize=9, inline=1)
        plt.ylabel(r"y")

        plt.subplot(224)
        plt.contourf(tAxis, xAxis, Zpredict.data.squeeze().mean(0).view(axialPre,axialPre).numpy(),vmin=rf.min(),vmax=rf.max(), cmap=cm)
        CS = plt.contour(tAxis,xAxis, Zpredict.data.squeeze().mean(0).view(axialPre,axialPre).numpy(),colors="black")
        plt.clabel(CS, fontsize=9, inline=1)
        plt.scatter(T.squeeze().numpy(),XStar.squeeze().numpy(),color="black",zorder=1)
        plt.xlabel(r'$\theta$')
        plt.show()
        plt.close()
        plt.plot(xAxis,Ypredict.squeeze().t().data.numpy(),color="gray",alpha=0.1)
        plt.plot(xAxis,Ypredict.squeeze().mean(0).data.numpy(),color="black")
        plt.scatter(X.squeeze().numpy(),Y.squeeze().numpy(),color="black",zorder=5)
        plt.show()
        plt.close()
        plt.contourf(tAxis, xAxis, Zpredict.data.squeeze().var(0).view(axialPre,axialPre).numpy(),vmin=0)
        plt.scatter(T.squeeze().numpy(),XStar.squeeze().numpy(),color="black",zorder=1)
        #CS = plt.contour(np.linspace(tgrid.min(),tgrid.max(),preGrid), np.linspace(xgrid.min(),xgrid.max(),preGrid), rf.view(preGrid,preGrid).numpy(),colors="black")
    plt.show()
        #plt.close()



##### DISPLAY
# Some functions for plotting
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

def plotCalibDomain(X, XStar, T, Y, Z, model,lower2,upper2,priorMean,priorCovRoot,trueTheta=None,axialPre=None,outputFile=None,log=False,subplot=None,returnPlot = False):
    dtype=X.type()
    D2 = lower2.size()[0]
    if  axialPre is None:
        axialPre = 20
    tGrid  = factorial(axialPre,D2).type(dtype)*(upper2-lower2).unsqueeze(0)+lower2.unsqueeze(0)
    qThetaExact = None
    qTheta = normalPDF(tGrid,model.calib_posterior.loc.data.squeeze(1),model.calib_posterior.tril,isRoot=True)
    pTheta = normalPDF(tGrid,model.calib_prior.loc.data.squeeze(1),model.calib_prior.tril,isRoot=True)
    output = plt.figure()
    if subplot is not None:
        outputPlt = subplot
    else:
        outputPlt = plt
    if (D2==1):
        t = np.linspace(lower2[0],upper2[0],axialPre)
        outputPlt.plot(t,qTheta.detach().numpy(),c="blue")
        outputPlt.plot(t,pTheta.detach().numpy(),c="black")
        if trueTheta is not None:
            outputPlt.plot([trueTheta[0],trueTheta[0]],[0,torch.cat((qTheta,pTheta),0).max()],color="red")
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

if __name__ == '__main__':
    seed = 0
    vcal.utilities.set_seed(seed)

    # Setup dataset
    obs_size = 8
    run_size = 15
    input_dim = 2
    calib_dim = 1
    true_calib = torch.rand(calib_dim)

    X = torch.linspace(0,1,obs_size).unsqueeze(1)
    Y = real_computer_experiment(X,true_calib) + real_discrepancy(X)
    X_star_T = lhs_maximin(run_size,input_dim,rep=500)
    X_star = X_star_T[:,:(input_dim-calib_dim)]
    T      = X_star_T[:,-calib_dim:]
    Z = real_computer_experiment(X_star,T)

    test_obs_size = 50
    axial_pre = 50
    test_run_size = axial_pre**input_dim
    XX      = torch.linspace(0,1,test_obs_size).unsqueeze(1)
    YY = real_computer_experiment(XX,true_calib) + real_discrepancy(XX)
    XX_star_axis = torch.linspace(0,1,axial_pre).unsqueeze(1)
    TT_axis      = torch.linspace(0,1,axial_pre).unsqueeze(1)
    XX_star_TT = factorial(axial_pre,input_dim)
    XX_star= XX_star_TT[:,:(input_dim-calib_dim)]
    TT     = XX_star_TT[:,-calib_dim:]
    ZZ = real_computer_experiment(XX_star,TT)
    train_obs_loader = DataLoader(TensorDataset(X,Y),
                                    batch_size=obs_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)
    train_run_loader = DataLoader(TensorDataset(X_star,T,Z),
                                    batch_size=run_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)
    test_obs_loader = DataLoader(TensorDataset(XX,YY),
                                    batch_size=test_obs_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)
    test_run_loader = DataLoader(TensorDataset(XX_star,TT,ZZ),
                                    batch_size=test_run_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)

    train_data_loader = MultiSpaceBatchLoader(train_obs_loader,train_run_loader)
    test_data_loader  = MultiSpaceBatchLoader(test_obs_loader,  test_run_loader)

    nmc_train = 5
    nmc_test  = 40
    nfeatures_run = 50
    nfeatures_obs = 50
    noise_std_run = 0.01
    noise_std_obs = 0.03

    eta   = GP(input_dim,           1,nfeatures=nfeatures_run, nmc_train=nmc_train, nmc_test=nmc_test)
    delta = GP(input_dim-calib_dim, 1,nfeatures=nfeatures_obs, nmc_train=nmc_train, nmc_test=nmc_test)
    eta.variances   = 1  *torch.ones(1)
    delta.variances = .05*torch.ones(1)
    eta.lengthscales   = .8*torch.ones(1)
    delta.lengthscales = .1*torch.ones(1)
    computer_model = RegressionNet(eta)
    discrepancy    = RegressionNet(delta)
    computer_model.likelihood.row_stddev = noise_std_run
    discrepancy.likelihood.row_stddev    = noise_std_obs

    calib_prior = GaussianVector(calib_dim,iid=True,constant_mean=.5,parameter=False)
    calib_prior.stddev=1
    calib_posterior = GaussianVector(calib_dim)
    #calib_posterior.set_to(calib_prior)
    #calib_posterior.row_cov.parameter.detach()
    
    calib_posterior.stddev = .15
    calib_posterior.loc.data = torch.randn(1)

    model = AdditiveDiscrepancy(computer_model,discrepancy,calib_prior,calib_posterior,true_calib=true_calib)

    ### Initialization of the computer model
    init_batchsize_run = run_size # all data are taken for initialization
    init_data_run,_=random_split(train_data_loader.loaders[1].dataset,[init_batchsize_run,run_size-init_batchsize_run])
    dataloader_run_for_init=SingleSpaceBatchLoader(DataLoader(init_data_run,batch_size=init_batchsize_run),cat_inputs=True)
    computer_model_initializer=IBLMInitializer(computer_model,dataloader_run_for_init,noise_var =noise_std_run**2)
    computer_model_initializer.initialize()
    
    lr = .02
    iterations_free_noise = 100
    device = None
    verbose = False
    lr_calib= 0.1
    outdir = vcal.vardl_utils.next_path('workspace/minimalist_example/%s/' % ('run-%04d/'))
    tb_logger = vcal.vardl_utils.logger.TensorboardLogger(path=outdir, model=model, directory=None)
    trainer = vcal.learning.Trainer(model, 'Adam', {'lr': lr}, train_data_loader,test_data_loader,device, seed,  tb_logger,debug=verbose,lr_calib=lr_calib)
    for p in model.likelihood.parameters():
        p.requires_grad = False
    delta.fix_hyperparameters()
    print(model.string_parameters_to_optimize())
    test_interval = iterations_free_noise
    trainer.fit(iterations_free_noise, test_interval, 1, time_budget=60//2)

    # Figure
    with torch.no_grad():
        calib_mean = model.calib_posterior.loc.detach()
        eta.local_reparam   = False
        delta.local_reparam = False
        YY_pred , ZZ_pred = model([XX,XX_star,TT])
        plt.figure(num=None, figsize=(10, 4), dpi=160)
        plt.subplot(121)
        display_model(XX,YY_pred,X=X,Y=Y)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$Y$')
        plt.subplot(122)
        display_model(torch.cat((XX_star,TT),-1),ZZ_pred,X=torch.cat((X_star,T),-1))
        plt.plot([torch.min(XX),torch.max(XX)],[true_calib.item(),true_calib.item()],color="black", label='True calibration value',zorder = 11)
        plt.plot([torch.min(XX),torch.max(XX)],[calib_mean.item(),calib_mean.item()],color="orange", label='Posterior of calibration value')
        qTheta = calib_posterior.log_prob(TT_axis).exp()
        plt.contourf([XX.min(),XX.max()],TT_axis.squeeze().numpy(), qTheta.unsqueeze(1).expand(axial_pre,2).numpy(), cmap=plt.get_cmap('Wistia'),alpha=1)
        plt.xlabel(r'$Z(x,t)$       $x$          ')
        plt.ylabel(r'$t$')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=2, fancybox=True).set_zorder(12)
    plt.savefig("minimalist_example.pdf",format="pdf")
    plt.close()
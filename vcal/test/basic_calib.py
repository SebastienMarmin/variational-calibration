import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
sys.path.append(os.path.join(os.path.dirname(__file__), "../../vcal"))
import vcal
from vcal.nets import AdditiveDiscrepancy, RegressionNet
from vcal.layers import FourierFeaturesGaussianProcess as GP
from vcal.stats import GaussianVector
from vcal.utilities import MultiSpaceBatchLoader

import matplotlib
from matplotlib import pyplot as plt


def gentxt(file,delimiter=";",dtype=None):
    X_np = torch.from_numpy(np.genfromtxt(file,delimiter=delimiter))
    if dtype is None:
        dtype = torch.get_default_dtype()
    X = X_np.type(dtype)
    if len(X.size())==1 or len(X.size())==0:
        X = X.unsqueeze(-1)
    return X

def setup_dataset():# TODO common setup
    dataset_unidir = join(args.dataset_dir, args.dataset)
    onlyfiles = [f for f in listdir(dataset_unidir) if isfile(join(dataset_unidir, f))]
    tensor_names = ("X","Y","XStar","T","Z")
    extension = "csv"
    delimiter = ";"
    files = [X+"."+extension for X in tensor_names]
    if not set(files).issubset(onlyfiles):
        logger.error("Dataset in "+str(dataset_unidir)+" must contains "+str(set(file)))
    else:
        X,Y,X_star,T,Z = (gentxt(join(dataset_unidir,f),delimiter=delimiter) for f in files)
        calib_dim = T.size()[1]
        input_dim = X.size()[1]+calib_dim
        output_dim = Y.size()[1]
        test_files = ["test_"+f for f in files]
        observations  = TensorDataset(X,Y)
        computer_runs = TensorDataset(X_star,T,Z)
        if set(test_files).issubset(onlyfiles):
            test_X,test_Y,test_X_star,test_T,test_Z=(gentxt(join(dataset_unidir,f),delimiter) for f in train_files)
            test_observations  = TensorDataset(test_X,test_Y)
            test_computer_runs = TensorDataset(test_X_star,test_T,test_Z)
            train_observations=observations
            train_computer_runs = computer_runs
        else:
            obs_size   = len(observations)
            run_size   = len(computer_runs)
            train_obs_size = int(args.split_ratio_obs * obs_size)
            train_run_size = int(args.split_ratio_run * run_size)
            test_obs_size = obs_size - train_obs_size
            test_run_size = run_size - train_run_size
            if test_run_size==0:
                train_computer_runs = test_computer_runs = computer_runs
            else:
                train_computer_runs, test_computer_runs = random_split(computer_runs, [train_run_size, test_run_size])
            if test_run_size==0:
                train_observations = test_observations = observations
            else:
                train_observations,  test_observations  = random_split(observations,  [train_obs_size, test_obs_size])

        logger.info('Loading dataset from %s' % str(dataset_unidir))

        train_obs_loader = DataLoader(train_observations,
                                    batch_size=args.batch_size_obs,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)

        train_run_loader = DataLoader(train_computer_runs,
                                    batch_size=args.batch_size_run,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)

        test_obs_loader = DataLoader(test_observations,
                                    batch_size=args.batch_size_obs,# * torch.cuda.device_count(),
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)

        test_run_loader = DataLoader(test_computer_runs,
                                    batch_size=args.batch_size_run,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)

        true_calib_name = "theta"+"."+extension
        if true_calib_name in onlyfiles:
            true_calib = gentxt(join(dataset_unidir,true_calib_name),delimiter)
        else:
            true_calib = None
        return train_obs_loader,train_run_loader,test_obs_loader,test_run_loader, input_dim, calib_dim,output_dim,true_calib


def display(model,XX,YY=None,X=None,Y=None,i=0,file_path=None,format="pdf",sample_paths=True):
    input_dim = XX.shape[1]
    if input_dim==1:
        if YY is not None:
            plt.plot(XX[:,0].numpy(),YY[:,i].numpy())
        model.eval()
        with torch.no_grad():
            fw = model(XX)
            nmc_test = fw.shape[0]
            if sample_paths==True:
                for j in range(nmc_test):
                    plt.plot(XX[:,0].numpy(),fw[j,:,i].detach().numpy(),c="orange")
            fmean = fw[:,:,i].detach().mean(0)
            fsd = fw[:,:,i].detach().var(0).sqrt()
            plt.plot(XX[:,0].numpy(),fmean.numpy(),c="black")
            plt.plot(XX[:,0].numpy(),(fmean+2*fsd).numpy(),c="grey")
            plt.plot(XX[:,0].numpy(),(fmean-2*fsd).numpy(),c="grey")
        if X is not None and Y is not None:
            plt.scatter(X[:,0].numpy(),Y[:,i].numpy(),zorder=10)
    if file_path is None:
        file_path = 'figure.pdf'
    matplotlib.pyplot.savefig(file_path)


def giveGrid(pre,dim,lower=None,upper=None,dtype=None) :
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





##### DISPLAY

import matplotlib.pyplot as plt
import matplotlib.lines as mli
import matplotlib.colors as colors
import matplotlib.cm as cmx

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
        epsilon_for_theta_sample = torch.randn(3, model.layers[0].D2).type(model.dtype)
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
    Z_noise_var  = model.computer_model.likelihood.row_cov.variance.detach()
    Y_noise_var  = model.discrepancy.likelihood.row_cov.variance.detach()
    model.forward(X,XStar,T, t.size()[0],(t))
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
    #qThetaExact = realThetaPosteriorGivenHyperParam(tGrid,priorMean,priorCovRoot,X, XStar, T, Y, Z, model,returnNumpy=False,log=log)
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
        #outputPlt.plot(t,qThetaExact.numpy(),c="green")
        outputPlt.plot(t,qTheta.numpy(),c="blue")
        outputPlt.plot(t,pTheta.numpy(),c="black")
        if trueTheta is not None:
            #outputPlt.plot([trueTheta[0],trueTheta[0]],[0,torch.cat((qThetaExact,qTheta,pTheta),0).max()],color="red")
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


    with torch.no_grad:
        meanYx_vi, varYx_vi,meanZx_vi,varZx_vi = model.layers[0].predictGivenTheta(
            x,XStar,T,thetaNum.size()[0],thetaNum,forcePhiCompute=True)
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


if __name__ == '__main__':
    args = parse_args()
    outdir = vcal.vardl_utils.next_path('%s/%s/%s/' % (args.outdir, args.dataset, args.model) + 'run-%04d/')
    try:
        os.makedirs(outdir)
    except OSError:  
        print ("Creation of the directory %s failed" % outdir)
    else:  
        print ("Successfully created the directory %s " % outdir)
    if args.verbose:
            logger = vcal.vardl_utils.setup_logger('vcal', outdir, 'DEBUG')
    else:
            logger = vcal.vardl_utils.setup_logger('vcal#', outdir)
    logger.info('Configuration:')
    for key, value in vars(args).items():
            logger.info('  %s = %s' % (key, value))

    # Save experiment configuration as yaml file in logdir
    with open(outdir + 'experiment_config.json', 'w') as fp:
            json.dump(vars(args), fp, sort_keys=True, indent=4)
    vcal.utilities.set_seed(args.seed)
    train_obs_loader,train_run_loader,test_obs_loader,test_run_loader, input_dim, calib_dim,output_dim,true_calib = setup_dataset()
    train_data_loader = MultiSpaceBatchLoader(train_obs_loader,train_run_loader)
    test_data_loader  = MultiSpaceBatchLoader(test_obs_loader,  test_run_loader)
    nmc_train = args.nmc_train
    nmc_test  = args.nmc_test
    eta   = GP(input_dim,           output_dim,nfeatures=args.nfeatures_run, nmc_train=nmc_train, nmc_test=nmc_test)
    delta = GP(input_dim-calib_dim, output_dim,nfeatures=args.nfeatures_obs, nmc_train=nmc_train, nmc_test=nmc_test)
    eta.prior_variances.data   =   eta.pf(torch.ones(1)) # TODO user friendly ?
    delta.prior_variances.data = delta.pf(.001*torch.ones(1)) #.1
    eta.lengthscales.data = eta.pf(.34*torch.ones(1))
    delta.lengthscales.data = delta.pf(.0354*torch.ones(1))
    eta.optimize(False)
    delta.optimize(False)
    npts_obs = len(train_data_loader.loaders[0].dataset)
    npts_run = len(train_data_loader.loaders[1].dataset)
    
    chol_dim_obs = min(args.chol_dim,npts_obs)
    chol_dim_run = min(args.chol_dim,npts_run)

    init_data_obs, _ =  random_split(train_data_loader.loaders[0].dataset,[chol_dim_obs,npts_obs-chol_dim_obs])
    init_data_run, asas =  random_split(train_data_loader.loaders[1].dataset,[chol_dim_run,npts_run-chol_dim_run])

    
    computer_model = RegressionNet(eta)
    discrepancy   = RegressionNet(delta)
    computer_model.likelihood.row_stddev = args.noise_std_run
    discrepancy.likelihood.row_stddev    = args.noise_std_obs
    calib_prior = GaussianVector(calib_dim,homoscedastic=True,const_mean=True)
    calib_prior.loc.data=torch.ones(1,1)*0.5
    calib_prior.row_cov.parameter.data=torch.ones(1,1)*50 # TODO user friendly?
    calib_prior.optimize(False)
    calib_posterior = GaussianVector(calib_dim)
    calib_posterior.set_to(calib_prior)
    calib_posterior.row_cov.parameter.detach()
    calib_posterior.loc.data = torch.ones(1,1)*torch.randn(1).item()*.5+.5

    calib_posterior.stddev = .05
    calib_posterior.cov.parameter.requires_grad = False
    model = AdditiveDiscrepancy(computer_model,discrepancy,calib_prior,calib_posterior,true_calib=true_calib)

    X, Y = next(iter(DataLoader(init_data_obs,batch_size=chol_dim_obs)))
    X_star, T, Z = next(iter(DataLoader(init_data_run,batch_size=chol_dim_run)))

    model.initialize([X,X_star,T],[Y,Z])

    tb_logger = vcal.vardl_utils.logger.TensorboardLogger(path=outdir, model=model, directory=None)
    trainer = vcal.learning.Trainer(model, 'Adam', {'lr': args.lr}, train_data_loader,test_data_loader,
                        args.device, args.seed, tb_logger, debug=args.verbose)
    for p in model.likelihood.parameters():
        p.requires_grad=False    
    print(model.string_parameters_to_optimize())
    def perturbation(m):
        m.calib_posterior.loc.data = torch.ones(1,1)*torch.randn(1).item()*.5+.5
        return m
    trainer.multistart(perturbation,100,nstarts=2)


    """ trainer.fit(args.iterations_fixed_noise, args.test_interval, 1, time_budget=args.time_budget//2)
    for p in model.likelihood.parameters():
        p.requires_grad=True
    print(model.string_parameters_to_optimize())
    trainer.fit(args.iterations_free_noise, args.test_interval, 1, time_budget=args.time_budget//2)
    """

    # Figure
    axialPre = 50
    D1 = input_dim - calib_dim
    D2 = calib_dim
    lower1 = (-0*torch.ones(D1));upper1 = (1*torch.ones(D1))
    lower2 = torch.min(T,0)[0].data;upper2 = torch.max(T,0)[0].data
    gr = giveGrid(axialPre,1,lower2,upper2)
    res = torch.zeros(axialPre)
    theta_opt = calib_posterior.loc.data.detach().clone()
    print(calib_posterior.row_cov.parameter)
    for i in range(axialPre):
        t = gr[i]
        model.calib_posterior.loc.data = t.expand([1,1])
        res[i] = trainer.compute_loss_test()
    plt.plot(gr.numpy(),res.numpy())
    plt.plot([true_calib[0].numpy(),true_calib[0].numpy()],[res.min(),res.max()],color="red")
    plt.show()
    calib_posterior.loc.data = theta_opt
    print(theta_opt)

    tGrid, qTheta  = plotCalibDomain(X.data, X_star.data, T.data, Y.data, Z.data, model,lower2,upper2,calib_prior.loc.data,calib_prior.cov,true_calib,axialPre)
    #values, indices = qTheta.max(0)
    #theta_MAP = tGrid[indices,:]
    #thetaAnaly = true_calib.unsqueeze(0)

    #XX = giveGrid(100,1)
    #print(XX)
    #TT = giveGrid(100,1).detach().clone()

    #display(model,XX,YY=None,X=None,Y=None,i=0,file_path=None,format="pdf",sample_paths=True)
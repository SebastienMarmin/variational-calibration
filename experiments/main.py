import os, sys
from os.path import isfile, join
from os import listdir
sys.path.append(os.path.join(os.path.dirname(__file__), "../../vcal"))#sys.path.append(os.path.join(".", "../../"))#

import argparse
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

models = {"calib_model":AdditiveDiscrepancy}

def parse_args():
    available_models = models.keys()
    available_datasets = ["calib_borehole","calib_currin","calib_case1","calib_test_full","borehole"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/workspace/research/datasets.gitlab/export',
                        help='Dataset directory')
    parser.add_argument('--split_ratio_run', type=float, default=1,
                        help='Train/test split ratio for computer runs')
    parser.add_argument('--split_ratio_obs', type=float, default=1,
                        help='Train/test split ratio for real observations')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbosity of training steps')
    parser.add_argument('--nmc_train', type=int, default=1,
                        help='Number of Monte Carlo samples during training')
    parser.add_argument('--nmc_test', type=int, default=100,
                        help='Number of Monte Carlo samples during testing')
    parser.add_argument('--batch_size_obs', type=int, default=20,
                        help='Batch size during training for real observations')
    parser.add_argument('--batch_size_run', type=int, default=20,
                        help='Batch size during training for computer runs')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('--nfeatures_run', type=int, default=20,
                        help='Dimensionality of hidden layers for the computer model',)
    parser.add_argument('--nfeatures_obs', type=int, default=20,
                        help='Dimensionality of hidden layers for the discrepancy model',)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training', )
    parser.add_argument('--lr_calib', type=float, default=1e-1,
                        help='Learning rate for training', )
    parser.add_argument('--model', choices=available_models, type=str,
                        help='Type of Bayesian model')
    parser.add_argument('--outdir', type=str,
                        default='workspace/',
                        help='Output directory base path',)
    parser.add_argument('--seed', type=int, default=2018,
                        help='Random seed',)
    parser.add_argument('--noise_std_run', type=float, default=0.01,
                        help='Observation noise standard deviation')
    parser.add_argument('--noise_std_obs', type=float, default=0.01,
                        help='Computer run noise standard deviation')
    parser.add_argument('--iterations_fixed_noise', type=int, default=100,
                        help='Training iteration without noise optimization')
    parser.add_argument('--iterations_free_noise', type=int, default=100,
                        help='Training iteration with noise optimization')
    parser.add_argument('--test_interval', type=int, default=100,
                        help='Interval between testing')
    parser.add_argument('--time_budget', type=int, default=720,
                        help='Time budget in minutes')
    parser.add_argument('--cuda', action='store_true',
                        help='Training on gpu or cpu')
    parser.add_argument('--save_model', action='store_true',
                        help='Save resulting model')
    parser.add_argument('--full_cov_W', type=int,default=0,
                        help='Switch from fully factorized to full cov for q(W)')
    parser.add_argument('--init_batchsize', type=int,default=10000,
                        help='Maximum number of data points for the initialization')

    args = parser.parse_args()

    args.dataset_dir = os.path.abspath(args.dataset_dir)+'/'
    args.outdir = os.path.abspath(args.outdir)+'/'

    if args.cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    return args



def setup_dataset():# TODO common setup
    dataset_unidir = join(args.dataset_dir, args.dataset)
    onlyfiles = [f for f in listdir(dataset_unidir) if isfile(join(dataset_unidir, f))]
    tensor_names = ("X","Y","XStar","T","Z")
    extension = "csv"
    delimiter = ";"
    files = [X+"."+extension for X in tensor_names]
    if not set(files).issubset(onlyfiles):
        logger.error("Dataset in "+str(dataset_unidir)+" must contains "+str(set(files)))
        raise VcalException("Error while loading the dataset.")
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
        outputPlt.plot(t,qTheta.detach().numpy(),c="blue")
        outputPlt.plot(t,pTheta.detach().numpy(),c="black")
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

if __name__ == '__main__':
    args = parse_args()
    outdir = vcal.vardl_utils.next_path('%s/%s/%s/' % (args.outdir, args.dataset, args.model) + 'run-%04d/')
    try:
        os.makedirs(outdir)
    except OSError:  
        print ("Creation of the directory %s failed" % outdir)

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
    eta.variances   = torch.ones(1)
    delta.variances = .1*torch.ones(1)
    eta.lengthscales   = .3*torch.ones(1)
    delta.lengthscales = .03*torch.ones(1)
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
    model = AdditiveDiscrepancy(computer_model,discrepancy,calib_prior,calib_posterior,true_calib=true_calib)

    ### Initialization of the computer model
    # Compute how big can be the batch size
    npts_run = len(train_data_loader.loaders[1].dataset)
    init_batchsize_run = min(args.init_batchsize,npts_run)
    init_data_run,_=random_split(train_data_loader.loaders[1].dataset,[init_batchsize_run,npts_run-init_batchsize_run])
    dataloader_run_for_init=SingleSpaceBatchLoader(DataLoader(init_data_run,batch_size=init_batchsize_run),cat_inputs=True)
    computer_model_initializer=IBLMInitializer(computer_model,dataloader_run_for_init,noise_var =args.noise_std_run**2)
    computer_model_initializer.initialize()
    
    tb_logger = vcal.vardl_utils.logger.TensorboardLogger(path=outdir, model=model, directory=None)
    trainer = vcal.learning.Trainer(model, 'Adam', {'lr': args.lr}, train_data_loader,test_data_loader,args.device, args.seed, tb_logger, debug=args.verbose,lr_calib=args.lr_calib)

    logger.info(model.string_parameters_to_optimize())
    trainer.fit(args.iterations_free_noise, args.test_interval, 1, time_budget=args.time_budget//2)

    """
    eta.optimize(False)
    delta.optimize(False)    
    calib_posterior.cov.parameter.requires_grad = False
    for p in model.likelihood.parameters():
        p.requires_grad=False    
    logger.info(model.string_parameters_to_optimize())
    def perturbation(m):
        m.calib_posterior.loc.data = torch.ones(1,1)*torch.randn(1).item()*.5+.5
        return m
    trainer.multistart(perturbation,300,nstarts=2)

    eta.optimize(True)
    delta.optimize(True)    
    calib_posterior.optimize(True)    
    logger.info(model.string_parameters_to_optimize())
    #print("hel")
    #print(trainer.optimizer.lr)
    #trainer.optimizer.lr = 10e-16
    trainer.fit(args.iterations_fixed_noise, args.test_interval, 1, time_budget=args.time_budget//2)
    for p in model.likelihood.parameters():
        p.requires_grad=False
    logger.info(model.string_parameters_to_optimize())
    trainer.fit(args.iterations_free_noise, args.test_interval, 1, time_budget=args.time_budget//2)

    trainer.multistart(perturbation,300,nstarts=2)
    """

    # Figure
    
    X, Y = next(iter(test_obs_loader))
    X_star, T, Z = next(iter(test_run_loader))

    axialPre = 50
    D1 = input_dim - calib_dim
    D2 = calib_dim
    lower1 = (-0*torch.ones(D1));upper1 = (1*torch.ones(D1))
    lower2 = torch.min(T,0)[0].data;upper2 = torch.max(T,0)[0].data
    gr = giveGrid(axialPre,1,lower2,upper2)
    res = torch.zeros(axialPre)
    theta_opt = calib_posterior.loc.data.detach().clone()

    for i in range(axialPre):
        t = gr[i]
        model.calib_posterior.loc.data = t.expand([1,1])
        res[i] = trainer.compute_loss_test()
    plt.plot(gr.numpy(),res.numpy())
    plt.plot([true_calib[0].numpy(),true_calib[0].numpy()],[res.min(),res.max()],color="red")
    plt.show()
    calib_posterior.loc.data = theta_opt

    tGrid, qTheta  = plotCalibDomain(X.data, X_star.data, T.data, Y.data, Z.data, model,lower2,upper2,calib_prior.loc.data,calib_prior.cov,true_calib,axialPre)

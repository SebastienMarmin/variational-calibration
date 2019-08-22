import os, sys
from os.path import isfile, join
from os import listdir
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import subprocess # only for launching the script generating the figures

import argparse
import numpy as np
import torch
torch.set_num_threads(32) 
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
#from collections import OrderedDict

import timeit
#import intertools
import vcal
from vcal.nets import AdditiveDiscrepancy, RegressionNet, GeneralDiscrepancy
from vcal.layers import FourierFeaturesGaussianProcess as GP
from vcal.stats import GaussianVector
from vcal.utilities import MultiSpaceBatchLoader,SingleSpaceBatchLoader, gentxt, VcalException
from vcal.vardl_utils.initializers import IBLMInitializer

import json

import matplotlib
from matplotlib import pyplot as plt
import timeit

#models = {"additive":AdditiveDiscrepancy}

def parse_args():
    #available_models = models.keys()
    available_datasets = ["calib_borehole","calib_currin","calib_case1","calib_case2","calib_nevada","calib_test_full"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/workspace/Datasets',
                        help='Dataset directory')
    parser.add_argument('--split_ratio_run', type=float, default=1,
                        help='Train/test split ratio for computer runs')
    parser.add_argument('--split_ratio_obs', type=float, default=1,
                        help='Train/test split ratio for real observations')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbosity of training steps')
    parser.add_argument('--nmc_train', type=int, default=1,
                        help='Number of Monte Carlo samples during training')
    parser.add_argument('--nmc_test', type=int, default=100,
                        help='Number of Monte Carlo samples during testing')
    parser.add_argument('--batch_size_obs', type=int, default=20,
                        help='Batch size during training for real observations')
    parser.add_argument('--batch_size_run', type=int, default=20,
                        help='Batch size during training for computer runs')
    parser.add_argument('--nlayers_obs', type=int, default=1,
                        help='Number of GP layers for discrepancy')
    parser.add_argument('--nlayers_run', type=int, default=1,
                        help='Number of GP layers for computer model')
    parser.add_argument('--nfeatures_run', type=int, default=20,
                        help='Dimensionality of hidden layers for the computer model',)
    parser.add_argument('--additive', type=int, default=1,
                        help='Use additive or general discrepancy',)
    parser.add_argument('--nfeatures_obs', type=int, default=20,
                        help='Dimensionality of hidden layers for the discrepancy model',)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training', )
    parser.add_argument('--lr_calib', type=float, default=1e-1,
                        help='Learning rate for training of the variational calibration', )
    parser.add_argument('--model', type=str,
                        help='Type of Bayesian model')
    parser.add_argument('--outdir', type=str,
                        default='workspace/',
                        help='Output directory base path',)
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed',)
    parser.add_argument('--noise_std_run', type=float, default=0.01,
                        help='Observation noise standard deviation')
    parser.add_argument('--noise_std_obs', type=float, default=0.01,
                        help='Computer run noise standard deviation')
    parser.add_argument('--discrepancy_level', type=float, default=0.05,
                        help='From 0 to 1, how much the additive discrepancy should explain the signals\' stddev?')
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
    parser.add_argument('--rff_optim_run', type=int,default=0,
                        help='Optimize the Fourier features instead of lengthscales for comp. model')
    parser.add_argument('--rff_optim_obs', type=int,default=0,
                        help='Optimize the Fourier features instead of lengthscales for discrepancy')
    parser.add_argument('--init_batchsize', type=int,default=10000,
                        help='Maximum number of data points for the initialization')

    args = parser.parse_args()

    args.dataset_dir = os.path.abspath(args.dataset_dir)+'/'
    args.outdir = os.path.abspath(args.outdir)+'/'

    if args.cuda:
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            print("Cuda not available. Using cpu.")
            args.device = 'cpu'
    else:
        args.device = 'cpu'
    return args


def setup_dataset():
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
    qTheta = normalPDF(tGrid,model.calib_posterior.mean,model.calib_posterior.tril,isRoot=True)
    pTheta = normalPDF(tGrid,model.calib_prior.mean,model.calib_prior.tril,isRoot=True)
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

class DGP(torch.nn.Sequential):
    def __init__(self, input_dim,output_dim,full_cov_W,nlayers,nfeatures,nmc_train,nmc_test,mean,scale,**kwargs):
        gp_list = list() # type: List(torch.nn.Module)
        nl = nlayers
        for i in range(nlayers):
            # Layer widths given by trapezoidal interpolation
            d_in = int((1-i/nl)*input_dim + i/nl*output_dim)
            d_out = int((1-(i+1)/nl)*input_dim + (i+1)/nl*output_dim)
            gp   = GP(d_in,d_out,nfeatures=nfeatures, nmc_train=nmc_train, nmc_test=nmc_test,full_cov_W=full_cov_W)
            if i<nlayers-1:
                gp._stddevs.requires_grad = False
                # Scale factor useful only for the last layer 
                # (because else it is equivalent to the lengthscale of the next GP)
            gp_list += [gp]
        gp_list[-1].mean = mean # last GP fit the data output mean and variance (equivalent to standardize the data)
        gp_list[-1].stddevs = scale
        super(DGP, self).__init__(*gp_list, **kwargs)
        
    def optimize_weights(self,b=True):
        for gp in self:
            gp.optimize_weights(b)
    def fix_hyperparameters(self,b=True):
        for gp in self:
            print(gp)
            gp.fix_hyperparameters(b)
        
"""
def DGP(input_dim,output_dim,full_cov_W,nlayers,nfeatures,nmc_train,nmc_test,mean,scale):
    gp_list = list() # type: List(torch.nn.Module)
    nl = nlayers
    for i in range(nlayers):
        # Layer widths given by trapezoidal interpolation
        d_in = int((1-i/nl)*input_dim + i/nl*output_dim)
        d_out = int((1-(i+1)/nl)*input_dim + (i+1)/nl*output_dim)
        gp   = GP(d_in,d_out,nfeatures=nfeatures, nmc_train=nmc_train, nmc_test=nmc_test,full_cov_W=full_cov_W)
        if i<nlayers-1:
            gp._stddevs.requires_grad = False
            # Scale factor useful only for the last layer 
            # (because else it is equivalent to the lengthscale of the next GP)
        gp_list += [gp]
    gp_list[-1].mean = mean # last GP fit the data output mean and variance (equivalent to standardize the data)
    gp_list[-1].stddevs = scale
    return torch.nn.Sequential(*gp_list)
"""

if __name__ == '__main__':
    
    args = parse_args()
    proce = True
    if (args.dataset == "calib_nevada" or args.dataset == "calib_borehole") and args.additive==0:
        proce = False
    if (args.dataset == "calib_case2" or args.dataset == "calib_borehole") and args.nlayers_run==2:
        proce = False
    if proce:
        outdir = vcal.vardl_utils.next_path('%s/%s/%s/' % (args.outdir, args.dataset, "additive_"+str(args.additive)) + 'run-%04d/')
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

        output_mean_run = train_run_loader.dataset.tensors[-1].mean(-2)
        output_std_run  = train_run_loader.dataset.tensors[-1].mean(-2)
        scale_factor_run = (output_std_run**2).mean().sqrt().item()
        output_mean_obs = train_obs_loader.dataset.tensors[-1].var(-2).sqrt()
        output_std_obs  = train_obs_loader.dataset.tensors[-1].var(-2).sqrt()
        scale_factor_obs = (output_std_run**2).mean().sqrt().item()

        dobs = train_data_loader.loaders[0].dataset
        drun = train_data_loader.loaders[1].dataset
        logger.info("Training observation points: {:4d}, in dimension {:3d}.".format(len(dobs),dobs.tensors[0].size(-1)))
        logger.info("Training computer runs:      {:4d}, in dimension {:3d}.".format(len(drun),drun.tensors[0].size(-1)+drun.tensors[1].size(-1)))
        logger.info("Calibration dimension: {:3d}".format(drun.tensors[1].size(-1)))


        eta = DGP(input_dim,output_dim,args.full_cov_W,args.nlayers_run,args.nfeatures_run,args.nmc_train,args.nmc_test,output_mean_run,output_std_run)
        for gp in list(eta):
            gp.optimize_fourier_features(args.rff_optim_run==1)
        if args.additive == 1:
            dim_delta = input_dim-calib_dim
            scale_delta = args.discrepancy_level * output_std_obs
            mean_delta = torch.zeros(output_dim)
        else:
            dim_delta = 1+ input_dim-calib_dim
            scale_delta = output_std_obs
            mean_delta = output_mean_obs
        logger.info("Discrepancy input dimension: {:3d}".format(dim_delta))
        delta = DGP(dim_delta, output_dim,args.full_cov_W,args.nlayers_obs,args.nfeatures_obs,args.nmc_train,args.nmc_test,mean_delta,scale_delta)
        for gp in list(delta):
            gp.optimize_fourier_features(args.rff_optim_obs==1)

        computer_model = RegressionNet(eta)
        discrepancy   = RegressionNet(delta)

        computer_model.likelihood.stddevs = args.noise_std_run*scale_factor_run
        discrepancy.likelihood.stddevs    = args.noise_std_obs*scale_factor_obs


        calib_prior = GaussianVector(calib_dim,constant_mean=.5,parameter=False)
        calib_prior.stddevs=np.sqrt(calib_dim) # proportional with the length of the hypercube diagonal
        calib_posterior = GaussianVector(calib_dim)
        calib_posterior.loc.data = torch.ones_like(calib_posterior.loc)*.5
        calib_posterior.stddevs = np.sqrt(calib_dim)
        if args.additive == 1:
            model = AdditiveDiscrepancy(computer_model,discrepancy,calib_prior,calib_posterior,true_calib=true_calib)
        else:
            model = GeneralDiscrepancy(computer_model,discrepancy,calib_prior,calib_posterior,true_calib=true_calib)


        ### Initialization of the computer model
        # Compute how big can be the batch size
        npts_run = len(train_data_loader.loaders[1].dataset)
        init_batchsize_run = min(args.init_batchsize,npts_run)
        init_data_run,_=random_split(train_data_loader.loaders[1].dataset,[init_batchsize_run,npts_run-init_batchsize_run])
        dataloader_run_for_init=SingleSpaceBatchLoader(DataLoader(init_data_run,batch_size=init_batchsize_run),cat_inputs=True)
        computer_model_initializer=IBLMInitializer(computer_model,dataloader_run_for_init,noise_var =0.01*scale_factor_run)
        computer_model_initializer.initialize()

        tb_logger = vcal.vardl_utils.logger.TensorboardLogger(path=outdir, model=model, directory=None)


        ###################
        # STAGE 1
        ### Mute discrepancy and calibration
        discrepancy.likelihood.stddevs = 2*scale_factor_obs
        delta.optimize_weights(False)
        delta.fix_hyperparameters(True)
        calib_posterior.optimize(False)
        ### Start
        trainer = vcal.learning.Trainer(model, 'Adam', {'lr': args.lr}, train_data_loader,test_data_loader,args.device, args.seed, tb_logger, debug=args.verbose,lr_calib=args.lr_calib)
        trainer.fit(args.iterations_fixed_noise, args.test_interval, 1, time_budget=args.time_budget//2)

        ###################
        # STAGE 2
        ### Desactivate computer_model learning
        eta.fix_hyperparameters(True)
        eta.optimize_weights(False)
        ### Reactivate discrepancy and calibration
        discrepancy.likelihood.stddevs = args.noise_std_obs*scale_factor_obs
        delta.optimize_weights(True)
        calib_posterior.optimize(True)
        ### Activate observation noise optim
        model.discrepancy.likelihood.optimize(True)
        ### Low initial variational posterior variance for improvig the search of calib parameter mean
        calib_posterior.stddevs = 0.05*np.sqrt(calib_dim)
        ### Start
        logger.info("Stage 1 finished. Stage 2:")
        logger.info(model.string_parameters_to_optimize())
        trainer.fit(args.iterations_free_noise, args.test_interval, 1, time_budget=args.time_budget//2)

        ###################
        logger.info("Training finished.")

        logger.info("Start testing.")
        test_mnll, test_error = trainer.test()
        logger.info("Testing finished.")



        results = {}
        for key, value in vars(args).items():
            results[key] = value
        results['outdir'] = outdir
        results['trainable_parameters'] = model.trainable_parameters
        results['test_mnll'] = float(test_mnll.item())
        results.update(test_error)
        results['total_iters'] = trainer.current_iteration

        results["calib_mean"] = tuple((t.item() for t in calib_posterior.mean))
        results["calib_stddev"] =  tuple((t.item() for t in calib_posterior.stddevs))




        if args.dataset == "calib_nevada":        
            pre = 80
            xx = torch.cat((torch.linspace(0,1,pre).unsqueeze(-1),torch.zeros(pre,1)),-1)

            theta_hat = calib_posterior.mean.unsqueeze(-2).expand(xx.size(-2),-1)
            model.eval()
            Y_mean = model.phenomenon(xx.to(theta_hat.device),theta_hat)
            results["y_mean"] =  tuple((y.item() for y in Y_mean))



        with open(outdir + 'results.json', 'w') as fp:
            json.dump(results, fp, sort_keys=True, indent=4)
        logger.info("Results saved.")
        logger.info("\n---- RUN ["+outdir+"] FINISHED ----\n\n\n")

        flag_loc = "workspace/flag_new.txt"
        with open(flag_loc, "w") as f:
            f.write('1')
            f.write('\n'+outdir)
        print("Finished all experiments.")
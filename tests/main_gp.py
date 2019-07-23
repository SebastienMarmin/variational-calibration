#  Copyright (C) 2019   Sébastien Marmin <marmin@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>

import os, sys
from os.path import isfile, join
from os import listdir

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torch import ones_like
from collections import OrderedDict

import timeit

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))#sys.path.append(os.path.join(".", "../../"))#
import vcal
from vcal.layers import FourierFeaturesGaussianProcess
from vcal.utilities import SingleSpaceBatchLoader
#import humanize
import json

import matplotlib
from matplotlib import pyplot as plt
#import matplotlib2tikz 
import timeit

#from vardl.layers import BaseVariationalLayer
#from vardl.distributions import available_distributions, kl_divergence
#from vardl.distributions import FullyFactorizedMultivariateGaussian


#from scipy.stats import chi2

class FourierGaussianProcessNet(vcal.nets.RegressionNet):
    def __init__(self, input_dim, output_dim, nmc_train, nmc_test, nfeatures):
        """
        Parameters
        ----------
        input_dim: int
        output_dim: int
        nmc_train: int
        nmc_test: int
        nlayers: int
        nfeatures: int
        """
        super(FourierGaussianProcessNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.nfearures = nfeatures

        self.layers = torch.nn.ModuleList()
        self.layers.add_module('f', FourierFeaturesGaussianProcess(input_dim, output_dim,nfeatures=nfeatures,                                                                      nmc_train=nmc_train, nmc_test=nmc_test))
        self.name = 'FourierGP'
        self.train()

    def forward(self, input):
        x = input * torch.ones(list(self.layers)[0].nmc, *input.size()).to(input.device)

        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


models = {"fgp":FourierGaussianProcessNet}



def parse_args():

    available_models = models.keys()
    available_datasets = ['yacht', 'boston', 'concrete', 'energy', 'kin8nm', 'naval', 'powerplant', 'protein', "simple1","simple2"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/workspace/research/datasets.gitlab/export',
                        help='Dataset directory')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                        help='Train/test split ratio')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbosity of training steps')
    parser.add_argument('--nmc_train', type=int, default=1,
                        help='Number of Monte Carlo samples during training')
    parser.add_argument('--nmc_test', type=int, default=100,
                        help='Number of Monte Carlo samples during testing')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size during training')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('--nfeatures', type=int, default=16,
                        help='Dimensionality of hidden layers',)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for training', )
    parser.add_argument('--model', choices=available_models, type=str, required=True,
                        help='Type of Bayesian model')
    parser.add_argument('--outdir', type=str,
                        default='workspace/',
                        help='Output directory base path',)
    parser.add_argument('--seed', type=int, default=2018,
                        help='Random seed',)
    parser.add_argument('--noise_std', type=float, default=0.01,
                        help='Observation noise standard deviation')
    parser.add_argument('--iterations_fixed_noise', type=int, default=500000,
                        help='Training iteration without noise optimization')
    parser.add_argument('--iterations_free_noise', type=int, default=500000,
                        help='Training iteration with noise optimization')
    parser.add_argument('--test_interval', type=int, default=500,
                        help='Interval between testing')
    parser.add_argument('--time_budget', type=int, default=720,
                        help='Time budget in minutes')
    parser.add_argument('--cuda', action='store_true',
                        help='Training on gpu or cpu')
    parser.add_argument('--save_model', action='store_true',
                        help='Save resulting model')
    parser.add_argument('--full_cov_W', type=int,default=0,
                        help='Switch from fully factorized to full cov for q(W)')

    args = parser.parse_args()

    args.dataset_dir = os.path.abspath(args.dataset_dir)+'/'
    args.outdir = os.path.abspath(args.outdir)+'/'

    if args.cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    return args

def setup_dataset():
    dataset_unidir = join(args.dataset_dir, args.dataset,"pytorch")
    onlyfiles = [f for f in listdir(dataset_unidir) if isfile(join(dataset_unidir, f))]
    if len(onlyfiles)==1:
        dataset_path = join(dataset_unidir,args.dataset+'.pth')
        dataset = TensorDataset(*torch.load(dataset_path))
        input_dim = dataset.tensors[0].size(1)
        output_dim = dataset.tensors[1].size(1)
        size = len(dataset)
        train_size = int(args.split_ratio * size)
        test_size = size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    elif len(onlyfiles)==2:
        train_dataset_path = join(dataset_unidir,'train_' + args.dataset + '.pth')
        test_dataset_path = join(dataset_unidir,'test_' + args.dataset + '.pth')
        train_dataset = TensorDataset(*torch.load(train_dataset_path))
        test_dataset = TensorDataset(*torch.load(test_dataset_path))
        input_dim = train_dataset.tensors[0].size(1)
        output_dim = train_dataset.tensors[1].size(1)
    logger.info('Loading dataset from %s' % str(dataset_unidir))
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,# * torch.cuda.device_count(),
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)

    
    return train_dataloader, test_dataloader, input_dim, output_dim


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
            logger = vcal.vardl_utils.setup_logger('vcal', outdir)
    logger.info('Configuration:')
    for key, value in vars(args).items():
            logger.info('  %s = %s' % (key, value))

    # Save experiment configuration as yaml file in logdir
    with open(outdir + 'experiment_config.json', 'w') as fp:
            json.dump(vars(args), fp, sort_keys=True, indent=4)
    vcal.utilities.set_seed(args.seed)
    train_dl, test_dl, input_dim, output_dim = setup_dataset()

    train_dataloader = SingleSpaceBatchLoader(train_dl)
    test_dataloader  = SingleSpaceBatchLoader(test_dl)
    scale_factor = list(train_dataloader.datasets)[0].tensors[1].var(0).sqrt()
    model = models[args.model](input_dim, output_dim,nmc_train=args.nmc_train, nmc_test=args.nmc_test, nfeatures=args.nfeatures)
    model.likelihood.stddevs = scale_factor*args.noise_std
    # TODO more user friedndl
    gp = list(model.layers)[0]
    gp.prior_means.values.data = list(train_dataloader.datasets)[0].tensors[1].mean(0)
    gp.variances =  scale_factor**2
    gp.lengthscales = 0.05*ones_like(gp.lengthscales)
    #gp.full_cov_W = True
    
    gp.reset_parameters()
    gp.set_to_prior()
    gp.fix_hyperparameters()

    
    logger.info("Trainable parameters: %d" % model.trainable_parameters)
    
    tb_logger = vcal.vardl_utils.logger.TensorboardLogger(path=outdir, model=model, directory=None)

    trainer = vcal.learning.Trainer(model, 'Adam', {'lr': args.lr}, train_dataloader, test_dataloader,
                                                            args.device, args.seed, tb_logger, debug=False)
    model.likelihood.optimize(False)
    trainer.fit(args.iterations_fixed_noise, args.test_interval, 1, time_budget=args.time_budget//2)
    model.likelihood.scale.require_grad = True # TODO user friendly
    trainer.fit(args.iterations_free_noise, args.test_interval, 1, time_budget=args.time_budget//2)


    # Save results
    gp.local_reparam = False

    XX = list(test_dataloader.datasets)[0].tensors[0]
    YY = list(test_dataloader.datasets)[0].tensors[1]
    X = list(train_dataloader.datasets)[0].tensors[0]
    Y = list(train_dataloader.datasets)[0].tensors[1]
    plt.figure()
    display(model,XX,YY,X,Y)
    #plt.close()
    print(model.likelihood.stddevs)

    print(model.likelihood.variances)
    gp.set_to_posterior(X,Y.squeeze(-1),noise_covariance=model.likelihood.variances)
    display(model,XX,YY,X,Y,file_path="figure_analytic.pdf",sample_paths=False)

    logger.info("Testing and saving results...")
    test_mnll, test_error = trainer.test()

    results = {}
    logger.info('Starting benchmarking...')
    trainer.test_verbose = False
    t = timeit.Timer("trainer.test()", setup="from __main__ import trainer")
    times = np.array(t.repeat(10, 1)) * 1000.
    logger.info("Inference time on entire test set (90 percentile): %.4f ms" % (np.percentile(times, 0.90)))
    results['inference_times'] = times.tolist()
    logger.info('Benchmarking done')

    for key, value in vars(args).items():
        results[key] = value
    results['trainable_parameters'] = model.trainable_parameters
    results['test_mnll'] = float(test_mnll.item())
    results['test_error'] = float(test_error.item())
    results['total_iters'] = trainer.current_iteration
    with open(outdir + 'results.json', 'w') as fp:
        json.dump(results, fp, sort_keys=True, indent=4)

    import torch.nn as nn

    

 #python3 main.py --dataset simple1 --dataset_dir /home/sebastien/Datasets/ --nmc_train 11 --nmc_test 50 --batch_size 100 --lr 5e-2 --iterations_fixed_noise 100 --iterations_free_noise 100 --model fgp --test_interval 50 --nfeatures 13 --verbose
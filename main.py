from os.path import join
from torch import tensor, float32, double
from vcal.nets import RegressionNet
from vcal.layers import FourierFeaturesGaussianProcess as GP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from vcal.vardl_utils.logger import TensorboardLogger
from vcal.learning import TrainerNoCalib
from vcal.vardl_utils.initializers import IBLMInitializer
from vcal.utilities import SingleSpaceBatchLoader

TENSORBOARD_DIR = "tensorbord"




def VI_RFF_DGP(X, Y, XX, YY,seed, num_layers = 2, g = 1e-4):

    """
    Inputs
    ------
    X: training data inputs
    Y: training data output
    XX: testing data inputs
    YY: testing data output
    num_layers: number of layers of deep-gp
    g: nugget/noise parameter
    
    Outputs
    -------
    mu: predicted mean
    var: predicted variance
    
    """

    # Model params
    nfeatures=100
    
    # Training params
    lr = 0.01
    optimizer = 'Adam'
    batch_size = 30
    time_budget = 120
    iterations = 10000    
    nmc_train=10
    precision = float32 

    # Visualisation params
    test_interval = 10000
    train_log_interval = 10000
    nmc_test = 100

    # Start running
    d = X.shape[1]
    dout = 1
    device = "cpu" # or cuda
    # Prepare data
    x = tensor(X).type(precision)
    xx = tensor(XX).type(precision)
    mm = tensor(Y).type(precision).mean().item()
    y = tensor(Y).type(precision)-mm
    yy = tensor(YY).type(precision)-mm # of course not used until the end!!
    train_data = TensorDataset(x,y)
    test_data  = TensorDataset(xx,yy)


    train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)


    test_loader = DataLoader(test_data,
                                batch_size=batch_size,# * torch.cuda.device_count(),
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)

    
    layers = [GP(d, d, fix_output_stddev=True, nfeatures=nfeatures,nmc_train=nmc_train,nmc_test=nmc_test) for _ in range(num_layers-1)]
    layers += [GP(d, dout, nfeatures=nfeatures)]

    model = RegressionNet(layers=layers)


    model.likelihood.stddevs = g
    
    tb_logger = TensorboardLogger(path=join(TENSORBOARD_DIR,"n_"+str(x.shape[0]),"rep_"+str(seed)), model=model, directory=None)

    optimizer_config = {'lr': lr}
    trainer = TrainerNoCalib( model,
                    optimizer, optimizer_config, train_loader, test_loader, device,
                 seed, tb_logger)
    
    trainer.fit(iterations, test_interval, train_log_interval=train_log_interval, time_budget=time_budget)
    model.eval()
    
    samplePaths = mm + model.forward(xx)
    
    return samplePaths.mean(0).detach().numpy(), samplePaths.var(0).detach().numpy()

if __name__ == "__main__":
    

    import numpy as np
    import pandas as pd
    import properscoring as ps
    import time
    from os.path import join
    
    repetitions = 10
    for seed in range(1,repetitions+1):
        save_time = (seed == 1)

        name = "dgpRF"

        for n in [100, 500, 1000]:
            
            np.random.seed(seed)
                
            # Load pre-stored training and testing data
            dataPath = "/mnt/6AA0C6CAA0C69C47/Users/marmin/Downloads/gramacylab-deepgp-ex-1dd8a0b387fb/vecchia/schaffer/data"
            train_name = join(dataPath,"train_d2_n" + str(n) + "_seed" + str(seed) + ".csv")
            test_name  = join(dataPath,"test_d2_seed" + str(seed) + ".csv")
            train = pd.read_csv(train_name)
            test  = pd.read_csv(test_name)
            X = np.array(train.iloc[:, :-1])
            Y = np.array(train["Y"]).reshape(-1, 1)
            XX = np.array(test.iloc[:, :-1])
            YY = np.array(test["Y"]).reshape(-1, 1)
            
            if True:
                tic = time.time()
                mu, var = VI_RFF_DGP(X, Y, XX, YY,seed, num_layers = 2)
                toc = time.time()
                
                rmse = np.sqrt(np.mean((mu - YY)**2))
                crps = crps = np.mean(ps.crps_gaussian(YY, mu = mu, sig = np.sqrt(var)))

            if save_time:
                t = pd.read_csv("time.csv")
                t.loc[len(t.index)] = [name, n, (toc - tic) / 60]
                t.to_csv("time.csv", index = False)
                
            results = pd.read_csv("results.csv")
            results.loc[len(results.index)] = [name,  n, seed, rmse, crps]
            results.to_csv("results.csv", index = False)


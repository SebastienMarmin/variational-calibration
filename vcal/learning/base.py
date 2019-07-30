
# Original code by Simone Rossi
import torch
import time
import bunch
import shutil
import os
from torch.utils import hooks as hooks
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from typing import Union


from ..vardl_utils.logger import TensorboardLogger
from ..utilities import set_seed
from ..utilities import VcalTimeOutException, VcalNaNLossException
from ..nets import BaseNet

import logging
logger = logging.getLogger(__name__)
available_optimizers = {'Adam': optim.Adam,
                        'SGD': optim.SGD,
                        'Adagrad': optim.Adagrad}

error_names = ("error_Y","error_Z","error_calib")

class Trainer(object):
    def __init__(self, model, optimizer, optimizer_config, train_dataloader, test_dataloader, device,
                 seed, tb_logger, **kwargs):
        """

        Args:
            model:
            optimizer:
            optimizer_config:
            train_dataloader (DataLoader)(or list of) :
            test_dataloader (DataLoader) (or list of) :
            device:
            seed:
            tb_logger (TensorboardLogger):
            lr_decay_config:
            **kwargs:
        """
        set_seed(seed)
        self.device = device
        self.model = model.to(self.device, non_blocking=True)  # type: Union[nn.Module, BaseBayesianNet]

        logger.info(self.model.string_parameters_to_optimize())
        logger.info('Total: %s' % model.trainable_parameters)



        optimizable_params = set(filter(lambda p: p.requires_grad, model.parameters()))
        likelihood_params = set(model.likelihood.parameters())
        lr_likelihood = kwargs.pop('lr_likelihood') if 'lr_likelihood' in kwargs else  optimizer_config['lr']/5.
        try:
            calib_params = set(model.calib_posterior.parameters())
            model_params = optimizable_params - likelihood_params - calib_params
            lr_calib = kwargs.pop('lr_calib') if 'lr_calib' in kwargs else  optimizer_config['lr']
            self.optimizer = available_optimizers[optimizer]([
                {'params': list(model_params)},
                {'params': list(likelihood_params), 'lr': lr_likelihood},
                {'params': list(calib_params), 'lr': lr_calib}
            ], **optimizer_config)
        except AttributeError:
            model_params = optimizable_params - likelihood_params
            self.optimizer = available_optimizers[optimizer]([
                {'params': list(model_params)},
                {'params': list(likelihood_params), 'lr': lr_likelihood}
            ], **optimizer_config)


        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.current_epoch = 0
        self.current_iteration = 0

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.tb_logger = tb_logger
        self.optimizer_config = optimizer_config

        if 'lr_decay_config' in kwargs:
            self.lr_decay_config = bunch.Bunch(kwargs.pop('lr_decay_config'))
        else:
            self.lr_decay_config = bunch.Bunch()
            self.lr_decay_config.p = 0.
            self.lr_decay_config.gamma = 0.

        if 'kl_decay_config' in kwargs:
            self.kl_decay_config = bunch.Bunch(kwargs.pop('kl_decay_config'))
        else:
            self.kl_decay_config = bunch.Bunch()
            self.kl_decay_config.alpha = 0.
            self.kl_decay_config.beta = 0.
            self.kl_decay_config.gamma = 2.

        self.debug = kwargs.pop('debug') if 'debug' in kwargs else False
        self.train_verbose = kwargs.pop('train_verbose') if 'train_verbose' in kwargs else False
        self.test_verbose = kwargs.pop('test_verbose') if 'test_verbose' in kwargs else True
        self.save_checkpoints = kwargs.pop('save_checkpoints') if 'save_checkpoints' in kwargs else False

        

#    def compute_nell(self, Y_pred, Y_true, n, m) -> torch.Tensor:
#        nell = - n / m * torch.sum(torch.mean(self.model.likelihood.log_cond_prob(Y_true, Y_pred), 0))
#        return nell

    def compute_kl(self):
        return self.model.kl_divergence() * self.kl_decay_config.gamma/(1 + np.exp(self.kl_decay_config.alpha * (
            self.current_iteration - self.kl_decay_config.beta))) #4500 ok

    def compute_loss(self, Y_pred, Y_true, n_over_m):
        return self.model.compute_nell(Y_pred, Y_true, n_over_m) + self.compute_kl()

    def _train_batch(self, data, target, train_log_interval=1000):
        self.current_iteration += 1

        self.optimizer.zero_grad()
        n_over_m = self.train_dataloader.n_over_m
        if self.debug:
            with torch.autograd.detect_anomaly():
                output = self.model(data)
                loss = self.compute_loss(output, target, n_over_m)
                error = self.model.compute_error(output, target,n_over_m).sqrt()
                loss.backward()
        else:
            output = self.model(data)
            loss = self.compute_loss(output, target, n_over_m)
            error = self.model.compute_error(output, target,n_over_m)
            loss.backward()

        if torch.isnan(loss):
            logger.error('At step %d loss became NaN and it cannot be recovered. Run with debug flag to investigate '
                         'where it was produced' % self.current_iteration)
            raise VcalNaNLossException

        self.optimizer.step()
        if self.current_iteration % train_log_interval == 0 and self.train_verbose:
            logger.debug('Train || iter=%5d  loss=%01.03e  dkl=%01.03e  error=%.3f ' %
                                   (self.current_iteration, loss.item(), self.compute_kl(), error.item()))

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # self.tb_logger.writer.add_histogram(name,
                    # param.clone().cpu().data.numpy(),
                    # self.current_iteration, )
                    self.tb_logger.writer.add_histogram(name + '.grad', param.grad.clone().cpu().data.numpy(),
                                                        self.current_iteration, )

        self.tb_logger.scalar_summary('loss/train', loss, self.current_iteration)
        n_over_m = self.train_dataloader.n_over_m
        self.tb_logger.scalar_summary('loss/train/nll',self.model.compute_nell(output, target,n_over_m),                                      self.current_iteration)
        #self.tb_logger.scalar_summary('error/train', error, self.current_iteration)
        for i in range(len(error)):
            self.tb_logger.scalar_summary('error_'+str(i)+'/train', error[i], self.current_iteration)
        self.tb_logger.scalar_summary('model/dkl', self.compute_kl(), self.current_iteration)
        for name, param in self.model.likelihood.named_parameters():
            self.tb_logger.scalar_summary('model/'+name, param.item(), self.current_iteration)
        self._adjust_learning_rate()

    def _adjust_learning_rate(self):
        gamma = self.lr_decay_config.gamma
        p = self.lr_decay_config.p

        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = param_group['initial_lr'] * ((1 + gamma * self.current_iteration) ** -p)
            param_group['lr'] = new_lr
            self.tb_logger.scalar_summary('model/lr%d' % i, new_lr, self.current_iteration)

    def _train_per_iterations(self, iterations, train_log_interval=1000):
        """ Implement the logic of training the model. """
        self.model.train()

        #dataloader_iterator = iter(self.train_dataloader)
        #self.current_epoch += 1

        for i in range(iterations):
            try:
                data, target = next(self.dataloader_iterator)
            except StopIteration:
#                del dataloader_iterator
                self.dataloader_iterator = self.train_dataloader.iterable(cycle=True,out_device=self.device)
                data, target = next(self.dataloader_iterator)
                logger.info('Train >> Epoch %d completed' % self.current_epoch) if self.train_verbose else None
                self.current_epoch += 1

            self._train_batch(data, target, train_log_interval)

    def compute_loss_test(self):
        self.model.eval()
        data, target = next(iter(self.test_dataloader))
        n_over_m = self.train_dataloader.n_over_m
        with torch.no_grad():
            output = self.model(data)
            l = self.compute_loss(output, target, n_over_m)
        return l

    def save_checkpoint(self, is_best=False):
        state = {}
        state['current_epoch'] = self.current_epoch
        state['current_iteration'] = self.current_iteration
        state['state_dict'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        filename = 'checkpoint.pth.tar'
        torch.save(state, self.tb_logger.directory + filename)
#        torch.save(self.model, self.tb_logger.directory + 'model_inference.pth.tar')
        if is_best:
            shutil.copyfile(self.tb_logger.directory + filename,  self.tb_logger.directory + 'checkpoint_best.pth.tar')
#            torch.save(self.model, self.tb_logger.directory + 'model_inference_best.pth.tar')

    def resume_checkpoint(self, filepath):
        if os.path.isfile(filepath):
            logger.info("Loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath)
            self.current_epoch = checkpoint['current_epoch']
            self.current_iteration = checkpoint['current_iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Loaded checkpoint '{}' (epoch {})".format(filepath, self.current_epoch))
        else:
            logger.error("No checkpoint found at '{}'".format(filepath))

    def test(self):
        self.model.eval()
        test_nell = 0.
        test_error_sq = 0.
        one_over_n = self.test_dataloader.one_over_n
        n_over_m   = self.test_dataloader.n_over_m
        with torch.no_grad():
            for inputs, targets in self.test_dataloader.iterable(cycle = False,out_device=self.device):
                outputs = self.model(inputs)
                batch_nell = self.model.compute_nell(outputs, targets,one_over_n)
                test_nell += batch_nell
                test_error_sq += self.model.compute_error(outputs, targets,n_over_m)**2
        test_error = test_error_sq.sqrt()

        if self.test_verbose:
            stri = 'Test || iter=%5d   mnll=%01.03e' %(self.current_iteration, test_nell.item())
            for i in range(len(test_error)):
                stri += "   "+error_names[i]+("=%01.03e"%(test_error[i]))
            logger.info(stri)
        self.tb_logger.scalar_summary('loss/test', test_nell, self.current_iteration)
        for i in range(len(test_error)):
            self.tb_logger.scalar_summary(error_names[i]+'/test', test_error[i], self.current_iteration)
        test_error_dic = {}
        for key, value in zip(error_names,test_error):
            test_error_dic[key] = value.item()
        return test_nell, test_error_dic

    def multistart(self,perturbation,iterations,nstarts=1, train_log_interval=1000, time_budget=120):
        if iterations == 0:
            logger.warning('Zero training iteration. Skipping...')
            return
        self.dataloader_iterator = self.train_dataloader.iterable(cycle=True,out_device=self.device)
        
        param_list = list((p for p in self.model.parameters() if p.requires_grad))
        nparam = len(param_list)
        backup = [param.data.clone().detach() for param in param_list]
        best = [param.data.clone().detach() for param in param_list]
        best_test_nell, _ = self.test()
        t_start = time.time()        
        try:
            for _ in range(nstarts):
                self.model = perturbation(self.model) # make the starting point different
                self._train_per_iterations(iterations, train_log_interval)
                test_nell, _ = self.test()
                if test_nell < best_test_nell:
                    best_test_nell = test_nell.detach().clone()
                    for i in range(nparam):
                        best[i].data = param_list[i].detach().clone()
                for i in range(nparam):# return to starting point
                    param_list[i].data = backup[i].detach().clone()
                if (time.time() - t_start) / 60 > time_budget:
                    raise VcalTimeOutException('Interrupting training as time budget elapsed')
            for i in range(nparam):# return to best values
                param_list[i].data =  best[i].detach().clone()
            test_nell, _ = self.test()
            return test_nell

        except KeyboardInterrupt:
            logger.warning('Training interrupted by user. Saving current model snapshot')
            self.save_checkpoint(is_best=False) if self.save_checkpoints else None
            return self.test()

        except VcalTimeOutException as e:
            logger.warning(e)
            return self.test()



    def fit(self, iterations, test_interval, train_log_interval=1000, time_budget=120,nstarts=1):
        if iterations == 0:
            logger.warning('Zero training iteration. Skipping...')
            return

        self.dataloader_iterator = self.train_dataloader.iterable(cycle=True,out_device=self.device)

        if iterations < test_interval:
            iterations = test_interval
            logger.warning('Too few iteration. Setting iterations=%d' % iterations)

        best_test_nell, best_test_error = self.test()
        is_best = True

        t_start = time.time()

        try:
            for _ in range(iterations // test_interval):

                self._train_per_iterations(test_interval, train_log_interval)

                test_nell, test_error = self.test()
                if test_nell < best_test_nell and sum(test_error.values()) < sum(best_test_error.values())  and self.save_checkpoints: # TODO more elegant criteria
                    logger.info('Current snapshot (MNLL: %.3f - ERR: %.3f) better than previous.' % (test_nell,
                                                                                                     test_error))
                    is_best = True
                    best_test_error = test_error
                    best_test_nell = test_nell

                self.save_checkpoint(is_best) if self.save_checkpoints else None

                if (time.time() - t_start) / 60 > time_budget:
                    raise VcalTimeOutException('Interrupting training due to time budget elapsed')

            test_nell, test_error = self.test()
            self.save_checkpoint(is_best) if self.save_checkpoints else None
            return test_nell, test_error

        except KeyboardInterrupt:
            logger.warning('Training interrupted by user. Saving current model snapshot')
            self.save_checkpoint(is_best=False) if self.save_checkpoints else None
            return self.test()

        except VcalTimeOutException as e:
            logger.warning(e)
            return self.test()


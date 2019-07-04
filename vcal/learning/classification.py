#  Copyright (c) 2019
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  Authors:
#      Simone Rossi <simone.rossi@eurecom.fr>
#      Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
#

import torch
from torch.utils.data import DataLoader
import numpy as np

from . import BaseTrainer
from ..logger import TensorboardLogger
from ..utils import calibration_test
import logging
logger = logging.getLogger(__name__)


class TrainerClassifier(BaseTrainer):

    def __init__(self, model, optimizer, optimizer_config, train_dataloader, test_dataloader, device,
                 seed, tb_logger, **kwargs):
        """

        Args:
            model:
            optimizer:
            optimizer_config:
            train_dataloader (DataLoader):
            test_dataloader (DataLoader):
            device:
            seed:
            tb_logger (TensorboardLogger):
            lr_decay_config:
            **kwargs:
        """
        super(TrainerClassifier, self).__init__(model, optimizer, optimizer_config, train_dataloader,
                                                test_dataloader, device, seed, tb_logger, **kwargs)

    def compute_error(self, Y_pred, Y_true):

        likel_pred = self.model.likelihood.predict(Y_pred)
        mean_prediction = torch.mean(likel_pred, 0)
        prediction = torch.argmax(mean_prediction, 1)
        target = torch.argmax(Y_true, 1)
        correct = torch.sum(prediction.data == target.data)

        return 1. - correct.float() / (Y_true.size(0))

    def compute_calibration(self, bins=10):
        self.model.eval()
        predictions = []
        targets = []
        torch.autograd.set_grad_enabled(False)
        for inputs, targets_b in self.test_dataloader:
            outputs = self.model(inputs.to(self.device)).detach()
            predictions.append(self.model.likelihood.predict(outputs).mean(0))
            targets.append(targets_b.numpy())
        predictions = np.vstack(predictions)
        targets = np.argmax(np.vstack(targets), 1)
        ece, conf, accu, bins = calibration_test(predictions, targets, bins)
        logger.info('Calibration test completed. ECE = %.4f' % ece)
        return ece, conf, accu, bins

    # def _prior_update(self):
    #
    #     if self.current_iteration % self.prior_update_interval == 0:
    #         #self._logger.debug('Step %s - Updating priors ' % self.current_iteration)
    #         for child in self.model.modules():
    #             if isinstance(child, BayesianConv2d):
    #                 # For conv2d, weights have shape [out_channels, in_channels, kernel_size, kernel_size]
    #                 prior_means = child.prior_W.mean.view(child.out_channels, child.in_channels,
    #                                                             child.kernel_size,
    #                                                              child.kernel_size)
    #                 prior_vars = child.prior_W.logvars.view(child.out_channels, child.in_channels,
    #                                                                    child.kernel_size,
    #                                                                    child.kernel_size).exp()
    #
    #                 q_means = child.q_posterior_W.mean.view(child.out_channels, child.in_channels,
    #                                                              child.kernel_size,
    #                                                              child.kernel_size)
    #                 q_vars = child.q_posterior_W.logvars.view(child.out_channels, child.in_channels,
    #                                                                 child.kernel_size,
    #                                                                 child.kernel_size).exp()
    #
    #                 new_prior_means = torch.zeros_like(prior_means, device=prior_means.device)
    #                 new_prior_logvars = torch.zeros_like(prior_vars, device=prior_vars.device)
    #
    #                 if self.prior_update_conv2d_type == 'layer':
    #                     m = q_means.mean()
    #                     s = (q_vars + torch.pow((prior_means - q_means), 2)).mean()
    #                     new_prior_means.fill_(m)
    #                     new_prior_logvars.fill_(torch.log(s))
    #
    #                 if self.prior_update_conv2d_type == 'outchannels':
    #                     for c_out in range(child.out_channels):
    #                         m = q_means[c_out].mean()
    #                         s = (q_vars[c_out] + torch.pow((prior_means[c_out] - q_means[c_out]), 2)).mean()
    #
    #                         new_prior_means[c_out].fill_(m)
    #                         new_prior_logvars[c_out].fill_(torch.log(s))
    #
    #                 if self.prior_update_conv2d_type == 'outchannels+inchannels':
    #                     for c_out in range(child.out_channels):
    #                         for c_in in range(child.in_channels):
    #                             m = q_means[c_out, c_in].mean()
    #                             s = (q_vars[c_out, c_in] + torch.pow((prior_means[c_out, c_in] - q_means[c_out, c_in]), 2)).mean()
    #
    #                             new_prior_means[c_out, c_in].fill_(m)
    #                             new_prior_logvars[c_out, c_in].fill_(torch.log(s))
    #
    #                 if self.prior_update_conv2d_type == 'outchannels+inchannels+inrows':
    #                     for c_out in range(child.out_channels):
    #                         for c_in in range(child.in_channels):
    #                             for r_in in range(child.in_height):
    #                                 m = q_means[c_out, c_in, r_in].mean()
    #                                 s = (q_vars[c_out, c_in, r_in] + torch.pow((prior_means[c_out, c_in, r_in] - q_means[c_out, c_in, r_in]), 2)).mean()
    #
    #                                 new_prior_means[c_out, c_in, r_in].fill_(m)
    #                                 new_prior_logvars[c_out, c_in, r_in].fill_(torch.log(s))
    #
    #
    #                 child.prior_W._mean.data = new_prior_means.view_as(child.prior_W.mean)
    #                 child.prior_W._logvars.data = new_prior_logvars.view_as(child.prior_W.logvars)
    #
    #
    #             if isinstance(child, BayesianLinear):
    #                 new_prior_mean = child.q_posterior_W._mean.data.mean()
    #                 new_prior_var = (child.q_posterior_W._logvars.data.exp() + torch.pow((child.prior_W._mean.data -
    #                                                                                       child.q_posterior_W._mean.data),
    #                                                                                      2)).mean()
    #                 child.prior_W._mean.data.fill_(new_prior_mean)
    #                 child.prior_W._logvars.data.fill_(torch.log(new_prior_var))


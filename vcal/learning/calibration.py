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

from . import BaseTrainer

import logging
logger = logging.getLogger(__name__)


class TrainerRegressor(BaseTrainer):

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
        super(TrainerRegressor, self).__init__(model, optimizer, optimizer_config, train_dataloader,
                                                test_dataloader, device, seed, tb_logger, **kwargs)

    def compute_error(self, Y_pred: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(torch.pow((Y_true - Y_pred), 2))
                          )  # ok for regression

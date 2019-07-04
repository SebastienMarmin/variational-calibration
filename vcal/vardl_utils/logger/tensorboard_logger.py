# Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from tensorboardX import SummaryWriter

from . import BaseLogger
from vcal.vardl_utils import next_path
import bunch


class TensorboardLogger(BaseLogger):

    def __init__(self, directory: str, model, path=None):
        super(TensorboardLogger, self).__init__()

        if path is None:
            self.directory = next_path(directory + '/run-%04d/')
        else:
            self.directory = path
        self.writer = SummaryWriter('%s/' % self.directory)
        self.model = model
        self.scalars = bunch.Bunch()

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def __del__(self):
        self.writer.close()

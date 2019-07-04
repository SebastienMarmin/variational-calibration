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


import abc


class BaseLogger:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __int__(self, directory: str):
        self.directory = directory
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def scalar_summary(self, tag, value, step):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def image_summary(self, tag, images, step):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def histo_summary(self, tag, values, step, bins=1000):
        raise NotImplementedError("Subclass should implement this.")

    def save_model(self, extra_info=''):
        self.model.save_model(self.directory + '/model_snapshot'+extra_info+'.pth')
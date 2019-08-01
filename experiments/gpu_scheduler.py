# Copied from:

# Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
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
import subprocess # only for generating figures at the end

import queue
import os
import argparse
import yaml

from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import json

def setup_queue(q: queue.Queue, n_gpus: int, jobs_per_gpus: int):
    for _ in range(jobs_per_gpus):
        for i in range(n_gpus):
            q.put(i)


def worker(q: queue.Queue, command, **kwargs):
    gpu = q.get()
    gpu += 1
    command = 'CUDA_VISIBLE_DEVICES=%d ' % gpu + command
    for key, value in kwargs.items():
        command += ' --%s %s ' % (key, value)
    command += '--cuda'
    print(command)
    os.system(command)
    q.put(gpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="config file path",
                        default='hyperparameters.json')
    args = parser.parse_args()

    with open(args.config) as data_file:
        config = json.load(data_file)

    parallel_jobs = config['n_gpus'] * config['jobs_per_gpu']

    #if parallel_jobs > psutil.cpu_count():
    #    print('WARNING - More jobs than CPU thread. Might impact on time performances')

    gpu_queue = queue.Queue(maxsize=parallel_jobs)

    setup_queue(gpu_queue,
                n_gpus=config['n_gpus'],
                jobs_per_gpus=config['jobs_per_gpu'])

    base_command = config['bin'] + ' ' + config['executable']

    combinations = list(ParameterGrid(config['hyperparameters']))

    list_of_jobs = []
    for combination in combinations:
        list_of_jobs.append(delayed(worker)(gpu_queue,
                                            base_command,
                                            **combination))

    print(parallel_jobs)
    try:
        Parallel(n_jobs=parallel_jobs, backend="threading", verbose=1)(list_of_jobs)
    except KeyboardInterrupt:
        import warnings
        warnings.warn('User interruption. Cleaning everything..')



if __name__ == '__main__':
    main()

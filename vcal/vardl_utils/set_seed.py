# Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>

import random

import numpy as np
import torch
# import tensorflow as tf
import torch.backends.cudnn


# Log-sum operation


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    # tf.set_random_seed(seed)

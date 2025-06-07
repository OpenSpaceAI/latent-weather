import logging
import os
import random

import numpy
import torch


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def print_log(message):
    print(message)
    logging.info(message)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

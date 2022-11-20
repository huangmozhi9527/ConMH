import os
import logging
import random
import numpy as np
import torch

def set_seed(cfg):
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # set random seed for all gpus

def set_log(cfg, file_name):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(os.path.join(cfg.file_path, file_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
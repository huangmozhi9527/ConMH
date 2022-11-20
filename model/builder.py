import torch
from .conmh import conmh

def get_model(cfg):
    if cfg.model_name == 'conmh':
        model = conmh(cfg)

    return model
    
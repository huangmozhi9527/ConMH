from .conmh_dataset import get_conmh_train_loader, get_conmh_eval_loader

def get_train_data(cfg):
    if cfg.model_name == 'conmh':
        return get_conmh_train_loader(cfg, shuffle=True)

def get_eval_data(cfg):
    if cfg.model_name == 'conmh':
        return get_conmh_eval_loader(cfg)
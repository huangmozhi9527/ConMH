from .conmh_optim import conmh_opt_schedule

def get_opt_schedule(cfg, model):
    if cfg.model_name == 'conmh':
        return conmh_opt_schedule(cfg, model)

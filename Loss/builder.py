from .conmh_loss import conmh_criterion

def get_loss(cfg, data, model, epoch, i, total_len, logger):
    if cfg.model_name == 'conmh':
        return conmh_criterion(cfg, data, model, epoch, i, total_len, logger)

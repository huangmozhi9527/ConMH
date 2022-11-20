from .conmh_inference import conmh_inference

def get_inference(cfg, data, model):
    if cfg.model_name == 'conmh':
        return conmh_inference(cfg, data, model)
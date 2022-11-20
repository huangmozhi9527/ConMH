import torch.optim

def get_optim(cfg, model):
    if cfg.optimizer_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def get_schedule(cfg, optimizer):
    if cfg.schedule == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_rate, gamma = cfg.lr_decay_gamma)


class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, cfg, model):
        self.cfg = cfg
        self._optimizer = get_optim(cfg, model)
        self._schedule = get_schedule(cfg, self._optimizer)

    def _optimizer_step(self):
        self._optimizer.step()

    def _schedule_step(self):
        self._schedule.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def lr(self):
        return self._optimizer.param_groups[0]['lr']

    def set_lr(self, x):
        self._optimizer.param_groups[0]['lr'] = x

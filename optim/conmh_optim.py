import torch.optim
from .basic import ScheduledOptim


class conmh_opt_schedule(ScheduledOptim):
    def _schedule_step(self):
        self._schedule.step()
        if self.lr() < self.cfg.min_lr:
            self.set_lr(self.cfg.min_lr)
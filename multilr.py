import torch

class MultiLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lambda_factories, last_epoch=-1, verbose=False):
        self.schedulers = []
        for factory in lambda_factories:
            self.schedulers.append(factory(optimizer))
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        result = []
        for idx, sched in enumerate(self.schedulers):
            result.append(sched.get_lr()[idx])
        return result

    def _get_closed_form_lr(self):
        result = []
        for idx, sched in enumerate(self.schedulers):
            if hasattr(self, "_get_closed_form_lr"):
                values = sched._get_closed_form_lr()
            else:
                values = sched.get_lr()
            result.append(values[idx])
        return result

    def step(self, epoch=None):
        for sched in self.schedulers:
            sched.step(epoch)
        super().step(epoch)

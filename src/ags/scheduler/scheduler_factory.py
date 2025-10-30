import torch.optim.lr_scheduler as lr_scheduler

def build_scheduler_exponential(cfg, optimizer):
    return lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.gamma,
    )
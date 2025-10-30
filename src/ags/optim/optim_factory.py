import torch

def build_adam(cfg, params):
    return torch.optim.Adam(
        params,
        lr=float(cfg.lr),
        betas=tuple(float(i) for i in cfg.betas),
        eps=float(cfg.eps),
        weight_decay=float(cfg.weight_decay),
    )

def build_adamw(cfg, params):
    return torch.optim.AdamW(
        params,
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

def build_sgd(cfg, params):
    return torch.optim.SGD(
        params,
        lr=cfg.lr,
        momentum=getattr(cfg, "momentum", 0.9),
        weight_decay=cfg.weight_decay,
    )


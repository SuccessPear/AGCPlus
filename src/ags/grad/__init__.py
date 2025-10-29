from .agc import AGC
#from .clipnorm import ClipGradNorm
#from .ags import AGS

def build_grad_transform(cfg):
    name = cfg.name.lower()
    if name == "none":
        return None
    elif name == "agc":
        return AGC(cfg.clip_factor, cfg.eps, cfg.exclude_bias)
    # elif name == "clipnorm":
    #     return ClipGradNorm(cfg.max_norm, cfg.norm_type)
    # elif name == "ags":
    #     return AGS(cfg.scale_factor, cfg.beta, cfg.eps)
    else:
        raise ValueError(f"Unknown grad transform: {name}")

import torch

class AGC:
    """Adaptive Gradient Clipping (Brock et al. 2021)
       https://arxiv.org/abs/2102.06171
    """
    def __init__(self, cfg):
        self.clip_factor = cfg.clip_factor
        self.eps = float(cfg.eps)
        self.exclude_bias = cfg.exclude_bias

    @torch.no_grad()
    def __call__(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if self.exclude_bias and p.ndim <= 1:
                continue
            param_norm = torch.norm(p.detach())
            grad_norm = torch.norm(p.grad.detach())
            if param_norm.item() < self.eps or grad_norm.item() < self.eps:
                continue
            max_grad_norm = self.clip_factor * (param_norm + self.eps)
            clipped_grad = p.grad * (max_grad_norm / grad_norm).clamp(max=1.0)
            p.grad.detach().copy_(clipped_grad)

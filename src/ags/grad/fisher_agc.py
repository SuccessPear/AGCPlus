import torch


class FisherEMA:
    """
    Tracks diagonal Fisher Information via EMA of squared gradients.
    Stored per parameter tensor (keyed by parameter name).
    """

    def __init__(self, beta=0.99, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.fisher = {}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            if name not in self.fisher:
                self.fisher[name] = torch.zeros_like(p.grad)

            g = p.grad.detach()
            self.fisher[name].mul_(self.beta).addcmul_(
                g, g, value=1.0 - self.beta
            )

    def get_scalar(self, name):
        """
        Return scalar Fisher value for this parameter tensor.
        """
        F = self.fisher.get(name, None)
        if F is None:
            return None
        return F.mean()


class FisherAGC:
    """
    Fisher-Aware Adaptive Gradient Clipping (drop-in AGC replacement)

    Called exactly like AGC:
        gc(model)

    Internally:
        1. Update Fisher EMA
        2. Apply Fisher-scaled AGC clipping
    """

    def __init__(self, cfg):
        self.clip_factor = cfg.clip_factor
        self.eps = float(cfg.eps)
        self.exclude_bias = cfg.exclude_bias

        self.fisher_beta = float(getattr(cfg, "fisher_beta", 0.99))
        self.fisher_eps = float(getattr(cfg, "fisher_eps", 1e-8))

        self.fisher = FisherEMA(
            beta=self.fisher_beta,
            eps=self.fisher_eps,
        )

    @torch.no_grad()
    def __call__(self, model):
        # --------------------------------------------------
        # 1) Update Fisher Information (from current grads)
        # --------------------------------------------------
        self.fisher.update(model)

        # --------------------------------------------------
        # 2) Fisher-aware AGC clipping
        # --------------------------------------------------
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if self.exclude_bias and p.ndim <= 1:
                continue

            param_norm = torch.norm(p.detach())
            grad_norm = torch.norm(p.grad.detach())

            if param_norm.item() < self.eps or grad_norm.item() < self.eps:
                continue

            # Fisher scaling (layer-wise / tensor-wise)
            fisher_val = self.fisher.get_scalar(name)
            if fisher_val is None:
                fisher_scale = 1.0
            else:
                fisher_scale = 1.0 / torch.sqrt(fisher_val + self.fisher_eps)
            # option B
            if fisher_val < 1e-4 or fisher_val is None:
                continue

            max_grad_norm = (
                self.clip_factor * (param_norm + self.eps) * fisher_scale
            )

            clipped_grad = p.grad * (max_grad_norm / grad_norm).clamp(max=1.0)
            p.grad.detach().copy_(clipped_grad)

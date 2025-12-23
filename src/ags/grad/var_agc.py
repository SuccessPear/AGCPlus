import torch


class GradStatsEMA:
    """
    Tracks EMA of gradient mean and squared mean
    to estimate gradient variance.
    """

    def __init__(self, beta=0.99, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.mean = {}
        self.mean_sq = {}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            g = p.grad.detach()

            if name not in self.mean:
                self.mean[name] = torch.zeros_like(g)
                self.mean_sq[name] = torch.zeros_like(g)

            self.mean[name].mul_(self.beta).add_(
                g, alpha=1.0 - self.beta
            )
            self.mean_sq[name].mul_(self.beta).addcmul_(
                g, g, value=1.0 - self.beta
            )

    def get_variance(self, name):
        """
        Returns scalar variance estimate for a parameter tensor.
        """
        if name not in self.mean:
            return None
        var = self.mean_sq[name] - self.mean[name].pow(2)
        return var.mean().clamp(min=self.eps)


class VarAGC:
    """
    Variance-Aware Adaptive Gradient Clipping (Var-AGC)

    Drop-in replacement for AGC:
        gc(model)

    Clipping threshold:
        max_grad = clip_factor * ||param|| / sqrt(Var(g) + eps)
    """

    def __init__(self, cfg):
        self.clip_factor = cfg.clip_factor
        self.eps = float(cfg.eps)
        self.exclude_bias = cfg.exclude_bias

        self.var_beta = float(getattr(cfg, "var_beta", 0.99))
        self.var_eps = float(getattr(cfg, "var_eps", 1e-8))

        self.grad_stats = GradStatsEMA(
            beta=self.var_beta,
            eps=self.var_eps,
        )

    @torch.no_grad()
    def __call__(self, model):
        # --------------------------------------------------
        # 1) Update gradient statistics
        # --------------------------------------------------
        self.grad_stats.update(model)

        # --------------------------------------------------
        # 2) Variance-aware AGC clipping
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

            var = self.grad_stats.get_variance(name)
            if var is None:
                var_scale = 1.0
            else:
                var_scale = 1.0 / torch.sqrt(var + self.var_eps)

            max_grad_norm = (
                self.clip_factor * (param_norm + self.eps) * var_scale
            )

            clipped_grad = p.grad * (max_grad_norm / grad_norm).clamp(max=1.0)
            p.grad.detach().copy_(clipped_grad)

import torch


class CurvatureEMA:
    """
    Tracks a curvature proxy via EMA of gradient norms.
    Curvature ≈ EMA(||g||) / ||w||
    Stored per parameter tensor (keyed by parameter name).
    """

    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.ema_grad_norm = {}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            if name not in self.ema_grad_norm:
                self.ema_grad_norm[name] = torch.zeros(
                    1, device=p.device
                )

            g = p.grad.detach()
            gn = torch.norm(g)

            self.ema_grad_norm[name].mul_(self.beta).add_(
                gn * (1.0 - self.beta)
            )

    def get_scalar(self, name, param_norm):
        """
        Return scalar curvature proxy for this parameter tensor.
        curvature ≈ EMA(||g||) / ||w||
        """
        ema = self.ema_grad_norm.get(name, None)
        if ema is None:
            return None
        return ema / (param_norm + self.eps)


class CurvatureAGC:
    """
    Curvature-Aware Adaptive Gradient Clipping (AGC-C)

    Same call interface as FisherAGC:
        gc(model)

    Internally:
        1. Update curvature EMA
        2. Apply curvature-scaled AGC clipping
    """

    def __init__(self, cfg):
        self.clip_factor = cfg.clip_factor
        self.eps = float(cfg.eps)
        self.exclude_bias = cfg.exclude_bias

        self.curv_beta = float(getattr(cfg, "curv_beta", 0.9))
        self.curv_alpha = float(getattr(cfg, "curv_alpha", 1.0))

        self.curvature = CurvatureEMA(
            beta=self.curv_beta,
            eps=self.eps,
        )

    @torch.no_grad()
    def __call__(self, model):
        # --------------------------------------------------
        # 1) Update curvature proxy (from current grads)
        # --------------------------------------------------
        self.curvature.update(model)

        # --------------------------------------------------
        # 2) Curvature-aware AGC clipping
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

            curvature_val = self.curvature.get_scalar(name, param_norm)
            if curvature_val is None:
                continue

            # curvature-scaled AGC threshold
            max_grad_norm = (
                self.clip_factor
                * (param_norm + self.eps)
                / (1.0 + self.curv_alpha * curvature_val)
            )

            clipped_grad = p.grad * (max_grad_norm / grad_norm).clamp(max=1.0)
            p.grad.detach().copy_(clipped_grad)

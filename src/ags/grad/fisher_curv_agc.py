import torch


# ============================================================
# 1) Fisher EMA (same as your FisherAGC)
# ============================================================

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
        F = self.fisher.get(name, None)
        if F is None:
            return None
        return F.mean()


# ============================================================
# 2) Curvature EMA (AGC-C style)
# ============================================================

class CurvatureEMA:
    """
    Tracks curvature proxy via EMA of gradient norms.
    curvature ≈ EMA(||g||) / ||w||
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

            gn = torch.norm(p.grad.detach())
            self.ema_grad_norm[name].mul_(self.beta).add_(
                gn * (1.0 - self.beta)
            )

    def get_scalar(self, name, param_norm):
        ema = self.ema_grad_norm.get(name, None)
        if ema is None:
            return None
        return ema / (param_norm + self.eps)


# ============================================================
# 3) Fisher + Curvature AGC (HIERARCHICAL)
# ============================================================

class FisherCurvatureAGC:
    """
    Fisher-gated Curvature-Aware AGC

    Logic:
        - Fisher decides IF clipping is needed (hard gate)
        - Curvature-aware AGC decides HOW STRONG clipping is
    """

    def __init__(self, cfg):
        # AGC base
        self.clip_factor = cfg.clip_factor
        self.eps = float(cfg.eps)
        self.exclude_bias = cfg.exclude_bias

        # Fisher
        self.fisher_beta = float(getattr(cfg, "fisher_beta", 0.99))
        self.fisher_eps = float(getattr(cfg, "fisher_eps", 1e-8))
        self.fisher_threshold = float(getattr(cfg, "fisher_threshold", 1e-4))

        # Curvature
        self.curv_beta = float(getattr(cfg, "curv_beta", 0.9))
        self.curv_alpha = float(getattr(cfg, "curv_alpha", 1.0))

        self.fisher = FisherEMA(
            beta=self.fisher_beta,
            eps=self.fisher_eps,
        )
        self.curvature = CurvatureEMA(
            beta=self.curv_beta,
            eps=self.eps,
        )

    @torch.no_grad()
    def __call__(self, model):
        # --------------------------------------------------
        # 1) Update statistics (from raw gradients)
        # --------------------------------------------------
        self.fisher.update(model)
        self.curvature.update(model)

        # --------------------------------------------------
        # 2) Hierarchical clipping
        # --------------------------------------------------
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if self.exclude_bias and p.ndim <= 1:
                continue

            w = p.detach()
            g = p.grad.detach()

            wn = torch.norm(w)
            gn = torch.norm(g)

            if wn.item() < self.eps or gn.item() < self.eps:
                continue

            # ---------- Fisher gate ----------
            fisher_val = self.fisher.get_scalar(name)
            if fisher_val is None or fisher_val < self.fisher_threshold:
                continue   # safe region → no clipping

            # ---------- Curvature control ----------
            curvature_val = self.curvature.get_scalar(name, wn)
            if curvature_val is None:
                continue

            max_grad_norm = (
                self.clip_factor
                * (wn + self.eps)
                / (1.0 + self.curv_alpha * curvature_val)
            )

            scale = (max_grad_norm / gn).clamp(max=1.0)
            p.grad.mul_(scale)

import torch


class SafeDynamicAGC:
    """
    Safe Dynamic Adaptive Gradient Clipping (Safe-DAGC)

    Design principles:
    - Always apply standard AGC (fast spike suppression)
    - Only relax clipping when training is fully stable
    - Never weaken AGC during spikes
    """

    def __init__(self, cfg):
        # ===== Base AGC parameters =====
        self.clip_factor = float(cfg.clip_factor)
        self.eps = float(cfg.eps)
        self.exclude_bias = bool(cfg.exclude_bias)

        # ===== Dynamic control parameters =====
        self.severity_thresh = float(getattr(cfg, "severity_thresh", 0.3))
        self.tighten_lr = float(getattr(cfg, "tighten_lr", 5e-3))
        self.relax_lr = float(getattr(cfg, "relax_lr", 1e-3))

        self.min_clip = float(getattr(cfg, "min_clip", 1e-4))
        self.max_clip = float(getattr(cfg, "max_clip", 1.0))

    @torch.no_grad()
    def __call__(self, model):
        clipped_any = False
        severe_spike = False

        for p in model.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if self.exclude_bias and p.ndim <= 1:
                continue

            param_norm = torch.norm(p.detach())
            grad_norm = torch.norm(p.grad.detach())

            if param_norm < self.eps or grad_norm < self.eps:
                continue

            max_grad_norm = self.clip_factor * (param_norm + self.eps)

            if grad_norm > max_grad_norm:
                scale = max_grad_norm / (grad_norm + self.eps)
                p.grad.mul_(scale)

                clipped_any = True
                if scale < self.severity_thresh:
                    severe_spike = True

        # ===== Adaptation logic =====
        if severe_spike:
            # Emergency tightening (rare, but immediate)
            self.clip_factor *= (1.0 - self.tighten_lr)

        elif not clipped_any:
            # Fully stable step → relax very slowly
            self.clip_factor *= (1.0 + self.relax_lr)

        # Clamp to safe bounds
        self.clip_factor = float(
            max(self.min_clip, min(self.clip_factor, self.max_clip))
        )

import torch


class DynamicAGC:
    """
    Dynamic Adaptive Gradient Clipping (Dynamic-AGC)

    Self-adapts clip_factor to maintain a target clipping rate.
    """

    def __init__(self, cfg):
        # base AGC params
        self.clip_factor = float(cfg.clip_factor)
        self.eps = float(cfg.eps)
        self.exclude_bias = cfg.exclude_bias

        # dynamic control params
        self.target_clip_rate = float(getattr(cfg, "target_clip_rate", 0.05))
        self.adapt_lr = float(getattr(cfg, "adapt_lr", 0.01))
        self.ema_beta = float(getattr(cfg, "ema_beta", 0.95))

        self.min_clip = float(getattr(cfg, "min_clip", 1e-4))
        self.max_clip = float(getattr(cfg, "max_clip", 1.0))

        # state
        self.clip_rate_ema = 0.0

    @torch.no_grad()
    def __call__(self, model):
        total, clipped = 0, 0

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
                clipped += 1

            total += 1

        if total == 0:
            return

        # update EMA clipping rate
        clip_rate = clipped / total
        self.clip_rate_ema = (
            self.ema_beta * self.clip_rate_ema
            + (1 - self.ema_beta) * clip_rate
        )

        # adapt clip_factor (log-space for stability)
        error = self.clip_rate_ema - self.target_clip_rate
        self.clip_factor *= torch.exp(
            torch.tensor(self.adapt_lr * error)
        ).item()

        # clamp for safety
        self.clip_factor = float(
            max(self.min_clip, min(self.clip_factor, self.max_clip))
        )

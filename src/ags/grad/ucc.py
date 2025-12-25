import torch


class UpdateConsistencyControl:
    """
    Update Consistency Control (UCC)

    Stabilizes training by suppressing gradient updates whose direction
    is inconsistent with recent descent directions.

    Key idea:
      - Keep an EMA of past gradients (direction memory)
      - Measure cosine agreement between current gradient and memory
      - Damp or block updates when agreement is low or negative

    This targets direction / curvature noise rather than gradient magnitude.
    """

    def __init__(self, cfg):
        self.beta = float(getattr(cfg, "beta", 0.9))          # EMA for direction memory
        self.tau = float(getattr(cfg, "tau", 0.1))            # weak-agreement threshold
        self.eps = float(getattr(cfg, "eps", 1e-8))
        self.exclude_bias = bool(getattr(cfg, "exclude_bias", True))

        # Per-parameter EMA storage
        self.state = {}

    @torch.no_grad()
    def __call__(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if self.exclude_bias and p.ndim <= 1:
                continue

            g = p.grad

            # init EMA buffer
            if name not in self.state:
                self.state[name] = torch.zeros_like(g)

            m = self.state[name]

            # update direction memory
            m.mul_(self.beta).add_(g, alpha=1.0 - self.beta)

            g_norm = torch.norm(g)
            m_norm = torch.norm(m)

            if g_norm < self.eps or m_norm < self.eps:
                continue

            # cosine agreement
            cos = torch.dot(g.flatten(), m.flatten()) / (g_norm * m_norm + self.eps)

            # ===== Update control =====
            if cos < 0.0:
                # Actively harmful direction → block update
                g.zero_()

            elif cos < self.tau:
                # Weak agreement → damp update
                g.mul_(cos / self.tau)

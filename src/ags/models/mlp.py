import torch.nn as nn

def _num_classes(cfg):
    return getattr(cfg.model, "num_classes", getattr(cfg.dataset, "num_classes", 10))

def _get_list(v, n=None):
    """
    Cho phép truyền: một số (áp dụng cho mọi layer) hoặc list cụ thể.
    v: int | list[int]
    n: số layer mong muốn (để broadcast 1 giá trị thành list)
    """
    if isinstance(v, (list, tuple)):
        return list(v)
    if n is None:
        return [v]
    return [v for _ in range(n)]

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout=0.0, activation="relu", norm=None):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h)]
            if norm == "bn": layers += [nn.BatchNorm1d(h)]
            layers += [act()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):    # x: [B, in_dim]
        return self.net(x)

@MODELS.register("mlp")
def build_mlp(cfg):
    num_layers   = getattr(cfg, "num_layers", 3)
    hidden_size  = getattr(cfg, "hidden_size", 256)   # int
    hidden_list  = getattr(cfg, "hidden_list", None)  # list[int] optional
    dropout      = getattr(cfg, "dropout", 0.0)
    activation   = getattr(cfg, "activation", "relu")
    norm         = getattr(cfg, "norm", None)

    # đầu vào: nếu dùng ảnh (CIFAR 32x32x3) mà bạn flatten trước khi vào MLP
    # thì in_dim = 32*32*3; nếu dữ liệu vector, đặt thẳng in_dim trong YAML
    in_dim = getattr(cfg, "in_dim", None)
    if in_dim is None:
        img_sz = getattr(cfg.dataset, "img_size", 32)
        in_ch  = getattr(cfg.dataset, "in_channels", 3)
        in_dim = img_sz * img_sz * in_ch

    hidden_sizes = _get_list(hidden_list if hidden_list is not None else hidden_size, n=num_layers)
    return MLP(in_dim=in_dim,
               num_classes=_num_classes(cfg),
               hidden_sizes=hidden_sizes,
               dropout=dropout,
               activation=activation,
               norm=norm)



import torch.nn as nn
from ags.registry import MODELS

def _num_classes(cfg):
    return getattr(cfg.model, "num_classes", getattr(cfg.dataset, "num_classes", 10))

def _get_list(v, n=None):
    if isinstance(v, (list, tuple)): return list(v)
    if n is None: return [v]
    return [v for _ in range(n)]

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm="bn", activation="relu", pool=None, dropout=0.0):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=(norm is None))]
        if norm == "bn": layers += [nn.BatchNorm2d(out_ch)]
        layers += [act()]
        if pool == "max": layers += [nn.MaxPool2d(2)]
        if pool == "avg": layers += [nn.AvgPool2d(2)]
        if dropout > 0: layers += [nn.Dropout2d(dropout)]
        self.block = nn.Sequential(*layers)

        # init conv
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)

class FlexibleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, channels, k=3, activation="relu", norm="bn",
                 pool_every=1, pool_type="max", dropout=0.0, head="gap"):
        """
        channels: list[int], độ rộng từng block
        pool_every: sau mỗi N block thì áp dụng pooling 2x
        head: 'gap' (GlobalAvgPool) hoặc 'flatten'
        """
        super().__init__()
        blocks = []
        in_ch = in_channels
        for i, ch in enumerate(channels):
            pool = pool_type if ((i + 1) % pool_every == 0) else None
            blocks += [ConvBlock(in_ch, ch, k=k, p=k//2, norm=norm, activation=activation, pool=pool, dropout=dropout)]
            in_ch = ch
        self.features = nn.Sequential(*blocks)

        if head == "gap":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_ch, num_classes)
            )
        elif head == "flatten":
            # dùng khi bạn muốn giữ kích thước spatial (cần biết H,W sau conv)
            self.head = nn.Identity()
            self.classifier = nn.Linear(in_ch, num_classes)  # placeholder, cần tính in_features thủ công
        else:
            raise ValueError("head must be 'gap' or 'flatten'")

        # init fc
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

@MODELS.register("my_cnn")
def build_my_cnn(cfg):
    in_ch       = getattr(cfg.dataset, "in_channels", getattr(cfg.model, "in_channels", 3))
    num_layers  = getattr(cfg.model, "num_layers", 4)
    width       = getattr(cfg.model, "width", 64)        # int hoặc list
    width_list  = getattr(cfg.model, "width_list", None) # list[int] optional
    k           = getattr(cfg.model, "kernel_size", 3)
    activation  = getattr(cfg.model, "activation", "relu")
    norm        = getattr(cfg.model, "norm", "bn")       # 'bn' hoặc None
    pool_every  = getattr(cfg.model, "pool_every", 1)    # ví dụ 1: pool sau mỗi block
    pool_type   = getattr(cfg.model, "pool_type", "max")
    dropout     = getattr(cfg.model, "dropout", 0.0)
    head        = getattr(cfg.model, "head", "gap")

    channels = _get_list(width_list if width_list is not None else width, n=num_layers)
    return FlexibleCNN(in_channels=in_ch,
                       num_classes=_num_classes(cfg),
                       channels=channels,
                       k=k,
                       activation=activation,
                       norm=norm,
                       pool_every=pool_every,
                       pool_type=pool_type,
                       dropout=dropout,
                       head=head)

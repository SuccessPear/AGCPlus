import torch
import torch.nn as nn


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _num_classes(cfg):
    return cfg.get("num_classes")


# --------------------------------------------------
# Patch Embedding
# --------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)              # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


# --------------------------------------------------
# Transformer Block (Pre-LN, no BN)
# --------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------
# Vision Transformer
# --------------------------------------------------

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        B = x.size(0)

        x = self.patch_embed(x)             # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)      # [B, N+1, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                   # CLS token
        return self.head(cls_out)


# --------------------------------------------------
# Builder (same style as build_mlp)
# --------------------------------------------------

def build_vit(cfg):
    return VisionTransformer(
        img_size     = cfg.get("img_size", 32),
        patch_size   = cfg.get("patch_size", 4),
        in_channels  = cfg.get("in_channels", 3),
        num_classes  = _num_classes(cfg),
        embed_dim    = cfg.get("embed_dim", 256),
        depth        = cfg.get("depth", 6),
        num_heads    = cfg.get("num_heads", 8),
        mlp_ratio    = cfg.get("mlp_ratio", 4.0),
        dropout      = cfg.get("dropout", 0.0),
    )

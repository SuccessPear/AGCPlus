# scripts/debug_seqmnist_sanity.py
import os
import torch
import torch.nn as nn

# Adjust these imports to your project structure if needed
from src.ags.utils.config import load_config, compose_named_configs  # if you have it
from src.ags.engine.builders import load_dataloaders, load_model  # if you have it


def _ce_sanity_checks(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> None:
    assert logits.ndim == 2, f"logits must be (B,C), got {tuple(logits.shape)}"
    assert logits.size(1) == num_classes, f"C mismatch: logits C={logits.size(1)} vs num_classes={num_classes}"
    assert y.ndim == 1, f"targets must be (B,), got {tuple(y.shape)}"
    assert y.dtype == torch.long, f"targets dtype must be torch.long for CrossEntropyLoss, got {y.dtype}"
    y_min = int(y.min().item())
    y_max = int(y.max().item())
    assert 0 <= y_min and y_max < num_classes, f"targets out of range: min={y_min}, max={y_max}, C={num_classes}"


@torch.no_grad()
def _describe_batch(x: torch.Tensor, y: torch.Tensor) -> None:
    print("x:", tuple(x.shape), x.dtype, "min/max:", float(x.min()), float(x.max()))
    print("y:", tuple(y.shape), y.dtype, "min/max:", int(y.min()), int(y.max()))
    uniq = torch.bincount(y.cpu(), minlength=10)
    print("label bincount:", uniq.tolist())


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.manual_seed(0)

    cfg = compose_named_configs("configs/defaults.yaml")
    train_loader, _val_loader, _test_loader = load_dataloaders(cfg)

    model = load_model(cfg).cuda()
    model.train()

    num_classes = int(cfg.dataset.params.num_classes)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    x, y = next(iter(train_loader))
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)

    _describe_batch(x, y)

    # Check forward/criterion contract
    logits = model(x)
    if logits.ndim == 3:
        # If your model returns (B,T,C) by mistake, take last step
        logits = logits[:, -1, :].contiguous()

    _ce_sanity_checks(logits, y, num_classes)

    # Try to overfit a single batch; acc should quickly rise above 0.112
    for step in range(200):
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.ndim == 3:
            logits = logits[:, -1, :].contiguous()

        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        if step % 20 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f"step={step:03d} loss={loss.item():.4f} acc={acc:.3f}")

    print("done")


if __name__ == "__main__":
    main()
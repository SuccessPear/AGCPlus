# entrypoint/eval.py
import os
import torch
from torch.utils.data import DataLoader

from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.seed import seed_everything

from .train import instantiate, _import_from_dotted_path  # reuse helpers

@torch.no_grad()
def evaluate(config_path: str, ckpt_path: str = None):
    """
    Evaluation entrypoint.
    Expect config keys:
      model, loss, metric (optional), data.test_loader
      device, seed (optional)
    If ckpt_path is None, will try cfg['ckpt']['best'] then cfg['ckpt']['last'] if present.
    """
    cfg = load_config(config_path)
    logger = get_logger(__name__)
    seed_everything(int(cfg.get("seed", 42)))

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Build model and loss/metric
    model = instantiate(cfg["model"]).to(device)
    loss_fn = instantiate(cfg["loss"])
    metric_fn = instantiate(cfg.get("metric")) if cfg.get("metric") else None

    # Resolve checkpoint path
    if ckpt_path is None:
        ckpt_path = cfg.get("ckpt", {}).get("best", None) or cfg.get("ckpt", {}).get("last", None)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load weights (assume saved as {'model': state_dict, ...})
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    logger.info(f"Loaded weights from: {ckpt_path}")

    # Build test loader
    if "data" not in cfg or ("test_loader" not in cfg["data"]):
        raise ValueError("Config must provide data.test_loader for evaluation.")
    test_loader = instantiate(cfg["data"]["test_loader"])

    # Loop
    model.eval()
    n, total_loss, metric_sum, mcount = 0, 0.0, 0.0, 0

    for batch in test_loader:
        x, y = (t.to(device) for t in batch)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        n += 1
        if metric_fn is not None:
            mv = metric_fn(pred, y)
            if torch.is_tensor(mv):
                mv = mv.item()
            metric_sum += float(mv)
            mcount += 1

    results = {"test_loss": total_loss / max(1, n)}
    if metric_fn is not None:
        results["test_metric"] = metric_sum / max(1, mcount)

    # Pretty print
    msg = " | ".join(f"{k}: {v:.6f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in results.items())
    logger.info(f"[EVAL] {msg}")
    return results


# ------------- CLI -------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config.")
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (overrides cfg.ckpt).")
    args = p.parse_args()
    evaluate(args.config, args.ckpt)

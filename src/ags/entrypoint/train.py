# entrypoint/train.py
import os
import types
import torch
from torch.utils.data import DataLoader

from src.ags.trainer.loop import Trainer
from src.ags.utils.config import compose_named_configs
from src.ags.utils.logging import get_logger
from src.ags.utils.seed import seed_everything
from src.ags.engine.builders import *
from src.ags.metrics.classification import *
import torch.nn as nn
from collections import defaultdict
import mlflow
import math
import json
import numpy as np
# =========================
# Default validation epoch
# =========================
@torch.no_grad()
def default_val_fn(state):
    """
    Validation với nhiều metrics.
    - state["metric_fns"]: dict[str, callable], mỗi callable: fn(pred, y) -> float | tensor | dict[str, float]
    - Trả về: {"val_loss": ..., <metric_name>: ..., <submetric_name>: ...}
    """
    model     = state["model"]
    device    = state["device"]
    loss_fn   = state["loss_fn"]
    loader    = state.get("val_loader")
    metric_fns = state.get("metric_fns") or state.get("metric_fn")  # backward compat

    if loader is None:
        return {"val_loss": None}

    model.eval()
    total_loss, total_count = 0.0, 0

    # lưu tổng (sum) và trọng số (count) cho từng metric key
    metric_sum   = defaultdict(float)
    metric_count = defaultdict(int)

    for batch in loader:
        x, y = (t.to(device) for t in batch)
        pred = model(x)

        bs = x.shape[0] if hasattr(x, "shape") else 1
        loss = loss_fn(pred, y)
        total_loss  += float(loss.item()) * bs
        total_count += bs

        if metric_fns:
            # Cho phép: 1) dict tên->fn  2) callable đơn (legacy)
            if callable(metric_fns):
                metric_fns = {"val_metric": metric_fns}

            for name, fn in metric_fns.items():
                out = fn(pred, y)
                # tensor -> float
                if torch.is_tensor(out):
                    out = out.item()
                if isinstance(out, dict):
                    for k, v in out.items():
                        key = k if name in k else f"{name}:{k}"
                        if torch.is_tensor(v): v = v.item()
                        metric_sum[key]   += float(v) * bs
                        metric_count[key] += bs
                else:
                    metric_sum[name]   += float(out) * bs
                    metric_count[name] += bs

    results = {"val_loss": total_loss / max(1, total_count)}
    for k in metric_sum:
        results[k] = metric_sum[k] / max(1, metric_count[k])
    return results


def _flatten_dict(d, parent=""):
    out = {}
    for k, v in dict(d).items():
        kk = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


def _save_ckpt(path, model, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def train(config_path: str, gc = ""):
    cfg = compose_named_configs(config_path, gc=gc)
    logger = get_logger(__name__)
    seed_everything(int(cfg.get("seed", 42)))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # --- MLflow settings
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))
    mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment", "ags"))
    run_name = cfg.get("mlflow", {}).get(
        "run_name",
        str(cfg.get("model", {}).get("target", "model"))
    )

    with mlflow.start_run(run_name=run_name):
        # 1) Log toàn bộ config làm params + artifact
        try:
            mlflow.log_params(_flatten_dict(dict(cfg)))
        except Exception:
            # tránh crash nếu có value không serializable
            safe = {k: str(v) for k, v in _flatten_dict(dict(cfg)).items()}
            mlflow.log_params(safe)

        os.makedirs("artifacts", exist_ok=True)
        cfg_json_path = os.path.join("artifacts", "config.json")
        with open(cfg_json_path, "w", encoding="utf-8") as f:
            json.dump(dict(cfg), f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(cfg_json_path)

        # 2) Build data/model/opt
        train_loader, val_loader, test_loader = load_dataloaders(cfg)
        model = load_model(cfg).to(device)
        print(model)

        loss_fn = load_criterion(cfg)
        optimizer = load_optimizer(model, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("train", {}).get("amp", False)))
        scheduler = load_scheduler(optimizer, cfg)
        grad_control = load_grad_control(cfg)

        # 3) Metrics (acc@1/5 + f1_macro)
        if cfg.task == "classification":
            metric_fns = {
                "acc": lambda p, t: accuracy(p, t, topk=(1,), logits=True),
                #"f1":  lambda p, t: f1_score(p, t, average="macro", logits=True),
            }
        else:
            metric_fns = {}
        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, scaler=scaler, scheduler=scheduler, gc = grad_control, device=device, task=cfg.task)

        # 4) Best checkpoint logic (min nếu chứa 'loss', max otherwise)
        ckpt_cfg = cfg.get("ckpt", {}) or {}
        ckpt_dir = ckpt_cfg.get("dir", "runs/exp")
        best_name = ckpt_cfg.get("best", os.path.join(ckpt_dir, "best.pt"))
        last_name = ckpt_cfg.get("last", os.path.join(ckpt_dir, "last.pt"))
        save_best_on = ckpt_cfg.get("save_best_on", "val_loss")

        if "loss" in save_best_on.lower():
            best_score = math.inf
            better = lambda a, b: a < b
        else:
            best_score = -math.inf
            better = lambda a, b: a > b

        # 5) Wrap val_fn để vừa compute metric vừa log MLflow và lưu best
        if val_loader:
            def _val_and_log(state):
                res = default_val_fn({
                    **state,
                    "val_loader": val_loader,
                    "metric_fns": metric_fns
                })
                ep = state["epoch"]

                # log metrics mỗi epoch
                batch_losses = state.get("batch_losses", [])
                log_kv = {"epoch_loss_train": float(state.get("epoch_loss_train", 0.0)),
                          "batch_loss_mean": float(np.mean(batch_losses)),
                            "batch_loss_std":  float(np.std(batch_losses)),
                            "batch_loss_max":  float(np.max(batch_losses)),}
                for k, v in res.items():
                    if isinstance(v, (int, float)):
                        log_kv[k] = float(v)
                mlflow.log_metrics(log_kv, step=ep)

                # lưu best nếu tốt hơn
                score = res.get(save_best_on)
                if isinstance(score, (int, float)):
                    nonlocal best_score
                    if better(score, best_score):
                        best_score = score
                        _save_ckpt(best_name, model, extra={"epoch": ep, "score": score})
                        # log ngay artifact best (hoặc chỉ cuối run)
                        mlflow.log_artifact(best_name, artifact_path="checkpoints")

                return res
            val_callback = _val_and_log
        else:
            val_callback = None

        # 6) Train
        max_epoch = int(cfg.get("trainer", {}).get("epochs", 10))
        logger.info(f"Start training for {max_epoch} epochs on device={device}")
        trainer.fit(train_loader=train_loader, max_epoch=max_epoch, val_fn=val_callback)

        # 7) Save & log last checkpoint
        #_save_ckpt(last_name, model, extra={"epoch": max_epoch - 1})
        #mlflow.log_artifact(last_name, artifact_path="checkpoints")

        # 8) Log toàn bộ thư mục checkpoint (tuỳ thích)
        # if os.path.isdir(ckpt_dir):
        #     mlflow.log_artifacts(ckpt_dir, artifact_path="checkpoints")

        # 9) Log model
        # try:
        #     # MLflow >= 2.17 có log_state_dict
        #     mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="model_state")
        # except Exception:
        #     # fallback: log entire model (pickle)
        #     mlflow.pytorch.log_model(model, artifact_path="torch_model",
        #                   registered_model_name=cfg.get("mlflow", {}).get("register_as"))



# # ------------- CLI -------------
# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config.")
#     args = p.parse_args()
#     train(args.config)

# train("../../../configs/defaults.yaml")

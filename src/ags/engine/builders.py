# src/engine/builders.py
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

def _import_from_dotted_path(dotted: str):
    if ":" in dotted:
        mod, attr = dotted.split(":")
    else:
        parts = dotted.split(".")
        mod, attr = ".".join(parts[:-1]), parts[-1]
    return getattr(import_module(mod), attr)

def _instantiate(spec: Dict[str, Any], **extra_kwargs):
    if spec is None:
        return None
    if not isinstance(spec, dict) or "target" not in spec:
        raise ValueError(f"Invalid spec (need dict with 'target'): {spec}")
    params = spec.get("params", {})
    target = _import_from_dotted_path(spec["target"])
    return target(params, **extra_kwargs)

# ----------------- utilities -----------------

def _extract_meta_from_dataset(ds) -> Dict[str, Any]:
    """Best-effort rút meta từ dataset tiêu chuẩn (TorchVision, custom…)."""
    meta: Dict[str, Any] = {}

    # num_classes
    for key in ("num_classes",):
        if hasattr(ds, key):
            meta["num_classes"] = int(getattr(ds, key))
            break
    if "num_classes" not in meta and hasattr(ds, "classes"):
        try:
            meta["num_classes"] = len(ds.classes)
        except Exception:
            pass

    # channels / in_channels
    for key in ("in_channels", "channels", "c"):
        if hasattr(ds, key):
            meta["in_channels"] = int(getattr(ds, key))
            break

    # image size (H, W)
    # torchvision datasets đôi khi có .size hoặc transforms biết trước.
    for key in ("image_size", "img_size", "size"):
        if hasattr(ds, key):
            val = getattr(ds, key)
            if isinstance(val, (tuple, list)) and len(val) == 2:
                meta["img_size"] = (int(val[0]), int(val[1]))
            elif isinstance(val, int):
                meta["img_size"] = (int(val), int(val))
            break
    # optional seq_len for sequence datasets
    for key in ("seq_len", "sequence_length"):
        if hasattr(ds, key):
            meta["seq_len"] = int(getattr(ds, key))
            break

    return meta

def _peek_batch_meta(loader) -> Dict[str, Any]:
    """Nếu dataset không có meta, lấy nhanh từ 1 batch (an toàn & rẻ)."""
    meta: Dict[str, Any] = {}
    try:
        batch = next(iter(loader))
        # batch có dạng (x, y)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            # hình ảnh: (B, C, H, W)
            if hasattr(x, "shape") and x.ndim >= 3:
                # cố gắng đọc C,H,W
                if x.ndim == 4:
                    _, C, H, W = x.shape
                    meta.setdefault("in_channels", int(C))
                    meta.setdefault("img_size", (int(H), int(W)))
                elif x.ndim == 3:
                    C, H, W = x.shape
                    meta.setdefault("in_channels", int(C))
                    meta.setdefault("img_size", (int(H), int(W)))
            # ước lượng num_classes nếu y là long và min/max hợp lý
            try:
                import torch
                if hasattr(y, "dtype") and y.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                    y_min = int(y.min().item())
                    y_max = int(y.max().item())
                    if y_min >= 0:
                        meta.setdefault("num_classes", y_max + 1)
            except Exception:
                pass
    except Exception:
        pass
    return meta

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Cập nhật lồng nhau: chỉ set khóa còn thiếu/ghi đè khóa trùng khi cần."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

# ----------------- loaders for each component -----------------

def load_dataloaders(cfg):
    """
    Trả về: train_loader, val_loader, test_loader
    """
    data = cfg["dataset"]
    data["params"]["batch_size"] = cfg["batch_size"]
    data_module = _instantiate(data)
    data_module.setup()
    train_loader = data_module.train_loader()
    val_loader = data_module.val_loader()
    test_loader = data_module.test_loader()

    return train_loader, val_loader, test_loader

def load_model(cfg) -> Any:
    """
    Tự động bổ sung params cho model từ data_meta nếu thiếu.
    Ví dụ: num_classes, in_channels, img_size, seq_len.
    """
    spec = cfg["model"]
    params = {**spec.get("params", {})}
    # fill nếu thiếu
    for k in ("num_classes", "in_channels", "img_size", "seq_len"):
        if k not in params and k in cfg["dataset"]["params"]:
            params[k] = cfg["dataset"]["params"][k]
    spec = {"target": spec["target"], "params": params}
    return _instantiate(spec)


def load_metric(cfg) -> Optional[Any]:
    m = cfg.get("metric")
    return _instantiate(m) if m else None

def load_optimizer(model, cfg) -> Any:
    opt_spec = cfg["optimizer"]
    return _instantiate(opt_spec, params=model.parameters())

def load_scheduler(optimizer, cfg) -> Optional[Any]:
    sch = cfg.get("scheduler")
    return _instantiate(sch, optimizer=optimizer) if sch else None

def load_grad_control(cfg) -> Optional[Any]:
    gc = cfg.get("grad") or cfg.get("grad_control")
    return _instantiate(gc) if gc else None

def load_criterion(cfg) -> Optional[Any]:
    loss_fn = cfg.get("criterion") or cfg.get("loss_fn")
    return _instantiate(loss_fn) if loss_fn else None

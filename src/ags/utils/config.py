# src/utils/config.py
import io
import json
import os
from typing import Any, Mapping, MutableMapping

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


class DotDict(dict):
    """
    dict hỗ trợ truy cập bằng dấu chấm: cfg["a"]["b"] -> cfg.a.b
    Tự động đệ quy để bọc các dict con thành DotDict.
    """
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]

    @staticmethod
    def _wrap(value):
        if isinstance(value, dict):
            return DotDict({k: DotDict._wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [DotDict._wrap(v) for v in value]
        return value

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "DotDict":
        return cls({k: cls._wrap(v) for k, v in m.items()})

# ---- add to src/utils/config.py ----
import os


def _configs_root_from_exp(exp_path: str) -> str:
    """
    exp_path = '.../configs/experiment/xxx.yaml' -> trả về '.../configs'
    nếu không đúng pattern, dùng thư mục cha của exp_path.
    """
    base = os.path.abspath(os.path.dirname(exp_path))
    if os.path.basename(base) in ("experiment", "experiments"):
        return os.path.dirname(base)
    return base

def _load_named_yaml(root: str, subdir: str, name: str) -> DotDict:
    """Tìm <root>/<subdir>/<name>.yaml|yml rồi load -> DotDict."""
    for ext in (".yaml", ".yml"):
        p = os.path.join(root, subdir, f"{name}{ext}")
        if os.path.exists(p):
            return load_config(p)
    raise FileNotFoundError(
        f"Cannot find '{name}.yaml' under {root} in any of {subdir}"
    )

def compose_named_configs(exp_cfg_or_path, base_dir: str | None = None, gc = None) -> DotDict:
    """
    - exp_cfg_or_path: đường dẫn tới configs/experiment/<exp>.yaml hoặc mapping
    - base_dir: gốc thư mục 'configs/'. Nếu None, sẽ suy ra từ exp path.
    Trả về DotDict đã GHÉP ĐẦY ĐỦ.
    """
    exp_cfg = load_config(exp_cfg_or_path)

    # xác định root 'configs/'
    if isinstance(exp_cfg_or_path, (str, os.PathLike)):
        exp_path = os.fspath(exp_cfg_or_path)
        root = _configs_root_from_exp(exp_path) if base_dir is None else os.path.abspath(base_dir)
    else:
        if base_dir is None:
            raise ValueError("compose_named_configs(mapping): please provide base_dir='path/to/configs'")
        root = os.path.abspath(base_dir)

    for key, name in exp_cfg.items():
        if key == "grad":
            name = gc
        if isinstance(name, str):
            cfg = _load_named_yaml(root, key, name)
            exp_cfg[key] = cfg
    grad_name = exp_cfg.grad.params.name if exp_cfg.grad else "nongrad"
    schedule_name = exp_cfg.schedule.params.name if exp_cfg.schedule else "nonsche"
    optim_name = exp_cfg.optimizer.params.name if exp_cfg.optimizer else "nonopt"
    exp_cfg.mlflow.run_name = f"{exp_cfg.model.params.name}_{exp_cfg.dataset.params.name}_{grad_name}_{schedule_name}_{optim_name}"
    return exp_cfg


def _load_from_str(buf: str, ext_hint: str | None = None) -> MutableMapping[str, Any]:
    """
    Tải config từ chuỗi JSON/YAML.
    Ưu tiên theo ext_hint; nếu không có thì thử JSON trước, rồi YAML (nếu có PyYAML).
    """
    if ext_hint and ext_hint.lower() in (".json", "json"):
        return json.loads(buf)
    if ext_hint and ext_hint.lower() in (".yml", ".yaml", "yml", "yaml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML chưa được cài (pip install pyyaml).")
        return yaml.safe_load(buf)

    # Không có gợi ý -> thử JSON rồi YAML
    try:
        return json.loads(buf)
    except Exception:
        if not _HAS_YAML:
            raise
        return yaml.safe_load(buf)


def load_config(path: str | os.PathLike | Mapping[str, Any]) -> DotDict:
    """
    Tải cấu hình từ:
      - đường dẫn .json / .yml / .yaml
      - một mapping (dict) đã có sẵn
    Trả về DotDict để truy cập bằng dấu chấm.
    """
    if isinstance(path, Mapping):
        return DotDict.from_mapping(path)

    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    _, ext = os.path.splitext(path)
    with io.open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = DotDict.from_mapping(config)
        return config

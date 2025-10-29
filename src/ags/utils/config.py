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

# --- bổ sung vào src/utils/config.py ---

import os
from typing import Dict, Tuple

# Mặc định gốc tìm config con: thư mục chứa experiment.yaml, hoặc bạn có thể
# trỏ thẳng tới "configs/" cấp trên. Ta sẽ suy luận hợp lý (xem hàm bên dưới).

# ---- add to src/utils/config.py ----
import os
from typing import Dict, Tuple

# map khóa -> thư mục con (theo cấu trúc bạn đang dùng)
# _RESOLVE_TABLE: Dict[str, Tuple[str, ...]] = {
#     "model": ("model",),
#     "optimizer": ("optimizer",),
#     "loss": ("loss", "losses"),      # nếu về sau bạn thêm mục 'loss/'
#     "metric": ("metric", "metrics"), # tương tự
#     "grad": ("grad",),               # agc / clipnorm / none
#     "scheduler": ("scheduler",),
#     "trainer": ("trainer",),         # các mặc định cho trainer (max_epoch, amp, ...)
# }
#
# # dataset có thể là 1 file gom cả 3 loader
# _DATASET_DIRS: Tuple[str, ...] = ("dataset",)

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

def compose_named_configs(exp_cfg_or_path, base_dir: str | None = None) -> DotDict:
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

    # # 1) dataset: cho phép 2 kiểu
    # #    - exp.dataset là TÊN -> load configs/dataset/<name>.yaml
    # #      file này nên chứa data.train_loader / val_loader / test_loader
    # ds_name = exp_cfg.get("dataset", None)
    # if isinstance(ds_name, str):
    #     ds_cfg = _load_named_yaml(root, _DATASET_DIRS, ds_name)
    #     # ghép vào exp_cfg['data'] (tạo nếu chưa có)
    #     data_block = exp_cfg.get("data", {})
    #     # ưu tiên các khóa trong file dataset
    #     for k in ("train_loader", "val_loader", "test_loader"):
    #         if k in ds_cfg:
    #             data_block[k] = ds_cfg[k]
    #         elif "data" in ds_cfg and isinstance(ds_cfg["data"], dict) and k in ds_cfg["data"]:
    #             data_block[k] = ds_cfg["data"][k]
    #     exp_cfg["data"] = data_block
    #
    # # 2) resolve các thành phần còn lại theo tên (model/optimizer/grad/metric/loss/scheduler/trainer)
    # for key, subdirs in _RESOLVE_TABLE.items():
    #     val = exp_cfg.get(key, None)
    #     if isinstance(val, str):
    #         exp_cfg[key] = _load_named_yaml(root, subdirs, val)
    for key, name in exp_cfg.items():
        if isinstance(name, str):
            cfg = _load_named_yaml(root, key, name)
            exp_cfg[key] = cfg

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


print(compose_named_configs("../../../configs/defaults.yaml"))
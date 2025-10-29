# src/utils/seed.py
import os
import random
from typing import Optional

def seed_everything(seed: int = 42, deterministic: bool = False, warn_if_cuda_missing: bool = False) -> int:
    """
    Cố định seed cho python/random, numpy (nếu có), torch (nếu có).
    - deterministic=True: cố gắng bật chế độ thuật toán quyết định trong PyTorch.
    - Trả về seed để tiện log.

    Gợi ý: một số kernel có thể chậm hơn khi deterministic=True (cuDNN).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # numpy (tùy chọn)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    # torch (tùy chọn)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Thiết lập tính quyết định
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Fallback cho phiên bản cũ
                try:
                    import torch.backends.cudnn as cudnn  # type: ignore
                    cudnn.deterministic = True
                    cudnn.benchmark = False
                except Exception:
                    pass
        else:
            # Cho phép auto-tune nếu không cần deterministic
            try:
                import torch.backends.cudnn as cudnn  # type: ignore
                cudnn.deterministic = False
                cudnn.benchmark = True
            except Exception:
                pass
    except Exception:
        if warn_if_cuda_missing:
            print("[seed_everything] Torch not available; seeded Python/NumPy only.")

    return seed

def stratified_split_indices(labels, val_ratio=0.1):
    import numpy as np
    rng = np.random.default_rng()
    y = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx

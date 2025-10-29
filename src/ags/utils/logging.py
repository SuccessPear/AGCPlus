# src/utils/logging.py
import logging
import os
import sys
from typing import Optional

_LOGGING_INITIALIZED = False

def _use_rich_handler() -> bool:
    return os.environ.get("USE_RICH_LOG", "1") not in ("0", "false", "False")


def _setup_root_logging(level: int):
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    fmt_plain = "[%(asctime)s] %(levelname)s | %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if _use_rich_handler():
        try:
            from rich.logging import RichHandler  # type: ignore
            handler = RichHandler(
                markup=False,
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                log_time_format=datefmt,
            )
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
        except Exception:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(logging.Formatter(fmt_plain, datefmt=datefmt))
    else:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt_plain, datefmt=datefmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    _LOGGING_INITIALIZED = True


def get_logger(name: Optional[str] = None, level: int | None = None) -> logging.Logger:
    """
    Lấy logger theo tên, đảm bảo root đã cấu hình handler & level một lần.
    - level: nếu None, lấy từ ENV LOG_LEVEL (mặc định INFO).
    - Dùng RichHandler nếu có (tắt bằng USE_RICH_LOG=0).
    """
    if level is None:
        env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    _setup_root_logging(level)
    return logging.getLogger(name if name else __name__)

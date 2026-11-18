"""Keep generated caches local to the checkout unless configured externally."""

import os
from pathlib import Path
import sys


def configure_environment() -> None:
    cache_root = Path(__file__).resolve().parent / ".cache"
    for variable, directory in (
        ("PYTHONPYCACHEPREFIX", "python"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("TORCH_HOME", "torch"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        path = Path(os.environ.setdefault(variable, str(cache_root / directory)))
        path.mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

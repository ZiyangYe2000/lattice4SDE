"""Shared runtime metadata and device settings."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys

import torch

from .estimators import RuntimeOptions, resolve_device


def source_digest(entrypoint: str) -> str:
    core = {
        "__init__.py", "bridge.py", "config.py", "estimators.py", "models.py",
        "randomization.py", "runtime.py", "sobol.py", "storage.py", entrypoint,
    }
    digest = hashlib.sha256()
    for path in sorted(path for path in Path(__file__).parent.glob("*.py") if path.name in core):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def metadata(source_sha256: str) -> dict:
    return {
        "schema_version": 1, "target": "fixed_euler_expectation",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "source_sha256": source_sha256,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_threads": torch.get_num_threads(),
    }


def synchronize(options: RuntimeOptions) -> None:
    if options.device == "cuda":
        torch.cuda.synchronize()


def runtime_options(cfg: dict, device_override: str | None, *, reference: bool) -> RuntimeOptions:
    requested = device_override or cfg.get("device", "auto")
    dtype_name = cfg.get("reference_dtype" if reference else "dtype", "float64")
    runtime = cfg.get("runtime", {})
    return RuntimeOptions(
        device=resolve_device(requested), dtype=getattr(torch, dtype_name),
        memory_fraction=float(runtime.get("gpu_memory_fraction", 0.45)),
        max_chunk_points=int(runtime.get("max_chunk_points", 131072)),
        cpu_chunk_points=int(runtime.get("cpu_chunk_points", 4096)),
        max_rules_per_batch=int(runtime.get("max_rules_per_batch", 4)),
    )

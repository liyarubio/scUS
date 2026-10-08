from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


def _resolve(value: Any, base: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _resolve(v, base, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, base, key) for v in value]
    if isinstance(value, str) and (key.endswith(("path", "dir", "root")) or key in {"input", "output", "checkpoint", "vocab", "ortholog"}):
        p = Path(value).expanduser()
        return str((base / p).resolve() if not p.is_absolute() else p.resolve())
    return value


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open() as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg = _resolve(deepcopy(cfg), path.parent)
    cfg["_config_path"] = str(path)
    return cfg

def apply_overrides(cfg: dict, **values: Any) -> dict:
    cfg = deepcopy(cfg)
    for key, value in values.items():
        if value is not None:
            cfg[key] = value
    return cfg

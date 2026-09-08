from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from .artifacts import sha256


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def stage_directory(cfg: dict, stage: str, tag: str | None = None) -> Path:
    root = Path(cfg.get("output_dir", "outputs"))
    path = root / (tag or cfg.get("tag", "main")) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance(cfg: dict, stage: str, tag: str | None = None, extra: dict | None = None) -> Path:
    out = stage_directory(cfg, stage, tag)
    cache_path = Path(cfg.get("output_dir", "outputs")) / ".input_hashes.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    hashes = {}
    for name in ("input", "checkpoint", "vocab", "ortholog"):
        value = cfg.get(name) or cfg.get("data", {}).get(name) or cfg.get("model", {}).get(name)
        if value and Path(value).is_file():
            path = Path(value).resolve(); stat = path.stat(); key = str(path)
            signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if key not in cache or any(cache[key].get(k) != v for k, v in signature.items()):
                cache[key] = {**signature, "sha256": sha256(path)}
            hashes[name] = cache[key]["sha256"]
    write_json(cache_path, cache)
    record = {"stage": stage, "created": time.strftime("%FT%T%z"), "config": cfg, "input_hashes": hashes}
    if extra:
        record.update(extra)
    write_json(out / "run_config.json", record)
    return out


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1)


def robust_z(values):
    values = np.asarray(values, float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    return (values - median) / (1.4826 * mad if mad > 0 else 1.0)

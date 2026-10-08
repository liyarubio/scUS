from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen
import yaml


MANIFEST = Path(__file__).resolve().parent / "resources" / "artifacts.yaml"


def load_manifest(path: Path = MANIFEST) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)["artifacts"]


def sha256(path: str | Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(name: str, path: str | Path) -> bool:
    spec = load_manifest()[name]
    return sha256(path) == spec["sha256"]


def download(name: str, output_dir: str | Path, url: str | None = None) -> Path:
    spec = load_manifest()[name]
    url = url or os.environ.get("SCUS_ARTIFACT_URL") or spec.get("url")
    if not url:
        raise RuntimeError("No release URL is configured. Pass --url or set SCUS_ARTIFACT_URL.")
    output = Path(output_dir) / spec["filename"]
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    with urlopen(url) as response, partial.open("wb") as handle:
        while chunk := response.read(8 << 20):
            handle.write(chunk)
    if sha256(partial) != spec["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {name}")
    partial.replace(output)
    return output

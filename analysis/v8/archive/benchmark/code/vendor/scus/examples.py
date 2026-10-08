from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .align import fit, project
from .data.prepare import prepare_data
from .runtime import robust_z, stage_directory, write_json
from .zero_shot import encode


def navigo_analyze(cfg: dict, tag: str | None = None) -> Path:
    raw = pd.read_parquet(stage_directory(cfg, "zero-shot", tag) / "cell_scores.parquet")
    aligned_path = stage_directory(cfg, "align-project", tag) / "cell_scores.parquet"
    aligned = pd.read_parquet(aligned_path) if aligned_path.exists() else None
    raw["developmental_day"] = pd.to_numeric(raw[cfg.get("example", {}).get("day_column", "day")].astype(str).str.replace(r"^[Ee]", "", regex=True), errors="coerce")
    raw["zero_shot_z"] = robust_z(raw.cell_score)
    raw["align_z"] = robust_z(aligned.cell_score) if aligned is not None else np.nan
    trajectory = cfg.get("example", {}).get("trajectory_column", "major_trajectory")
    result = raw.groupby([trajectory, "developmental_day"], observed=True).agg(zero_shot=("zero_shot_z", "median"), aligned=("align_z", "median"), cells=("cell_index", "size")).reset_index()
    out = stage_directory(cfg, "navigo-analysis", tag); result.to_csv(out / "developmental_profiles.csv", index=False)
    write_json(out / "status.json", {"status": "complete", "trajectories": int(result[trajectory].nunique()), "semantics": "Navigo Mu/Ms moments; not raw U/S counts or RNA velocity"})
    return out


def navigo_visualize(cfg: dict, tag: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    source = stage_directory(cfg, "navigo-analysis", tag) / "developmental_profiles.csv"
    data = pd.read_csv(source); trajectory = cfg.get("example", {}).get("trajectory_column", "major_trajectory")
    out = stage_directory(cfg, "navigo-figures", tag)
    matrix = data.pivot(index=trajectory, columns="developmental_day", values="zero_shot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    whole = data.groupby("developmental_day", observed=True).apply(
        lambda frame: np.average(frame.zero_shot, weights=frame.cells)
    )
    axes[0].plot(whole.index, whole.values, color="#315C9B", lw=2); axes[0].set(xlabel="developmental day", ylabel="zero-shot robust z", title="Whole embryo")
    sns.heatmap(matrix, cmap="vlag", center=0, robust=True, ax=axes[1], cbar_kws={"label": "zero-shot distance z"}); axes[1].set(title="Major trajectories", xlabel="developmental day")
    fig.suptitle("Navigo Mu/Ms — scUS zero-shot profiles"); fig.tight_layout(); fig.savefig(out / "developmental_profiles.png", dpi=200); fig.savefig(out / "developmental_profiles.pdf"); plt.close(fig)
    write_json(out / "status.json", {"status": "complete"}); return out


def run_navigo(action: str, cfg: dict, tag: str | None = None, device: str | None = None):
    if action == "prepare": return prepare_data(cfg, tag)
    if action == "zero-shot": return encode(cfg, tag, device_name=device)
    if action == "align": fit(cfg, tag, device); return project(cfg, tag, device_name=device)
    if action == "analyze": return navigo_analyze(cfg, tag)
    if action == "visualize": return navigo_visualize(cfg, tag)
    raise ValueError(action)

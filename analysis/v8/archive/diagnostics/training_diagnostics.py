from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "stage", "protocol", "seed", "epoch", "global_step", "split",
    "loss_total", "loss_u", "loss_s", "loss_adv", "loss_disc",
    "loss_relational", "loss_variance", "loss_covariance", "loss_identity",
    "disc_accuracy", "learning_rate", "adv_weight", "adapter_alpha",
    "effective_rank", "distance_median", "distance_iqr", "knn_overlap",
    "geometry_u_spearman", "geometry_s_spearman",
    "raw_projected_distance_spearman", "matched_shuffle_effect", "coverage",
    "checkpoint_eligible", "checkpoint_selection_score",
]


def _empty_standard(rows: int) -> pd.DataFrame:
    frame = pd.DataFrame(index=np.arange(rows))
    for column in STANDARD_COLUMNS:
        frame[column] = np.nan
    return frame


def _copy_first(target: pd.DataFrame, source: pd.DataFrame, output: str, names: Iterable[str]) -> None:
    for name in names:
        if name in source:
            target[output] = source[name].to_numpy()
            return


def _read_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "step" not in frame:
        raise ValueError(f"Metrics file has no step column: {path}")
    return frame


def canonical_pretraining_metrics(log_root: str | Path, resume_step: int = 191_380, seed: int = 618) -> pd.DataFrame:
    """Merge the canonical pretraining run across its Lightning resume boundary.

    Historical versions 0/1 are incomplete exploratory runs. Version 3 only
    records the resume learning rate. Version 2 contributes steps before the
    boundary and version 4 contributes the resumed run from the boundary.
    """
    root = Path(log_root)
    sources = [("version_2", lambda value: value < resume_step), ("version_4", lambda value: value >= resume_step)]
    pieces = []
    for version, predicate in sources:
        path = root / version / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = _read_metrics(path)
        step = pd.to_numeric(frame["step"], errors="coerce")
        frame = frame.loc[step.notna() & predicate(step)].copy()
        frame["global_step"] = pd.to_numeric(frame["step"], errors="coerce").astype("int64")
        frame["source_version"] = version
        pieces.append(frame)
    raw = pd.concat(pieces, ignore_index=True).sort_values(["global_step", "source_version"], kind="stable")

    rows: list[pd.DataFrame] = []
    definitions = [
        ("train_step", "train_loss_step", "train_loss_u_step", "train_loss_s_step"),
        ("val_step", "val_loss_step", "val_loss_u_step", "val_loss_s_step"),
        ("train", "train_loss_epoch", "train_loss_u_epoch", "train_loss_s_epoch"),
        ("val", "val_loss_epoch", "val_loss_u_epoch", "val_loss_s_epoch"),
    ]
    for split, total, u_name, s_name in definitions:
        present = [name for name in (total, u_name, s_name) if name in raw]
        if not present:
            continue
        subset = raw.loc[raw[present].notna().any(axis=1)].copy()
        if subset.empty:
            continue
        out = _empty_standard(len(subset))
        out["stage"] = "pretraining"
        out["protocol"] = "canonical_resume"
        out["seed"] = seed
        out["epoch"] = pd.to_numeric(subset.get("epoch"), errors="coerce").to_numpy()
        out["global_step"] = subset["global_step"].to_numpy()
        out["split"] = split
        if total in subset:
            out["loss_total"] = pd.to_numeric(subset[total], errors="coerce").to_numpy()
        if u_name in subset:
            out["loss_u"] = pd.to_numeric(subset[u_name], errors="coerce").to_numpy()
        if s_name in subset:
            out["loss_s"] = pd.to_numeric(subset[s_name], errors="coerce").to_numpy()
        if "lr-Adam" in subset:
            out["learning_rate"] = pd.to_numeric(subset["lr-Adam"], errors="coerce").to_numpy()
        out["source_version"] = subset["source_version"].to_numpy()
        rows.append(out)
    canonical = pd.concat(rows, ignore_index=True)
    canonical = canonical.sort_values(["global_step", "split"], kind="stable")
    canonical = canonical.drop_duplicates(["global_step", "split"], keep="last").reset_index(drop=True)
    return canonical


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"epoch=(\d+)", path.name)
    return int(match.group(1)) if match else None


def checkpoint_evaluation_from_logs(
    checkpoint_dir: str | Path | None,
    canonical: pd.DataFrame,
    selected_epoch: int = 11,
) -> pd.DataFrame:
    """Create an auditable checkpoint table without fabricating unlogged metrics."""
    epoch_rows = canonical.loc[canonical["split"].isin(["train", "val"])].copy()
    pivot = epoch_rows.pivot_table(index="epoch", columns="split", values=["loss_total", "loss_u", "loss_s"], aggfunc="last")
    epochs: set[int] = set()
    paths: dict[int, Path] = {}
    if checkpoint_dir is not None:
        for path in Path(checkpoint_dir).glob("*.ckpt"):
            epoch = _checkpoint_epoch(path)
            if epoch is not None:
                epochs.add(epoch)
                paths[epoch] = path
    epochs.update(int(value) for value in epoch_rows["epoch"].dropna().unique())
    records = []
    for epoch in sorted(epochs):
        record = {"epoch": epoch, "checkpoint": str(paths.get(epoch, "")), "selected": epoch == selected_epoch}
        for metric in ("loss_total", "loss_u", "loss_s"):
            for split in ("train", "val"):
                key = (metric, split)
                record[f"{split}_{metric}"] = float(pivot.loc[epoch, key]) if epoch in pivot.index and key in pivot and pd.notna(pivot.loc[epoch, key]) else np.nan
        value = record.get("val_loss_total", np.nan)
        record["perplexity"] = math.exp(value) if np.isfinite(value) else np.nan
        record.update({
            "masked_accuracy": np.nan,
            "macro_f1": np.nan,
            "expected_calibration_error": np.nan,
            "u_to_s_surprisal": np.nan,
            "s_to_u_surprisal": np.nan,
            "true_matched_shuffle_effect": np.nan,
            "evaluation_status": "not_evaluated_missing_fixed_validation_tokens",
        })
        records.append(record)
    return pd.DataFrame(records)


def canonical_align_metrics(path: str | Path, protocol: str, seed: int = 42) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    out = _empty_standard(len(source))
    out["stage"] = "align"
    out["protocol"] = protocol
    out["seed"] = seed
    _copy_first(out, source, "epoch", ["epoch"])
    _copy_first(out, source, "global_step", ["global_step", "step"])
    _copy_first(out, source, "split", ["split"])
    if out["split"].isna().all():
        out["split"] = "val"
    mappings = {
        "loss_total": ["loss_total"],
        "loss_adv": ["loss_adv", "val_confusion", "validation_confusion"],
        "loss_disc": ["loss_disc"],
        "loss_relational": ["loss_relational"],
        "loss_variance": ["loss_variance"],
        "loss_covariance": ["loss_covariance"],
        "loss_identity": ["loss_identity"],
        "disc_accuracy": ["disc_acc", "disc_accuracy"],
        "adv_weight": ["adv_lambda", "adv_weight"],
        "adapter_alpha": ["projector_alpha", "alpha"],
        "effective_rank": ["effective_rank", "qc_effective_rank"],
        "distance_median": ["distance_median", "qc_projected_distance_median", "val_matched_distance", "validation_matched_distance"],
        "distance_iqr": ["distance_iqr", "qc_distance_iqr"],
        "knn_overlap": ["knn_overlap", "qc_knn_overlap"],
        "geometry_u_spearman": ["geometry_u_spearman", "qc_within_u_geometry_spearman"],
        "geometry_s_spearman": ["geometry_s_spearman", "qc_within_s_geometry_spearman"],
        "raw_projected_distance_spearman": ["raw_projected_distance_spearman", "qc_raw_projected_distance_spearman"],
        "matched_shuffle_effect": ["matched_shuffle_effect", "qc_matched_shuffle_effect"],
        "coverage": ["coverage", "qc_coverage"],
        # `qc_eligible` in historical adversarial-only runs used fewer gates
        # than the current anti-collapse contract. Never silently promote it
        # to current checkpoint eligibility.
        "checkpoint_eligible": ["checkpoint_eligible"],
        "checkpoint_selection_score": ["checkpoint_selection_score", "qc_selection_score"],
    }
    for output, inputs in mappings.items():
        _copy_first(out, source, output, inputs)
    for extra in [
        "qc_probe_auc", "qc_probe_balanced_accuracy", "qc_mean_within_geometry_spearman",
        "qc_projected_raw_distance_std_ratio", "qc_raw_distance_median", "qc_finite_qc",
        "qc_eligible", "qc_selection_score",
    ]:
        out[extra] = source[extra].to_numpy() if extra in source else np.nan
    out["source_file"] = str(Path(path).resolve())
    return out


def _rolling(series: pd.Series, window: int = 101) -> pd.Series:
    return series.rolling(window=min(window, max(3, len(series) // 20)), center=True, min_periods=1).median()


def reconstructed_validation_steps(log_root: str | Path) -> pd.DataFrame:
    """Recover validation-batch curves and map them into training-step space.

    Historical Lightning rows omit `epoch` for validation steps. Each logger
    version contains an integer and constant number of validation batches per
    logged epoch, so batches are assigned sequentially and interpolated between
    adjacent epoch-end training steps. The returned mapping is explicitly
    marked as reconstructed rather than treated as an original global step.
    """
    records = []
    previous_epoch_end = -1
    for version in ("version_2", "version_4"):
        source = _read_metrics(Path(log_root) / version / "metrics.csv")
        epoch_rows = source.loc[source.get("val_loss_epoch").notna(), ["epoch", "step"]].copy()
        epoch_rows = epoch_rows.dropna().sort_values("epoch")
        batch_rows = source.loc[source.get("val_loss_step").notna()].sort_values("step").copy()
        if epoch_rows.empty or batch_rows.empty:
            continue
        if len(batch_rows) % len(epoch_rows):
            raise ValueError(f"{version}: validation batches cannot be evenly assigned to logged epochs")
        batches_per_epoch = len(batch_rows) // len(epoch_rows)
        for offset, epoch_row in enumerate(epoch_rows.itertuples(index=False)):
            chunk = batch_rows.iloc[offset * batches_per_epoch:(offset + 1) * batches_per_epoch]
            epoch_end = int(epoch_row.step)
            epoch_start = previous_epoch_end + 1
            within = np.arange(1, batches_per_epoch + 1)
            mapped = epoch_start + within / (batches_per_epoch + 1) * (epoch_end - epoch_start)
            for position, (_, row) in enumerate(chunk.iterrows()):
                records.append({
                    "epoch": int(epoch_row.epoch), "validation_batch_in_epoch": position,
                    "mapped_global_step": float(mapped[position]), "source_version": version,
                    "loss_total": row.get("val_loss_step", np.nan),
                    "loss_u": row.get("val_loss_u_step", np.nan),
                    "loss_s": row.get("val_loss_s_step", np.nan),
                    "mapping_status": "reconstructed_from_982_batches_and_epoch_end_steps",
                })
            previous_epoch_end = epoch_end
    return pd.DataFrame(records)


def _axes_grid(rows: int, columns: int, figsize=(14, 9)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(rows, columns, figsize=figsize, constrained_layout=True)
    return plt, figure, np.asarray(axes).reshape(-1)


def _save(plt, figure, output: Path) -> None:
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _not_logged(axis, label="not logged") -> None:
    axis.text(0.5, 0.5, label, transform=axis.transAxes, ha="center", va="center", color="0.45", fontsize=11)
    axis.set_xticks([]); axis.set_yticks([])


def plot_pretraining(canonical: pd.DataFrame, checkpoints: pd.DataFrame, output: Path, resume_step: int, selected_epoch: int) -> None:
    plt, figure, axes = _axes_grid(2, 3, (16, 9))
    colors = {"loss_total": "#202020", "loss_u": "#3B82F6", "loss_s": "#EF4444"}
    step = canonical.loc[canonical["split"] == "train_step"].sort_values("global_step")
    for metric, color in colors.items():
        valid = step.loc[step[metric].notna()]
        if not valid.empty:
            axes[0].plot(valid["global_step"], _rolling(valid[metric]), label=metric.replace("loss_", ""), color=color, lw=1.4)
    axes[0].axvline(resume_step, ls="--", color="0.5", lw=1, label="resume")
    axes[0].set(title="A  Step-level training loss", xlabel="global step", ylabel="cross-entropy"); axes[0].legend(frameon=False)

    epoch = canonical.loc[canonical["split"].isin(["train", "val"])].copy()
    for split, color in [("train", "#2563EB"), ("val", "#DC2626")]:
        part = epoch.loc[(epoch["split"] == split) & epoch["loss_total"].notna()].sort_values("epoch")
        axes[1].plot(part["epoch"], part["loss_total"], marker="o", label=split, color=color)
    axes[1].axvline(selected_epoch, ls="--", color="#7C3AED", label=f"selected epoch {selected_epoch}")
    axes[1].set(title="B  Epoch loss and generalization", xlabel="epoch", ylabel="cross-entropy"); axes[1].legend(frameon=False)

    val = epoch.loc[epoch["split"] == "val"].sort_values("epoch")
    axes[2].plot(val["epoch"], val["loss_u"], marker="o", color=colors["loss_u"], label="U")
    axes[2].plot(val["epoch"], val["loss_s"], marker="o", color=colors["loss_s"], label="S")
    axes[2].axvline(selected_epoch, ls="--", color="#7C3AED")
    axes[2].set(title="C  Validation modality losses", xlabel="epoch", ylabel="cross-entropy"); axes[2].legend(frameon=False)

    train_epoch = epoch.loc[epoch["split"] == "train", ["epoch", "loss_total"]].dropna().drop_duplicates("epoch", keep="last")
    val_epoch = epoch.loc[epoch["split"] == "val", ["epoch", "loss_total"]].dropna().drop_duplicates("epoch", keep="last")
    gap = val_epoch.merge(train_epoch, on="epoch", suffixes=("_val", "_train"))
    if gap.empty:
        _not_logged(axes[3])
    else:
        axes[3].axhline(0, color="0.5", lw=1)
        axes[3].plot(gap["epoch"], gap["loss_total_val"] - gap["loss_total_train"], marker="o", color="#D97706")
    axes[3].set(title="D  Generalization gap", xlabel="epoch", ylabel="validation - train")

    if not checkpoints.empty:
        axes[4].plot(checkpoints["epoch"], checkpoints["perplexity"], marker="o", color="#059669", label="perplexity")
        right = axes[4].twinx()
        if checkpoints["masked_accuracy"].notna().any():
            right.plot(checkpoints["epoch"], checkpoints["masked_accuracy"], color="#7C3AED", marker="s", label="accuracy")
            right.set_ylabel("masked-bin accuracy")
        else:
            right.text(0.98, 0.95, "accuracy: fixed-token re-evaluation pending", transform=right.transAxes, ha="right", va="top", color="0.45", fontsize=8)
            right.set_yticks([])
        axes[4].set(title="E  Validation prediction performance", xlabel="epoch", ylabel="perplexity")
    else:
        _not_logged(axes[4], "checkpoint evaluation unavailable")

    if not checkpoints.empty:
        y = checkpoints["val_loss_total"]
        axes[5].scatter(checkpoints["epoch"], y, c=np.where(checkpoints["selected"], "#7C3AED", "#94A3B8"), s=np.where(checkpoints["selected"], 80, 32))
        selected = checkpoints.loc[checkpoints["selected"] & y.notna()]
        if not selected.empty:
            axes[5].annotate("selected", (selected.iloc[0]["epoch"], selected.iloc[0]["val_loss_total"]), xytext=(8, 8), textcoords="offset points")
    axes[5].set(title="F  Checkpoint selection", xlabel="epoch", ylabel="logged validation loss")
    figure.suptitle("scUS pretraining diagnostics (seed 618; frozen historical run)", fontsize=15)
    _save(plt, figure, output)


def plot_pretraining_step_loss_styled(
    canonical: pd.DataFrame,
    output: str | Path,
    resume_step: int | None = None,
    center_window: int = 301,
    band_window: int = 801,
    validation_steps: pd.DataFrame | None = None,
) -> Path:
    """Render a publication-style standalone version of pretraining panel 1A.

    Lines are centered rolling medians. Shaded regions are centered rolling
    25th--75th percentiles of the observed step-level loss, not confidence
    intervals or replicate uncertainty.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    step = canonical.loc[canonical["split"].eq("train_step")].sort_values("global_step")
    if step.empty:
        raise ValueError("No train_step rows are available for pretraining panel 1A")

    def valid_window(requested: int, size: int) -> int:
        window = min(requested, size if size % 2 else size - 1)
        return max(3, window)

    palette = {
        "loss_total": ("Total", "#172033"),
        "loss_u": ("Unspliced (U)", "#2F6BFF"),
        "loss_s": ("Spliced (S)", "#E64B5D"),
    }
    plot_rows = []
    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#FBFCFE")
    for metric in ("loss_u", "loss_s"):
        label, color = palette[metric]
        valid = step.loc[step[metric].notna(), ["global_step", "epoch", metric]].copy()
        if valid.empty:
            continue
        center_size = valid_window(center_window, len(valid))
        band_size = valid_window(band_window, len(valid))
        values = valid[metric]
        center = values.rolling(center_size, center=True, min_periods=1).median()
        lower = values.rolling(band_size, center=True, min_periods=max(3, band_size // 5)).quantile(.25)
        upper = values.rolling(band_size, center=True, min_periods=max(3, band_size // 5)).quantile(.75)
        lower, upper = lower.fillna(center), upper.fillna(center)
        x = valid["global_step"].to_numpy(dtype=float)
        axis.fill_between(x, lower.to_numpy(), upper.to_numpy(), color=color, alpha=.13, linewidth=0)
        axis.plot(x, center.to_numpy(), color=color, lw=1.35, label=label, solid_capstyle="round")
        plot_rows.append(pd.DataFrame({
            "global_step": valid["global_step"].to_numpy(), "epoch": valid["epoch"].to_numpy(),
            "split": "train", "metric": metric,
            "rolling_median": center.to_numpy(), "rolling_q25": lower.to_numpy(),
            "rolling_q75": upper.to_numpy(), "center_window": center_size,
            "band_window": band_size,
        }))

    if validation_steps is not None and not validation_steps.empty:
        validation_window = max(31, center_window // 5)
        for metric in ("loss_u", "loss_s"):
            label, color = palette[metric]
            valid = validation_steps.loc[
                validation_steps[metric].notna(), ["mapped_global_step", "epoch", metric, "mapping_status"]
            ].sort_values("mapped_global_step")
            center = valid[metric].rolling(validation_window, center=True, min_periods=1).median()
            axis.plot(
                valid["mapped_global_step"], center, color=color, lw=1.2,
                ls=(0, (5, 3)), alpha=.95, solid_capstyle="round",
            )
            plot_rows.append(pd.DataFrame({
                "global_step": valid["mapped_global_step"].to_numpy(),
                "epoch": valid["epoch"].to_numpy(), "split": "validation_reconstructed",
                "metric": metric, "rolling_median": center.to_numpy(),
                "rolling_q25": np.nan, "rolling_q75": np.nan,
                "center_window": validation_window, "band_window": np.nan,
                "mapping_status": valid["mapping_status"].to_numpy(),
            }))

    if resume_step is not None:
        axis.axvline(resume_step, color="#64748B", lw=1.25, ls=(0, (4, 3)), label="Resume boundary")
    axis.text(
        .985, .965, "Line: rolling median\nBand: training interquartile range",
        transform=axis.transAxes, ha="right", va="top", fontsize=11, color="#64748B",
    )
    axis.set(
        title="scUS pretraining convergence",
        xlabel="Training step", ylabel="Masked-token cross-entropy",
    )
    axis.set_title("scUS pretraining convergence", fontsize=20, fontweight="semibold", pad=38)
    axis.xaxis.label.set_size(15)
    axis.yaxis.label.set_size(15)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k" if value else "0"))
    axis.grid(axis="y", color="#CBD5E1", alpha=.55, lw=.8)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#94A3B8")
    axis.tick_params(colors="#334155", labelsize=12)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=palette["loss_u"][1], lw=1.35, label="Unspliced (U)"),
        Line2D([0], [0], color=palette["loss_s"][1], lw=1.35, label="Spliced (S)"),
        Line2D([0], [0], color="#334155", lw=1.35, label="Train"),
    ]
    if validation_steps is not None and not validation_steps.empty:
        legend_handles.append(Line2D([0], [0], color="#334155", lw=1.2, ls=(0, (5, 3)), label="Validation"))
    axis.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=4, frameon=False, fontsize=13)

    epoch_positions = (
        step.loc[step["epoch"].notna(), ["epoch", "global_step"]]
        .assign(epoch=lambda frame: frame["epoch"].astype(int))
        .groupby("epoch", as_index=False).global_step.median()
        .sort_values("epoch")
    )
    if not epoch_positions.empty:
        epoch_axis = axis.twiny()
        epoch_axis.set_xlim(axis.get_xlim())
        epoch_axis.set_xticks(epoch_positions.global_step)
        epoch_axis.set_xticklabels(epoch_positions.epoch.astype(str))
        epoch_axis.set_xlabel("Epoch", color="#64748B", labelpad=8, fontsize=14)
        epoch_axis.tick_params(axis="x", colors="#64748B", labelsize=11, length=3)
        epoch_axis.spines["top"].set_color("#CBD5E1")
        epoch_axis.spines[["left", "right", "bottom"]].set_visible(False)

    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    pd.concat(plot_rows, ignore_index=True).to_csv(destination.with_suffix(".csv"), index=False)
    return destination


def plot_checkpoint_validation(checkpoints: pd.DataFrame, output: Path) -> None:
    plt, figure, axes = _axes_grid(2, 3, (16, 9))
    specs = [
        ("val_loss_total", "Validation cross-entropy", None),
        ("perplexity", "Perplexity", None),
        ("masked_accuracy", "Masked-bin accuracy", (0, 1)),
        ("macro_f1", "Macro-F1", (0, 1)),
        ("expected_calibration_error", "Expected calibration error", (0, 1)),
        ("true_matched_shuffle_effect", "True vs matched-shuffle effect", None),
    ]
    for axis, (metric, title, limits) in zip(axes, specs):
        values = checkpoints.get(metric, pd.Series(dtype=float))
        if len(values) and values.notna().any():
            axis.plot(checkpoints["epoch"], values, marker="o", color="#2563EB")
            chosen = checkpoints["selected"].fillna(False)
            axis.scatter(checkpoints.loc[chosen, "epoch"], values.loc[chosen], s=80, color="#7C3AED", zorder=3)
        else:
            _not_logged(axis, "fixed-token re-evaluation pending")
        if limits:
            axis.set_ylim(*limits)
        axis.set(title=title, xlabel="checkpoint epoch")
    figure.suptitle("Frozen-checkpoint validation performance", fontsize=15)
    _save(plt, figure, output)


def _protocol_validation(frame: pd.DataFrame) -> pd.DataFrame:
    validation = frame.loc[frame["split"].astype(str).str.lower().isin(["val", "validation"])].copy()
    return validation if not validation.empty else frame.copy()


def plot_align_training(frame: pd.DataFrame, output: Path) -> None:
    protocols = list(dict.fromkeys(frame["protocol"].astype(str)))
    diagnostic_fields = [
        "loss_total", "loss_adv", "loss_disc", "disc_accuracy",
        "geometry_u_spearman", "geometry_s_spearman",
        "raw_projected_distance_spearman", "checkpoint_selection_score",
    ]
    canonical_protocol = max(
        protocols,
        key=lambda name: int(
            frame.loc[frame["protocol"].astype(str) == name, diagnostic_fields]
            .notna().sum().sum()
        ),
    )
    for protocol in protocols:
        data = frame.loc[frame["protocol"].astype(str) == protocol].copy()
        validation = _protocol_validation(data).sort_values("epoch")
        train = data.loc[data["split"].astype(str).str.lower() == "train"].sort_values("global_step")
        plt, figure, axes = _axes_grid(2, 4, (19, 9))
        specs = [
            ("loss_total", "A  Total loss"),
            (("loss_adv", "loss_disc"), "B  Adversarial / discriminator"),
            ("disc_accuracy", "C  Discriminator accuracy"),
            ("loss_relational", "D  Relational preservation"),
            (("loss_variance", "loss_covariance", "loss_identity"), "E  Anti-collapse / identity"),
            (("adv_weight", "adapter_alpha"), "F  Warm-up / adapter alpha"),
            (("distance_median", "distance_iqr"), "G  Validation distance"),
            ("checkpoint_selection_score", "H  Checkpoint selection score"),
        ]
        for axis, (metrics, title) in zip(axes, specs):
            metrics = (metrics,) if isinstance(metrics, str) else metrics
            plotted = False
            for metric in metrics:
                source = validation if validation[metric].notna().any() else train
                x = "epoch" if source is validation else "global_step"
                valid = source.loc[source[metric].notna()]
                if not valid.empty:
                    axis.plot(valid[x], valid[metric], marker="o" if source is validation else None, lw=1.4, label=metric.replace("loss_", ""))
                    plotted = True
            if not plotted:
                _not_logged(axis)
            elif len(metrics) > 1:
                axis.legend(frameon=False, fontsize=8)
            if metrics == ("disc_accuracy",):
                axis.axhline(0.5, ls="--", color="0.45", lw=1, label="chance")
            axis.set_title(title)
        figure.suptitle(f"scUS-Align training diagnostics — {protocol}", fontsize=15)
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", protocol).strip("_") or "align"
        target = output if len(protocols) == 1 else output.with_name(f"{output.name}_{suffix}")
        # Keep the documented canonical filename as a convenient entry point,
        # while retaining one non-overwritten figure per Align protocol.
        if len(protocols) > 1 and protocol == canonical_protocol:
            figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
            figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        _save(plt, figure, target)


def plot_align_qc(frame: pd.DataFrame, output: Path) -> None:
    validation_parts = []
    for protocol, part in frame.groupby("protocol", sort=False):
        selected = _protocol_validation(part).copy()
        selected["protocol"] = protocol
        validation_parts.append(selected)
    validation = pd.concat(validation_parts, ignore_index=True) if validation_parts else frame.iloc[0:0].copy()
    plt, figure, axes = _axes_grid(2, 4, (19, 9))
    specs = [
        ("effective_rank", "Effective rank", None),
        ("qc_projected_raw_distance_std_ratio", "Projected / Raw distance SD", 0.5),
        ("geometry_u_spearman", "Within-U geometry Spearman", 0.9),
        ("geometry_s_spearman", "Within-S geometry Spearman", 0.9),
        ("raw_projected_distance_spearman", "Raw/projected distance Spearman", 0.9),
        ("knn_overlap", "kNN overlap", 0.9),
        ("qc_probe_balanced_accuracy", "Modality probe balanced accuracy", 0.5),
        ("checkpoint_selection_score", "Selection score", None),
    ]
    for axis, (metric, title, threshold) in zip(axes, specs):
        plotted = False
        for protocol, data in validation.groupby("protocol"):
            valid = data.loc[data[metric].notna()].sort_values("epoch") if metric in data else data.iloc[0:0]
            if not valid.empty:
                axis.plot(valid["epoch"], valid[metric], marker="o", label=protocol)
                plotted = True
        if threshold is not None:
            axis.axhline(threshold, ls="--", color="0.45", lw=1)
        if not plotted:
            _not_logged(axis)
        axis.set(title=title, xlabel="epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    figure.suptitle("scUS-Align anti-collapse and structure-preservation QC", fontsize=15)
    _save(plt, figure, output)


def plot_training_downstream_dashboard(frame: pd.DataFrame, downstream: pd.DataFrame | None, output: Path) -> pd.DataFrame:
    validation = frame.loc[frame["split"].astype(str).str.lower().isin(["val", "validation"])].copy()
    available = [
        name for name in ["loss_total", "disc_accuracy", "effective_rank", "geometry_u_spearman", "geometry_s_spearman", "distance_iqr", "matched_shuffle_effect", "checkpoint_selection_score"]
        if name in validation and validation[name].notna().any()
    ]
    downstream_metrics = ["cell_type_balanced_accuracy", "time_spearman", "adjacent_stage_accuracy", "perturbation_auprc", "tpr_at_5fpr", "cross_splice_surprisal_auroc"]
    correlations = []
    merged = validation
    if downstream is not None and not downstream.empty:
        keys = [key for key in ["protocol", "seed", "epoch"] if key in downstream and key in validation]
        merged = validation.merge(downstream, on=keys, how="inner", suffixes=("", "_downstream")) if keys else validation.iloc[0:0]
        for left in available:
            for right in downstream_metrics:
                if right in merged and merged[[left, right]].dropna().shape[0] >= 3:
                    value = merged[[left, right]].corr(method="spearman").iloc[0, 1]
                    correlations.append({"training_metric": left, "downstream_metric": right, "spearman": value, "n": merged[[left, right]].dropna().shape[0]})
    correlation_frame = pd.DataFrame(correlations, columns=["training_metric", "downstream_metric", "spearman", "n"])

    plt, figure, axes = _axes_grid(2, 3, (16, 9))
    training_specs = [("loss_total", "Validation loss"), ("disc_accuracy", "Modality accuracy"), ("checkpoint_selection_score", "Selection score")]
    for axis, (metric, title) in zip(axes[:3], training_specs):
        plotted = False
        if metric in validation:
            for protocol, data in validation.groupby("protocol"):
                valid = data.loc[data[metric].notna()].sort_values("epoch")
                if not valid.empty:
                    axis.plot(valid["epoch"], valid[metric], marker="o", label=protocol); plotted = True
        if not plotted: _not_logged(axis)
        axis.set(title=title, xlabel="epoch")
    for axis, metric in zip(axes[3:], downstream_metrics[:3]):
        if metric in merged and merged[metric].notna().any():
            for protocol, data in merged.groupby("protocol"):
                valid = data.loc[data[metric].notna()].sort_values("epoch")
                axis.plot(valid["epoch"], valid[metric], marker="o", label=protocol)
        else:
            _not_logged(axis, f"{metric}\nnot logged per checkpoint")
        axis.set(title=metric.replace("_", " "), xlabel="epoch")
    figure.suptitle("Training-to-downstream checkpoint dashboard", fontsize=15)
    _save(plt, figure, output)
    return correlation_frame


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode()); digest.update(b"\n")
    return digest.hexdigest()


def write_report(
    output: Path,
    canonical: pd.DataFrame,
    checkpoints: pd.DataFrame,
    align: pd.DataFrame,
    correlations: pd.DataFrame,
    resume_step: int,
    selected_epoch: int,
) -> None:
    epoch = canonical.loc[canonical["split"].isin(["train", "val"])]
    train = epoch.loc[epoch["split"] == "train"].dropna(subset=["loss_total"]).sort_values("epoch")
    val = epoch.loc[epoch["split"] == "val"].dropna(subset=["loss_total"]).sort_values("epoch")
    selected = checkpoints.loc[checkpoints["epoch"] == selected_epoch]
    val_loss = float(selected.iloc[0]["val_loss_total"]) if not selected.empty and pd.notna(selected.iloc[0]["val_loss_total"]) else float("nan")
    available_align = [column for column in STANDARD_COLUMNS if column in align and align[column].notna().any()]
    missing_align = [column for column in ["loss_relational", "loss_variance", "loss_covariance", "loss_identity", "effective_rank", "distance_iqr", "knn_overlap", "matched_shuffle_effect", "coverage"] if column not in available_align]
    report = f"""# scUS training diagnostics report

## Scope

- Canonical pretraining log: version 2 before step {resume_step:,}, version 4 from that step onward.
- Historical versions 0/1 are exploratory interrupted runs; version 3 marks the resume boundary.
- Selected historical checkpoint: epoch {selected_epoch}.
- The frozen Transformer was not retrained.

## Pretraining convergence

- Canonical metric rows: {len(canonical):,}.
- Epoch summaries: train={len(train)}, validation={len(val)}.
- First/final train loss: {train.iloc[0]['loss_total'] if not train.empty else float('nan'):.6f} / {train.iloc[-1]['loss_total'] if not train.empty else float('nan'):.6f}.
- First/final validation loss: {val.iloc[0]['loss_total'] if not val.empty else float('nan'):.6f} / {val.iloc[-1]['loss_total'] if not val.empty else float('nan'):.6f}.
- Logged epoch-{selected_epoch} validation loss: {val_loss:.6f}.
- Validation checkpoint rows hash: `{_sha256_lines(checkpoints['checkpoint'].astype(str))}`.

The original fixed validation-token identities were not found in an index format consumed by the public package. Consequently masked-bin accuracy, macro-F1, calibration, cross-splice surprisal, random-checkpoint comparison, and empirical-frequency comparison are marked as pending rather than reconstructed from loss.

## Align diagnostics

- Protocols: {', '.join(sorted(align['protocol'].dropna().astype(str).unique())) if not align.empty else 'none'}.
- Canonical Align rows: {len(align):,}.
- Available standardized metrics: {', '.join(available_align)}.
- Metrics not logged by the supplied historical runs: {', '.join(missing_align) if missing_align else 'none'}.

Historical adversarial-only runs are labeled as legacy adapters. A discriminator accuracy near 0.5 is not treated as success unless geometry, rank, distance spread, coverage, and downstream-signal gates are also available and pass.
Historical `qc_eligible` values are retained as source fields but are not promoted to current `checkpoint_eligible`, because the old rule did not contain all current gates.

## Training-to-downstream audit

- Checkpoint-aligned downstream correlations available: {len(correlations)}.
- The dashboard displays `not logged per checkpoint` where held-out biological metrics are absent; it does not impute values.

## Acceptance status

- Pretraining loss curves: **available**.
- U/S validation losses: **available**.
- Fixed-token checkpoint re-evaluation: **pending fixed validation token index**.
- Historical Align optimization curves: **available**.
- Align anti-collapse QC: **partial**, limited to fields present in each supplied log.
- Checkpoint-to-held-out biological-performance correlation: **pending checkpoint-level downstream table**.

The figures demonstrate optimization and the availability of QC evidence. They do not by themselves establish biological validity.
"""
    output.write_text(report)


def run_training_diagnostics(
    pretrain_log_root: str | Path,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    align_logs: Iterable[tuple[str, str | Path]] = (),
    downstream_metrics: str | Path | None = None,
    resume_step: int = 191_380,
    selected_epoch: int = 11,
    seed: int = 618,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    canonical = canonical_pretraining_metrics(pretrain_log_root, resume_step, seed)
    checkpoints = checkpoint_evaluation_from_logs(checkpoint_dir, canonical, selected_epoch)
    align_frames = [canonical_align_metrics(path, protocol) for protocol, path in align_logs]
    align = pd.concat(align_frames, ignore_index=True) if align_frames else _empty_standard(0)
    downstream = pd.read_csv(downstream_metrics) if downstream_metrics else None

    canonical.to_csv(output / "pretraining_metrics_canonical.csv", index=False)
    checkpoints.to_csv(output / "pretraining_checkpoint_evaluation.csv", index=False)
    align.to_csv(output / "align_metrics_long.csv", index=False)
    selection_columns = [
        "protocol", "seed", "epoch", "checkpoint_eligible",
        "checkpoint_selection_score", "qc_eligible", "qc_selection_score",
    ]
    selection = align.loc[
        align["split"].astype(str).str.lower().isin(["val", "validation"]),
        [column for column in selection_columns if column in align],
    ].copy()
    selection = selection.sort_values(["protocol", "seed", "epoch"]).drop_duplicates(
        ["protocol", "seed", "epoch"], keep="last"
    )
    selection["current_gate_status"] = np.where(
        selection["checkpoint_eligible"].notna(), "evaluated", "not_evaluable_missing_current_qc"
    )
    selection["selected"] = False
    for (_, _), group in selection.groupby(["protocol", "seed"], dropna=False):
        eligible = group.loc[group["checkpoint_eligible"].eq(True)]
        scored = eligible.dropna(subset=["checkpoint_selection_score"])
        if not scored.empty:
            selection.loc[scored["checkpoint_selection_score"].idxmax(), "selected"] = True
    selection.to_csv(output / "checkpoint_selection.csv", index=False)

    plot_pretraining(canonical, checkpoints, output / "01_pretraining_training_curves", resume_step, selected_epoch)
    plot_checkpoint_validation(checkpoints, output / "02_pretraining_validation_performance")
    if not align.empty:
        plot_align_training(align, output / "03_align_training_curves")
        plot_align_qc(align, output / "04_align_anticollapse_qc")
    else:
        plt, figure, axes = _axes_grid(1, 1, (8, 5)); _not_logged(axes[0], "no Align log supplied"); _save(plt, figure, output / "03_align_training_curves")
        plt, figure, axes = _axes_grid(1, 1, (8, 5)); _not_logged(axes[0], "no Align QC log supplied"); _save(plt, figure, output / "04_align_anticollapse_qc")
    correlations = plot_training_downstream_dashboard(align, downstream, output / "05_training_downstream_dashboard")
    correlations.to_csv(output / "training_downstream_correlations.csv", index=False)
    write_report(output / "TRAINING_DIAGNOSTICS_REPORT.md", canonical, checkpoints, align, correlations, resume_step, selected_epoch)
    return output

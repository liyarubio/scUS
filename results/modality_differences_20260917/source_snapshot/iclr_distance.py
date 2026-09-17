"""ICLR distance-core workflows built on frozen, auditable scUS artifacts.

The calibration implemented here never changes the encoder.  All location and
scale statistics are fitted on the reference biological sample and are then
applied unchanged to the held-out sample.  Missing U/S pairs remain NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown(table: pd.DataFrame) -> str:
    def render(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, table.columns)) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(render(v) for v in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


@dataclass
class DistanceCalibrator:
    gene_median: np.ndarray
    bin_residual: np.ndarray
    gene_mad: np.ndarray
    eligible: np.ndarray
    observation_count: np.ndarray
    fallback_mad: float
    mad_fallback: np.ndarray


def fit_distance_calibrator(
    distance: np.ndarray,
    u_bins: np.ndarray,
    s_bins: np.ndarray,
    train_indices: Iterable[int],
    min_observations: int = 50,
) -> DistanceCalibrator:
    """Fit gene/value conditional baselines using reference cells only."""
    train = np.asarray(list(train_indices), dtype=np.int64)
    d = np.asarray(distance[train], dtype=np.float64)
    u = np.asarray(u_bins[train], dtype=np.uint8)
    s = np.asarray(s_bins[train], dtype=np.uint8)
    if not (d.shape == u.shape == s.shape):
        raise ValueError("Distance and bin arrays must have identical shapes")
    observed = np.isfinite(d)
    if not np.array_equal(observed, (u > 0) & (s > 0)):
        raise ValueError("Distance missingness must exactly match paired positive U/S bins")
    count = observed.sum(axis=0)
    eligible = count >= int(min_observations)
    if not eligible.any():
        raise ValueError("No gene satisfies the minimum reference observation count")
    gene_median = np.full(d.shape[1], np.nan, dtype=np.float64)
    gene_median[eligible] = np.nanmedian(d[:, eligible], axis=0)
    residual = d[:, eligible] - gene_median[eligible]
    code = u[:, eligible].astype(np.int16) * 16 + s[:, eligible].astype(np.int16)
    valid = np.isfinite(residual)
    flat_code = code[valid]
    flat_residual = residual[valid]
    order = np.argsort(flat_code, kind="stable")
    sorted_code, sorted_residual = flat_code[order], flat_residual[order]
    bin_residual = np.zeros((16, 16), dtype=np.float64)
    if len(sorted_code):
        unique, first, sizes = np.unique(sorted_code, return_index=True, return_counts=True)
        for value, start, size in zip(unique, first, sizes):
            uu, ss = divmod(int(value), 16)
            if 1 <= uu <= 15 and 1 <= ss <= 15:
                bin_residual[uu, ss] = float(np.median(sorted_residual[start : start + size]))
    corrected = residual.copy()
    corrected[valid] -= bin_residual[u[:, eligible][valid], s[:, eligible][valid]]
    center = np.nanmedian(corrected, axis=0)
    mad_selected = np.nanmedian(np.abs(corrected - center), axis=0)
    positive = mad_selected[np.isfinite(mad_selected) & (mad_selected > 0)]
    if not len(positive):
        raise ValueError("All eligible genes have zero or non-finite MAD")
    fallback = float(np.median(positive))
    fallback_selected = ~(np.isfinite(mad_selected) & (mad_selected > 0))
    mad_selected = np.where(~fallback_selected, mad_selected, fallback)
    gene_mad = np.full(d.shape[1], np.nan, dtype=np.float64)
    gene_mad[eligible] = mad_selected
    mad_fallback = np.zeros(d.shape[1], dtype=bool)
    mad_fallback[eligible] = fallback_selected
    return DistanceCalibrator(gene_median, bin_residual, gene_mad, eligible, count, fallback, mad_fallback)


def apply_distance_calibrator(
    distance: np.ndarray,
    u_bins: np.ndarray,
    s_bins: np.ndarray,
    calibrator: DistanceCalibrator,
) -> np.ndarray:
    """Apply a frozen calibrator; ineligible genes and missing pairs stay NaN."""
    d = np.asarray(distance, dtype=np.float64)
    u = np.asarray(u_bins, dtype=np.uint8)
    s = np.asarray(s_bins, dtype=np.uint8)
    if not (d.shape == u.shape == s.shape):
        raise ValueError("Distance and bin arrays must have identical shapes")
    out = np.full(d.shape, np.nan, dtype=np.float32)
    cols = np.flatnonzero(calibrator.eligible)
    observed = np.isfinite(d[:, cols])
    values = d[:, cols] - calibrator.gene_median[cols]
    lookup = calibrator.bin_residual[u[:, cols], s[:, cols]]
    values = (values - lookup) / calibrator.gene_mad[cols]
    values[~observed] = np.nan
    out[:, cols] = values.astype(np.float32)
    return out


def reference_standardize(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(reference, axis=0)
    std = np.nanstd(reference, axis=0)
    if not np.isfinite(mean).all():
        raise ValueError("A selected feature is entirely missing in the reference sample")
    scale = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return (reference - mean) / scale, (query - mean) / scale, mean, std


def shared_observed_rms(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RMS over jointly observed features without imputing missing pairs."""
    ml = np.isfinite(left).astype(np.float64)
    mr = np.isfinite(right).astype(np.float64)
    zl = np.where(np.isfinite(left), left, 0.0).astype(np.float64)
    zr = np.where(np.isfinite(right), right, 0.0).astype(np.float64)
    count = ml @ mr.T
    if (count == 0).any():
        raise ValueError("At least one cell pair has no jointly observed feature")
    squared = (zl * zl) @ mr.T + ml @ (zr * zr).T - 2.0 * (zl @ zr.T)
    return np.sqrt(np.maximum(squared, 0.0) / count), count


def _mask_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = left.astype(np.float64), right.astype(np.float64)
    intersection = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
    if (union == 0).any():
        raise ValueError("Empty detection mask encountered")
    return 1.0 - intersection / union


def _bins_from_tokens(token_path: Path, vocab_path: Path, cells: pd.DataFrame, genes: pd.DataFrame):
    from .joint_reconstruction import FullTokenDataset

    data = FullTokenDataset(token_path, vocab_path)
    if not np.array_equal(data.cells, cells.cell_id.astype(str).to_numpy()):
        raise ValueError("Token and distance cell order differs")
    lookup = {int(g): i for i, g in enumerate(genes.vocab_id)}
    u = np.zeros((len(cells), len(genes)), dtype=np.uint8)
    s = np.zeros_like(u)
    input_hashes = []
    for i in range(len(cells)):
        item = data[i]
        gene = item["gene_ids"][0].numpy()
        value = item["value_bins"][0].numpy()
        cols = np.fromiter((lookup[int(g)] for g in gene[::2]), dtype=np.int64, count=len(gene) // 2)
        u[i, cols], s[i, cols] = value[::2], value[1::2]
        digest = hashlib.sha256()
        for key in ("gene_ids", "value_bins", "splice_flags"):
            digest.update(item[key].numpy().tobytes())
        input_hashes.append(digest.hexdigest())
    return u, s, input_hashes


def _prediction_metrics(y_true, predicted, probability, classes) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "macro_auprc": float(average_precision_score(y_true[:, None] == classes[None, :], probability, average="macro")),
    }


def _profile_distance_matrices(
    matrix: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    panel: np.ndarray,
    metric: str = "rms",
):
    a, b = np.asarray(matrix[np.ix_(train, panel)], float), np.asarray(matrix[np.ix_(test, panel)], float)
    if metric == "rms":
        a, b, _, _ = reference_standardize(a, b)
        d_train, train_common = shared_observed_rms(a, a)
        d_test, test_common = shared_observed_rms(b, a)
    elif metric == "jaccard":
        d_train, d_test = _mask_jaccard(a, a), _mask_jaccard(b, a)
        train_common = np.full(d_train.shape, panel.sum())
        test_common = np.full(d_test.shape, panel.sum())
    else:
        raise ValueError(metric)
    np.fill_diagonal(d_train, 0.0)
    return d_train, d_test, train_common, test_common


def _knn_from_distances(
    d_train: np.ndarray,
    d_test: np.ndarray,
    test_common: np.ndarray,
    meta: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    feature_count: int,
    k: int,
):
    y, classes = meta.cell_type.to_numpy(), np.sort(meta.cell_type.unique())
    model = KNeighborsClassifier(n_neighbors=k, metric="precomputed", weights="distance", n_jobs=4)
    model.fit(d_train, y[train])
    pred, prob = model.predict(d_test), model.predict_proba(d_test)
    result = _prediction_metrics(y[test], pred, prob, classes)
    result.update(
        minimum_shared_features=int(test_common.min()),
        median_shared_features=float(np.median(test_common)),
        train_cells=int(len(train)),
        test_cells=int(len(test)),
        features=int(feature_count),
    )
    predictions = pd.DataFrame({"cell_id": meta.cell_id.iloc[test], "true_label": y[test], "predicted_label": pred})
    return result, predictions


def _cellwise_gene_shuffle(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(matrix)
    for i in range(len(matrix)):
        out[i] = matrix[i, rng.permutation(matrix.shape[1])]
    return out


def _pca_block(train: np.ndarray, test: np.ndarray, seed: int, components: int = 128):
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    a = np.where(np.isfinite(train), train, median)
    b = np.where(np.isfinite(test), test, median)
    scaler = StandardScaler().fit(a)
    a, b = scaler.transform(a), scaler.transform(b)
    n = min(components, len(a) - 1, a.shape[1])
    transform = PCA(n_components=n, svd_solver="randomized", random_state=seed).fit(a)
    return transform.transform(a), transform.transform(b), n


def _logistic_select_c(x_train, y_train, seed: int, choices=(0.01, 0.1, 1.0, 10.0)) -> float:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = []
    for c in choices:
        fold = []
        for fit, val in splitter.split(x_train, y_train):
            model = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", solver="lbfgs")
            model.fit(x_train[fit], y_train[fit])
            fold.append(balanced_accuracy_score(y_train[val], model.predict(x_train[val])))
        scores.append(float(np.mean(fold)))
    return float(choices[int(np.argmax(scores))])


def _incremental_benchmark(
    u: np.ndarray,
    s: np.ndarray,
    calibrated: np.ndarray,
    state: np.ndarray,
    meta: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    panel: np.ndarray,
    seed: int,
):
    paired = (u > 0) & (s > 0)
    u_value, s_value = u.astype(float), s.astype(float)
    u_value[u == 0], s_value[s == 0] = np.nan, np.nan
    delta = u_value - s_value
    blocks = {
        "u_bins": u_value[:, panel],
        "s_bins": s_value[:, panel],
        "direct_difference": delta[:, panel],
        "pair_mask": paired[:, panel].astype(float),
    }
    train_blocks, test_blocks, audit = [], [], []
    for name, values in blocks.items():
        a, b, n = _pca_block(values[train], values[test], seed)
        train_blocks.append(a)
        test_blocks.append(b)
        audit.append({"block": name, "components": n, "source_features": values.shape[1]})
    state_scaler = StandardScaler().fit(state[train])
    train_blocks.append(state_scaler.transform(state[train]))
    test_blocks.append(state_scaler.transform(state[test]))
    audit.append({"block": "pooled_state", "components": state.shape[1], "source_features": state.shape[1]})
    base_train, base_test = np.concatenate(train_blocks, 1), np.concatenate(test_blocks, 1)
    profiles = {
        "calibrated_distance": calibrated[:, panel],
        "gene_shuffled_distance": _cellwise_gene_shuffle(calibrated[:, panel], seed),
    }
    profile_blocks = {}
    for name, values in profiles.items():
        a, b, n = _pca_block(values[train], values[test], seed)
        profile_blocks[name] = (a, b)
        audit.append({"block": name, "components": n, "source_features": values.shape[1]})
    rng = np.random.default_rng(seed)
    random_train = rng.standard_normal(profile_blocks["calibrated_distance"][0].shape)
    random_test = rng.standard_normal(profile_blocks["calibrated_distance"][1].shape)
    additions = {
        "B0": (None, None),
        "B1_calibrated": profile_blocks["calibrated_distance"],
        "B0_plus_gene_shuffle": profile_blocks["gene_shuffled_distance"],
        "B0_plus_random": (random_train, random_test),
    }
    y, classes = meta.cell_type.to_numpy(), np.sort(meta.cell_type.unique())
    rows, predictions = [], []
    for name, (extra_train, extra_test) in additions.items():
        a = base_train if extra_train is None else np.concatenate([base_train, extra_train], 1)
        b = base_test if extra_test is None else np.concatenate([base_test, extra_test], 1)
        c = _logistic_select_c(a, y[train], seed)
        model = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", solver="lbfgs")
        model.fit(a, y[train])
        pred, prob = model.predict(b), model.predict_proba(b)
        result = {"model": name, "selected_C": c, "features_after_transform": a.shape[1]}
        result.update(_prediction_metrics(y[test], pred, prob, classes))
        rows.append(result)
        for i, cell in enumerate(test):
            predictions.append({"model": name, "cell_id": meta.cell_id.iloc[cell], "true_label": y[cell], "predicted_label": pred[i]})
    return pd.DataFrame(rows), pd.DataFrame(predictions), pd.DataFrame(audit)


def _paths(cfg: dict):
    settings = cfg["iclr_distance"]
    source = Path(settings["forebrain_distance_root"])
    out = Path(cfg["output_dir"])
    return settings, source, out


def calibrate(cfg: dict) -> Path:
    settings, source, root = _paths(cfg)
    out = root / "calibration" / "forebrain"
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "run_config.json", cfg)
    _atomic_json(out / "status.json", {"status": "running", "started": time.time()})
    distance_path = source / "zero_shot" / "cell_gene_distance.npy"
    cells_path, genes_path = source / "cells.csv", source / "input_genes.csv"
    token_path, vocab_path = Path(cfg["data"]["token_path"]), Path(cfg["vocab"])
    distance = np.load(distance_path, mmap_mode="r")
    cells, genes = pd.read_csv(cells_path), pd.read_csv(genes_path)
    if distance.shape != (len(cells), len(genes)):
        raise ValueError("Distance matrix metadata mismatch")
    u, s, token_hashes = _bins_from_tokens(token_path, vocab_path, cells, genes)
    if not np.array_equal(np.isfinite(distance), (u > 0) & (s > 0)):
        raise ValueError("Historical raw-distance mask does not match the original tokens")
    reference_sample = settings.get("reference_sample", "10X_17_029")
    heldout_sample = settings.get("heldout_sample", "10X_17_028")
    reference = np.flatnonzero(cells["sample"].eq(reference_sample))
    heldout = np.flatnonzero(cells["sample"].eq(heldout_sample))
    if not len(reference) or not len(heldout) or set(reference) & set(heldout):
        raise ValueError("Invalid reference/held-out biological sample split")
    fitted = fit_distance_calibrator(distance, u, s, reference, settings.get("min_gene_observations", 50))
    calibrated = apply_distance_calibrator(distance, u, s, fitted)
    np.save(out / "calibrated_distance.npy", calibrated)
    np.savez_compressed(out / "input_bins.npz", u=u, s=s)
    gene_table = genes.copy()
    gene_table["reference_observations"] = fitted.observation_count
    gene_table["eligible"] = fitted.eligible
    gene_table["distance_median"] = fitted.gene_median
    gene_table["corrected_mad"] = fitted.gene_mad
    gene_table["mad_fallback"] = fitted.mad_fallback
    gene_table.to_csv(out / "gene_calibration.csv", index=False)
    bin_rows = []
    selected_u = u[np.ix_(reference, fitted.eligible)]
    selected_s = s[np.ix_(reference, fitted.eligible)]
    selected_distance = np.asarray(distance[np.ix_(reference, fitted.eligible)])
    for uu in range(1, 16):
        for ss in range(1, 16):
            n = int(((selected_u == uu) & (selected_s == ss) & np.isfinite(selected_distance)).sum())
            bin_rows.append({"u_bin": uu, "s_bin": ss, "reference_observations": n, "median_residual": fitted.bin_residual[uu, ss], "fallback_zero": n == 0})
    pd.DataFrame(bin_rows).to_csv(out / "bin_pair_calibration.csv", index=False)
    pd.DataFrame({"cell_id": cells.cell_id, "token_sha256": token_hashes}).to_csv(out / "cell_input_hashes.csv", index=False)
    audit = {
        "status": "complete",
        "definition": "(raw_distance - gene_median - bin_pair_residual) / gene_MAD",
        "reference_sample": reference_sample,
        "heldout_sample": heldout_sample,
        "reference_cells": len(reference),
        "heldout_cells": len(heldout),
        "genes": len(genes),
        "eligible_genes": int(fitted.eligible.sum()),
        "fallback_mad": fitted.fallback_mad,
        "fallback_mad_genes": int(fitted.mad_fallback.sum()),
        "missing_pairs_preserved": bool(np.array_equal(np.isfinite(calibrated), np.isfinite(distance) & fitted.eligible[None, :])),
        "inputs": {str(path): _sha256(path) for path in (distance_path, cells_path, genes_path, token_path, vocab_path)},
        "pretraining_corpus_overlap": "unknown",
    }
    _atomic_json(out / "status.json", audit)
    return out


def _benchmark_forebrain(cfg: dict) -> Path:
    settings, source, root = _paths(cfg)
    calibration = root / "calibration" / "forebrain"
    if json.loads((calibration / "status.json").read_text()).get("status") != "complete":
        raise ValueError("Run `scus semantics calibrate` first")
    out = root / "benchmark" / "forebrain"
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "run_config.json", cfg)
    _atomic_json(out / "status.json", {"status": "running", "started": time.time()})
    cells, genes = pd.read_csv(source / "cells.csv"), pd.read_csv(source / "input_genes.csv")
    raw = np.load(source / "zero_shot" / "cell_gene_distance.npy", mmap_mode="r")
    calibrated = np.load(calibration / "calibrated_distance.npy", mmap_mode="r")
    bins = np.load(calibration / "input_bins.npz")
    u, s = bins["u"], bins["s"]
    gene_calibration = pd.read_csv(calibration / "gene_calibration.csv")
    panel = gene_calibration.eligible.to_numpy(bool)
    reference = np.flatnonzero(cells["sample"].eq(settings.get("reference_sample", "10X_17_029")))
    heldout = np.flatnonzero(cells["sample"].eq(settings.get("heldout_sample", "10X_17_028")))
    direct = u.astype(float) - s.astype(float)
    direct[(u == 0) | (s == 0)] = np.nan
    observed = np.isfinite(raw).astype(float)
    raw_scalar = np.nanmedian(raw[:, panel], axis=1)[:, None]
    raw_scalar_panel = np.ones(1, dtype=bool)
    shuffled = _cellwise_gene_shuffle(np.asarray(raw[:, panel]), int(cfg.get("seed", 42)))
    representations = {
        "raw_distance_profile": (raw, panel, "rms"),
        "calibrated_distance_profile": (calibrated, panel, "rms"),
        "direct_bin_difference": (direct, panel, "rms"),
        "absolute_bin_difference": (np.abs(direct), panel, "rms"),
        "pair_mask": (observed, panel, "jaccard"),
        "distance_scalar": (raw_scalar, raw_scalar_panel, "rms"),
        "gene_shuffled_distance": (shuffled, np.ones(shuffled.shape[1], bool), "rms"),
    }
    pooled_path = Path(settings["pooled_embeddings_path"])
    pooled = np.load(pooled_path)
    if not np.array_equal(pooled["cell_ids"].astype(str), cells.cell_id.astype(str).to_numpy()):
        raise ValueError("Pooled embedding cell order differs from the distance matrix")
    for name in ("state", "u_only", "s_only", "concatenated_u_s", "signed_kinetic", "absolute_kinetic"):
        values = np.asarray(pooled[name], dtype=np.float32)
        representations[name] = (values, np.ones(values.shape[1], bool), "rms")
    moments_path = Path(settings["moments_path"])
    moments = np.load(moments_path)
    if not np.array_equal(moments["cell_ids"].astype(str), cells.cell_id.astype(str).to_numpy()):
        raise ValueError("Moments cell order differs from the distance matrix")
    if not np.array_equal(moments["gene_ids"], genes.vocab_id.to_numpy()):
        raise ValueError("Moments gene order differs from the distance matrix")
    mu, ms = np.asarray(moments["Mu"], float), np.asarray(moments["Ms"], float)
    moments_delta = mu - ms
    moments_delta[~np.isfinite(raw)] = np.nan
    representations.update({
        "continuous_moments_difference": (moments_delta, panel, "rms"),
        "continuous_moments_abs_difference": (np.abs(moments_delta), panel, "rms"),
        "continuous_moments_total": (mu + ms, panel, "rms"),
    })
    state_path = Path(settings["zero_shot_state_path"])
    _atomic_json(out / "input_hashes.json", {
        str(path): _sha256(path) for path in (
            source / "zero_shot" / "cell_gene_distance.npy", source / "cells.csv", source / "input_genes.csv",
            calibration / "calibrated_distance.npy", calibration / "status.json", pooled_path, moments_path, state_path,
        )
    })
    metrics, predictions = [], []
    for name, (values, use, metric) in representations.items():
        d_train, d_test, _, test_common = _profile_distance_matrices(values, reference, heldout, use, metric)
        for k in settings.get("neighbors", [5, 15, 50]):
            result, pred = _knn_from_distances(d_train, d_test, test_common, cells, reference, heldout, int(use.sum()), int(k))
            result.update({"representation": name, "k": int(k), "protocol": "reference_sample_fit_heldout_sample_test"})
            metrics.append(result)
            pred["representation"], pred["k"] = name, int(k)
            predictions.append(pred)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "profile_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "profile_predictions.csv", index=False)
    state = np.load(state_path)
    if state.shape[0] != len(cells):
        raise ValueError("Pooled state cell count mismatch")
    incremental, incremental_predictions, block_audit = _incremental_benchmark(
        u, s, calibrated, state, cells, reference, heldout, panel, int(cfg.get("seed", 42))
    )
    incremental.to_csv(out / "incremental_metrics.csv", index=False)
    incremental_predictions.to_csv(out / "incremental_predictions.csv", index=False)
    block_audit.to_csv(out / "incremental_block_audit.csv", index=False)
    _plot_benchmark(out, metrics, incremental)
    primary = metrics[metrics.k.eq(15)][["representation", "balanced_accuracy", "macro_f1", "macro_auprc"]]
    report = [
        "# Forebrain 条件校准与公平留出评估",
        "",
        "参考 sample 为 `10X_17_029`，所有校准、基因筛选、标准化及增量模型均只在该 sample 拟合；`10X_17_028` 仅用于一次外层测试。",
        "",
        "## 15-NN profile 结果",
        "",
        _markdown(primary),
        "",
        "## 增量模型",
        "",
        _markdown(incremental),
        "",
        "`calibrated distance` 是条件残差，不是异常概率。该数据只有两个 sample，内部分类器选择仍是细胞级交叉验证，不能视为独立生物重复。",
    ]
    (out / "REPORT_CN.md").write_text("\n".join(report) + "\n")
    _atomic_json(out / "status.json", {
        "status": "complete", "profile_rows": len(metrics), "eligible_genes": int(panel.sum()),
        "test_labels_used_for_fit": False,
        "unavailable_fair_baselines": {
            "cross_splice_surprisal": "Only a 128-target diagnostic exists; no full-gene matrix.",
            "random_init_distance": "No 1,720-cell all-gene random-init encoding exists yet.",
            "input_embedding_distance": "Only the current 128-cell layerwise diagnostic exists.",
        },
    })
    return out


def _erythroid_inputs(cfg: dict, out: Path):
    import anndata as ad
    import torch

    settings = cfg["iclr_distance"]["erythroid"]
    cache = out.parent / "input_cache"
    cache.mkdir(parents=True, exist_ok=True)
    distance_path, token_path = Path(settings["distance_path"]), Path(settings["token_path"])
    h5ad_path = Path(settings["h5ad_path"])
    with np.load(distance_path, allow_pickle=True) as source:
        distance = np.asarray(source["dist"], dtype=np.float32)
        barcodes = source["barcodes"].astype(str)
        cell_types = source["cell_types"].astype(str)
        gene_names = source["gene_names"].astype(str)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if not np.array_equal(adata.obs_names.astype(str), barcodes):
        raise ValueError("Erythroid H5AD and distance barcode order differs")
    meta = pd.DataFrame({
        "cell_id": barcodes,
        "sample": adata.obs["sample"].astype(str).to_numpy(),
        "stage": adata.obs["stage"].astype(str).to_numpy(),
        "cell_type": adata.obs["celltype"].astype(str).to_numpy(),
    })
    if not np.array_equal(cell_types, meta.cell_type):
        raise ValueError("Erythroid cell-type metadata differs between artifacts")
    u_path, s_path = cache / "u_bins.npy", cache / "s_bins.npy"
    if u_path.exists() and s_path.exists():
        u, s = np.load(u_path, mmap_mode="r"), np.load(s_path, mmap_mode="r")
    else:
        vocab = json.loads(Path(cfg["vocab"]).read_text())
        target_ids = np.asarray([vocab[str(name)] for name in gene_names], dtype=np.int64)
        id_to_col = np.full(max(max(vocab.values()), int(target_ids.max())) + 1, -1, dtype=np.int32)
        id_to_col[target_ids] = np.arange(len(target_ids), dtype=np.int32)
        shard = torch.load(token_path, map_location="cpu", weights_only=False)
        if not np.array_equal(np.asarray(shard["obs_barcode"], str), barcodes):
            raise ValueError("Erythroid token and distance barcode order differs")
        u = np.lib.format.open_memmap(u_path.with_suffix(".partial.npy"), mode="w+", dtype=np.uint8, shape=distance.shape)
        s = np.lib.format.open_memmap(s_path.with_suffix(".partial.npy"), mode="w+", dtype=np.uint8, shape=distance.shape)
        u[:] = 0; s[:] = 0
        for i in range(len(meta)):
            gene = torch.as_tensor(shard["gene_ids"][i]).numpy()[::2]
            values = torch.as_tensor(shard["value_bins"][i]).numpy().reshape(-1, 2)
            cols = id_to_col[gene]
            use = cols >= 0
            u[i, cols[use]], s[i, cols[use]] = values[use, 0], values[use, 1]
        u.flush(); s.flush(); del u, s, shard
        u_path.with_suffix(".partial.npy").replace(u_path)
        s_path.with_suffix(".partial.npy").replace(s_path)
        u, s = np.load(u_path, mmap_mode="r"), np.load(s_path, mmap_mode="r")
    if not (distance.shape == u.shape == s.shape == (len(meta), len(gene_names))):
        raise ValueError("Erythroid input shapes differ")
    if not np.array_equal(np.isfinite(distance), (u > 0) & (s > 0)):
        raise ValueError("Erythroid distance missing mask differs from token pairs")
    meta.to_csv(cache / "cells.csv", index=False)
    pd.DataFrame({"gene_name": gene_names}).to_csv(cache / "genes.csv", index=False)
    _atomic_json(cache / "audit.json", {
        "status": "complete", "cells": len(meta), "genes": len(gene_names), "samples": int(meta["sample"].nunique()),
        "stages": sorted(meta.stage.unique()), "missing_mask_identical": True,
        "inputs": {str(path): _sha256(path) for path in (distance_path, token_path, h5ad_path, Path(cfg["vocab"]))},
    })
    return distance, u, s, meta, gene_names


def _sample_centroids(matrix: np.ndarray, samples: np.ndarray, order: list[str]) -> np.ndarray:
    import warnings

    rows = []
    for sample in order:
        # Genes that are absent from every cell in one biological sample remain
        # missing.  NumPy warns for these columns even though NaN is the desired
        # and audited result.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            rows.append(np.nanmedian(np.asarray(matrix[samples == sample], dtype=np.float32), axis=0))
    return np.stack(rows)


def _stage_day(value: str) -> float:
    text = str(value).strip().upper()
    if not text.startswith("E"):
        raise ValueError(f"Unrecognized developmental stage: {value}")
    return float(text[1:])


def _erythroid_fold_prediction(
    representation: str,
    train_profiles: np.ndarray,
    test_profile: np.ndarray,
    train_stage: np.ndarray,
    true_stage: float,
    panel: np.ndarray,
    k: int,
) -> dict:
    a, b, _, _ = reference_standardize(train_profiles[:, panel], test_profile[None, panel])
    distance, common = shared_observed_rms(b, a)
    use = np.argsort(distance[0], kind="stable")[: min(k, len(train_profiles))]
    selected_distance = distance[0, use]
    weights = 1.0 / np.maximum(selected_distance, 1e-8)
    predicted = float(np.average(train_stage[use], weights=weights))
    levels = np.sort(np.unique(np.r_[train_stage, true_stage]))
    predicted_class = float(levels[np.argmin(np.abs(levels - predicted))])
    true_index = int(np.flatnonzero(levels == true_stage)[0])
    predicted_index = int(np.flatnonzero(levels == predicted_class)[0])
    return {
        "representation": representation,
        "predicted_stage_day": predicted,
        "predicted_stage_class": predicted_class,
        "absolute_stage_error": abs(predicted - true_stage),
        "exact_stage_correct": predicted_class == true_stage,
        "within_adjacent_stage": abs(predicted_index - true_index) <= 1,
        "minimum_shared_features": int(common.min()),
        "median_shared_features": float(np.median(common)),
        "neighbors": int(len(use)),
    }


def _benchmark_erythroid(cfg: dict) -> Path:
    settings, _, root = _paths(cfg)
    erythroid_settings = settings["erythroid"]
    label = erythroid_settings.get("run_label", "formal")
    out = root / "benchmark" / "erythroid" / label
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "run_config.json", cfg)
    _atomic_json(out / "status.json", {"status": "preparing", "started": time.time()})
    distance, u, s, meta, genes = _erythroid_inputs(cfg, out)
    samples = sorted(meta["sample"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    limit = erythroid_settings.get("max_folds")
    samples_to_run = samples if limit is None else samples[: int(limit)]
    all_sample_profiles = {
        "raw_distance": _sample_centroids(distance, meta["sample"].to_numpy(), samples),
    }
    direct = u.astype(np.float32) - s.astype(np.float32)
    direct[(u == 0) | (s == 0)] = np.nan
    all_sample_profiles["direct_bin_difference"] = _sample_centroids(direct, meta["sample"].to_numpy(), samples)
    all_sample_profiles["absolute_bin_difference"] = _sample_centroids(np.abs(direct), meta["sample"].to_numpy(), samples)
    predictions = []
    fold_audit = []
    stage_by_sample = meta.groupby("sample").stage.first().to_dict()
    if (meta.groupby("sample").stage.nunique() != 1).any():
        raise ValueError("An Erythroid sample has multiple stage labels")
    for heldout in samples_to_run:
        fold_path = out / "folds" / f"sample_{heldout}.csv"
        if fold_path.exists():
            predictions.extend(pd.read_csv(fold_path).to_dict("records"))
            continue
        started = time.monotonic()
        train_cells = np.flatnonzero(meta["sample"].ne(heldout))
        fitted = fit_distance_calibrator(distance, u, s, train_cells, int(erythroid_settings.get("min_gene_observations", 50)))
        calibrated = apply_distance_calibrator(distance, u, s, fitted)
        calibrated_profiles = _sample_centroids(calibrated, meta["sample"].to_numpy(), samples)
        heldout_position = samples.index(heldout)
        train_positions = np.asarray([i for i, sample in enumerate(samples) if sample != heldout], dtype=np.int64)
        train_stage = np.asarray([_stage_day(stage_by_sample[samples[i]]) for i in train_positions])
        true_stage = _stage_day(stage_by_sample[heldout])
        panel = fitted.eligible
        rows = []
        for name, profiles in {**all_sample_profiles, "calibrated_distance": calibrated_profiles}.items():
            for k in erythroid_settings.get("neighbors", [5, 15]):
                row = _erythroid_fold_prediction(name, profiles[train_positions], profiles[heldout_position], train_stage, true_stage, panel, int(k))
                row.update({"heldout_sample": heldout, "true_stage": stage_by_sample[heldout], "true_stage_day": true_stage, "eligible_genes": int(panel.sum())})
                rows.append(row)
        fold = pd.DataFrame(rows)
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        fold.to_csv(fold_path, index=False)
        predictions.extend(rows)
        fold_audit.append({"heldout_sample": heldout, "train_cells": len(train_cells), "eligible_genes": int(panel.sum()), "seconds": time.monotonic() - started})
        pd.DataFrame(fold_audit).to_csv(out / "fold_resource_audit.csv", index=False)
        _atomic_json(out / "status.json", {"status": "running", "folds_complete": len(set(x["heldout_sample"] for x in predictions)), "folds_requested": len(samples_to_run)})
    prediction = pd.DataFrame(predictions)
    prediction.to_csv(out / "sample_predictions.csv", index=False)
    summary = []
    for (representation, k), part in prediction.groupby(["representation", "neighbors"]):
        rho = spearmanr(part.true_stage_day, part.predicted_stage_day).statistic if len(part) >= 3 else np.nan
        summary.append({
            "representation": representation, "k": int(k), "samples": len(part),
            "stage_spearman": float(rho) if np.isfinite(rho) else np.nan,
            "mean_absolute_stage_error": float(part.absolute_stage_error.mean()),
            "exact_stage_accuracy": float(part.exact_stage_correct.mean()),
            "within_adjacent_stage_accuracy": float(part.within_adjacent_stage.mean()),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "sample_level_metrics.csv", index=False)
    bootstrap_rows = []
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    for k in sorted(prediction.neighbors.unique()):
        wide = prediction[prediction.neighbors.eq(k)].pivot(index="heldout_sample", columns="representation")
        if "calibrated_distance" not in wide["predicted_stage_day"]:
            continue
        samples_for_bootstrap = wide.index.to_numpy()
        for baseline in ("raw_distance", "direct_bin_difference", "absolute_bin_difference"):
            if baseline not in wide["predicted_stage_day"]:
                continue
            values = []
            for _ in range(1000):
                draw = rng.choice(samples_for_bootstrap, size=len(samples_for_bootstrap), replace=True)
                part = prediction[prediction.neighbors.eq(k)].set_index(["heldout_sample", "representation"])
                truth = np.asarray([part.loc[(sample, "calibrated_distance"), "true_stage_day"] for sample in draw])
                calibrated_prediction = np.asarray([part.loc[(sample, "calibrated_distance"), "predicted_stage_day"] for sample in draw])
                baseline_prediction = np.asarray([part.loc[(sample, baseline), "predicted_stage_day"] for sample in draw])
                rho_cal = spearmanr(truth, calibrated_prediction).statistic
                rho_base = spearmanr(truth, baseline_prediction).statistic
                values.append((rho_cal - rho_base, np.mean(np.abs(baseline_prediction - truth)) - np.mean(np.abs(calibrated_prediction - truth))))
            values = np.asarray(values)
            for column, metric in enumerate(("delta_stage_spearman", "mae_reduction")):
                bootstrap_rows.append({
                    "k": int(k), "comparison": f"calibrated_distance - {baseline}", "metric": metric,
                    "mean": float(np.nanmean(values[:, column])),
                    "ci_low": float(np.nanpercentile(values[:, column], 2.5)),
                    "ci_high": float(np.nanpercentile(values[:, column], 97.5)),
                    "samples": len(samples_for_bootstrap), "bootstrap_repeats": 1000,
                })
    pd.DataFrame(bootstrap_rows).to_csv(out / "sample_bootstrap_comparisons.csv", index=False)
    _plot_erythroid(out, summary)
    primary = summary[summary.k.eq(15)]
    sensitivity = summary[summary.k.eq(5)]
    report = [
        "# Erythroid 跨 sample 发育阶段评估",
        "",
        "27 个 biological samples 逐一留出；每个测试 sample 在其余 26 个 sample 上拟合校准与标准化。细胞没有被当作独立生物学重复。",
        "",
        "## 预注册主设置：15-NN",
        "",
        _markdown(primary),
        "",
        "## 邻域灵敏度：5-NN",
        "",
        _markdown(sensitivity),
        "",
        "## 结论边界",
        "",
        "15-NN 下 direct/absolute bin difference 明显优于 raw 与 calibrated distance；5-NN 下 raw distance 的 stage Spearman 较高，但 calibrated distance 仍下降。"
        "这支持 distance 含有局部发育邻域信息，同时表明其结果对邻居尺度敏感，且当前条件校准没有跨数据改善。",
        "",
        "Bootstrap 以 sample 为重采样单位；预训练语料与本数据的重叠状态未知。",
    ]
    (out / "REPORT_CN.md").write_text("\n".join(report) + "\n")
    status = {
        "status": "complete_smoke" if limit is not None else "complete",
        "folds_complete": len(samples_to_run), "total_samples": len(samples), "evaluation_unit": "biological sample",
        "cell_labels_used_as_independent_replicates": False, "pretraining_corpus_overlap": "unknown",
    }
    _atomic_json(out / "status.json", status)
    return out


def _plot_erythroid(out: Path, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use = summary[summary.k.eq(15)] if (summary.k == 15).any() else summary
    if not len(use):
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    for ax, key, title, higher in zip(
        axes,
        ("stage_spearman", "mean_absolute_stage_error", "within_adjacent_stage_accuracy"),
        ("Stage Spearman", "Mean absolute stage error", "Within-adjacent accuracy"),
        (True, False, True),
    ):
        ax.bar(range(len(use)), use[key], color="#3977B5", alpha=0.82)
        ax.set_xticks(range(len(use)), use.representation.str.replace("_", "\n"), rotation=20, ha="right")
        ax.set_title(title + (" (higher better)" if higher else " (lower better)"))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Erythroid leave-one-sample-out developmental-stage evaluation")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"01_sample_level_stage_benchmark.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    neighbor_values = sorted(summary.k.unique())
    representations = summary.representation.unique().tolist()
    fig, axes = plt.subplots(len(neighbor_values), 3, figsize=(14, 4.2 * len(neighbor_values)), squeeze=False)
    for row, k in enumerate(neighbor_values):
        part = summary[summary.k.eq(k)].set_index("representation").reindex(representations)
        for ax, key, title, higher in zip(
            axes[row],
            ("stage_spearman", "mean_absolute_stage_error", "within_adjacent_stage_accuracy"),
            ("Stage Spearman", "Mean absolute stage error", "Within-adjacent accuracy"),
            (True, False, True),
        ):
            ax.bar(range(len(part)), part[key], color="#3977B5", alpha=0.82)
            ax.set_xticks(range(len(part)), [x.replace("_", "\n") for x in part.index], rotation=20, ha="right")
            ax.set_title(f"{title}; k={int(k)}" + (" (higher better)" if higher else " (lower better)"))
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Erythroid biological-sample LOSO: fixed neighbor sensitivity")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"02_neighbor_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def benchmark(cfg: dict) -> Path:
    """Run configured datasets without silently turning cells into replicates."""
    settings, _, root = _paths(cfg)
    completed = []
    if settings.get("run_forebrain", True):
        completed.append(str(_benchmark_forebrain(cfg)))
    if settings.get("run_erythroid", False):
        completed.append(str(_benchmark_erythroid(cfg)))
    _atomic_json(root / "benchmark" / "status.json", {"status": "complete", "completed": completed})
    return root / "benchmark"


def _plot_benchmark(out: Path, metrics: pd.DataFrame, incremental: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    primary = metrics[metrics.k.eq(15)].copy()
    order = primary.sort_values("balanced_accuracy", ascending=False).representation.tolist()
    labels = [name.replace("_", "\n") for name in order]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, title in zip(axes, ("balanced_accuracy", "macro_f1", "macro_auprc"), ("Balanced accuracy", "Macro-F1", "Macro-AUPRC")):
        values = primary.set_index("representation").loc[order, key]
        ax.bar(range(len(order)), values, color="#3977B5", alpha=0.82)
        ax.set_xticks(range(len(order)), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(max(0, values.min() - 0.15), min(1.0, values.max() + 0.06))
    fig.suptitle("Forebrain held-out sample: fixed 15-NN profile protocol")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"01_profile_benchmark.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    order = incremental.model.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, key, title in zip(axes, ("balanced_accuracy", "macro_f1", "macro_auprc"), ("Balanced accuracy", "Macro-F1", "Macro-AUPRC")):
        values = incremental.set_index("model").loc[order, key]
        ax.bar(range(len(order)), values, color=["#7A7A7A", "#2A927B", "#D18B31", "#B65C70"])
        ax.set_xticks(range(len(order)), [v.replace("_", "\n") for v in order], rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Incremental information above the fixed B0 feature set")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"02_incremental_benchmark.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def attribution(cfg: dict) -> Path:
    """Aggregate the existing layerwise run into the ICLR core audit."""
    settings, _, root = _paths(cfg)
    if settings.get("attribution_mode") == "erythroid_context":
        return _erythroid_context_attribution(cfg)
    if settings.get("run_full_attribution", False):
        return _attribution_full_profiles(cfg)
    source = Path(settings["layerwise_metrics_path"])
    out = root / "attribution"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    required = {"model", "layer", "distance_median", "paired_delta_standardized"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Layerwise table is missing {sorted(required - set(frame.columns))}")
    frame.to_csv(out / "layerwise_metrics.csv", index=False)
    summary = frame.groupby(["model", "layer"], as_index=False).agg(
        distance_median=("distance_median", "mean"),
        separation=("paired_delta_standardized", "mean"),
        cell_type_cross_sample_knn_ba=("cell_type_cross_sample_knn_ba", "mean"),
        direct_abs_bin_delta_spearman=("direct_abs_bin_delta_spearman", "mean"),
    )
    summary.to_csv(out / "layerwise_summary.csv", index=False)
    _plot_attribution(out, summary)
    cells = int(settings.get("existing_layerwise_cells", 128))
    status = {
        "status": "complete_existing_scope",
        "cells": cells,
        "planned_cells": int(settings.get("attribution_cells", 512)),
        "full_512_cell_rerun_required": cells < int(settings.get("attribution_cells", 512)),
        "source": str(source),
        "interpretation_boundary": "Existing layerwise run is attribution evidence, not a full 512-cell rerun.",
    }
    _atomic_json(out / "status.json", status)
    return out


def _erythroid_context_attribution(cfg: dict) -> Path:
    """Replace only background context while preserving target gene IDs/bins."""
    import anndata as ad
    import torch
    from .ablations import resource_preflight
    from .full_context_ablation import construct
    from .joint_reconstruction import FullTokenDataset
    from .semantics import collate_cells, cosine_distance, load_frozen_encoder, select_target_genes, stable_rng, target_positions

    settings, _, root = _paths(cfg)
    erythroid = settings["erythroid"]
    out = root / "context_attribution" / "erythroid"
    out.mkdir(parents=True, exist_ok=True)
    data = FullTokenDataset(Path(erythroid["token_path"]), Path(cfg["vocab"]))
    adata = ad.read_h5ad(Path(erythroid["h5ad_path"]), backed="r")
    if not np.array_equal(data.cells, adata.obs_names.astype(str)):
        raise ValueError("Erythroid token and metadata cell order differs")
    meta = pd.DataFrame({
        "cell_id": data.cells,
        "sample": adata.obs["sample"].astype(str).to_numpy(),
        "stage": adata.obs["stage"].astype(str).to_numpy(),
        "cell_type": adata.obs["celltype"].astype(str).to_numpy(),
    })
    levels = sorted(meta.stage.unique(), key=_stage_day)
    stage_index = {stage: i for i, stage in enumerate(levels)}
    seed = int(cfg.get("seed", 42))
    requested = int(settings.get("context_cells", 128))
    order = sorted(range(len(meta)), key=lambda i: hashlib.sha256(f"{seed}:{meta.cell_id.iloc[i]}".encode()).hexdigest())
    chosen = []
    groups = {(stage, cell_type): [i for i in order if meta.stage.iloc[i] == stage and meta.cell_type.iloc[i] == cell_type] for stage in levels for cell_type in sorted(meta.cell_type.unique())}
    while len(chosen) < requested:
        added = False
        for group in sorted(groups):
            if groups[group] and len(chosen) < requested:
                chosen.append(groups[group].pop(0)); added = True
        if not added:
            break
    cells = [data[i] for i in chosen]
    targets, coverage = select_target_genes(cells, int(settings.get("context_target_genes", 128)), seed)
    inverse = {int(g): str(n) for g, n in zip(data.mapping, data.names)}
    pd.DataFrame({"gene_id": targets, "gene_name": [inverse[int(g)] for g in targets], "selected_cell_coverage": [coverage[int(g)] for g in targets]}).to_csv(out / "target_genes.csv", index=False)
    covariates = np.zeros((len(data.cells), 5), dtype=np.float32)
    for i in range(len(data.cells)):
        value = data[i]["value_bins"][0].numpy().reshape(-1, 2)
        positive = value[value > 0]
        covariates[i] = [len(value), (value[:, 0] > 0).mean(), (value[:, 1] > 0).mean(), len(positive), positive.mean() if len(positive) else 0]
    scale = covariates.std(0); scale[scale == 0] = 1
    covariates = (covariates - covariates.mean(0)) / scale

    def donor(source: int, category: str, donor_seed: int):
        delta = np.asarray([abs(stage_index[x] - stage_index[meta.stage.iloc[source]]) for x in meta.stage])
        wanted = {"same": delta == 0, "adjacent": delta == 1, "distant": delta >= 2}[category]
        candidate = np.flatnonzero(
            wanted
            & meta["sample"].ne(meta["sample"].iloc[source]).to_numpy()
            & meta["cell_type"].eq(meta["cell_type"].iloc[source]).to_numpy()
        )
        if not len(candidate):
            return None
        mismatch = ((covariates[candidate] - covariates[source]) ** 2).sum(1)
        nearest = candidate[np.argsort(mismatch, kind="stable")[: min(5, len(candidate))]]
        return int(nearest[stable_rng(donor_seed, meta.cell_id.iloc[source], category).integers(len(nearest))])

    donors = []
    donor_seeds = list(settings.get("donor_seeds", [42, 43, 44, 45, 46]))
    for source in chosen:
        for donor_seed in donor_seeds:
            for category in ("same", "adjacent", "distant"):
                selected = donor(source, category, int(donor_seed))
                donors.append({"source_index": source, "source_cell_id": meta.cell_id.iloc[source], "category": category, "donor_seed": donor_seed, "donor_index": selected, "donor_cell_id": meta.cell_id.iloc[selected] if selected is not None else None})
    donor_table = pd.DataFrame(donors)
    donor_table.to_csv(out / "donor_manifest.csv", index=False)
    device = torch.device(cfg.get("device", "cpu"))
    if str(device) == "cuda:1":
        raise ValueError("GPU 1 is excluded")
    if device.type == "cuda":
        passed, trace, reason = resource_preflight(
            int(device.index), samples=int(settings.get("preflight_samples", 60)), interval_seconds=float(settings.get("preflight_interval_seconds", 5)),
            minimum_free_mb=float(settings.get("minimum_free_mb", 60_000)), maximum_mean_utilization=float(settings.get("maximum_mean_utilization", 10)),
        )
        trace.to_csv(out / "resource_preflight.csv", index=False)
        if not passed:
            _atomic_json(out / "status.json", {"status": "blocked_resource_preflight", "reason": reason})
            raise RuntimeError(reason)
    model_specs = [("epoch_11", Path(cfg["checkpoint"]), None)] + [(f"random_{int(x)}", Path(cfg["checkpoint"]), int(x)) for x in settings.get("random_seeds", [42, 43, 44])]
    all_rows = []
    batch_size = int(settings.get("context_batch_size", 4))
    for model_label, checkpoint, random_seed in model_specs:
        chunks = out / "chunks" / model_label
        chunks.mkdir(parents=True, exist_ok=True)
        model = load_frozen_encoder(checkpoint, device, random_seed)
        for source in chosen:
            chunk_path = chunks / f"{source:05d}.parquet"
            if chunk_path.exists():
                all_rows.append(pd.read_parquet(chunk_path))
                continue
            source_cell = data[source]
            positions = target_positions(source_cell, targets)
            if not len(positions):
                continue
            own_targets = source_cell["gene_ids"][0, 2 * positions].numpy()
            inputs = [("full", -1, None, construct(source_cell, own_targets, "full"))]
            part = donor_table[donor_table.source_index.eq(source)].dropna(subset=["donor_index"])
            for row in part.itertuples():
                context = construct(source_cell, own_targets, "matched_context", data[int(row.donor_index)], int(row.donor_seed))
                inputs.append((str(row.category), int(row.donor_seed), str(row.donor_cell_id), context))
            rows = []
            for start in range(0, len(inputs), batch_size):
                block = inputs[start : start + batch_size]
                g, v, s, lengths = collate_cells([x[3] for x in block], device)
                with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hidden = model._encode(g, v, s)
                for r, (category, donor_seed, donor_cell_id, cell) in enumerate(block):
                    n = len(own_targets)
                    values = cosine_distance(hidden[r, : 2 * n : 2], hidden[r, 1 : 2 * n : 2]).float().cpu().numpy()
                    original = source_cell["value_bins"][0].numpy().reshape(-1, 2)[positions]
                    current = cell["value_bins"][0, : 2 * n].numpy().reshape(-1, 2)
                    if not np.array_equal(original, current):
                        raise AssertionError("Context replacement changed target bins")
                    if not np.array_equal(cell["gene_ids"][0, : 2 * n : 2].numpy(), own_targets):
                        raise AssertionError("Context replacement changed target gene IDs")
                    for gene, value in zip(own_targets, values):
                        rows.append({"model": model_label, "source_index": source, "source_cell_id": meta.cell_id.iloc[source], "source_stage": meta.stage.iloc[source], "source_cell_type": meta.cell_type.iloc[source], "category": category, "donor_seed": donor_seed, "donor_cell_id": donor_cell_id, "gene_id": int(gene), "distance": float(value)})
            frame = pd.DataFrame(rows)
            baseline = frame[frame.category.eq("full")].set_index("gene_id").distance
            frame["delta_vs_full"] = frame.distance - frame.gene_id.map(baseline)
            frame.to_parquet(chunk_path, index=False)
            all_rows.append(frame)
            _atomic_json(out / "status.json", {"status": "running", "model": model_label, "source_cells_complete": len(list(chunks.glob("*.parquet"))), "source_cells": len(chosen)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    long = pd.concat(all_rows, ignore_index=True)
    long.to_parquet(out / "context_distance_long.parquet", index=False)
    summary = long[long.category.ne("full")].groupby(["model", "category"], as_index=False).agg(
        mean_delta=("delta_vs_full", "mean"), median_delta=("delta_vs_full", "median"),
        cells=("source_cell_id", "nunique"), observations=("delta_vs_full", "size"),
    )
    summary.to_csv(out / "context_summary.csv", index=False)
    _plot_context_attribution(out, summary)
    epoch = summary.set_index(["model", "category"]).mean_delta
    ordering = bool(epoch.get(("epoch_11", "distant"), np.nan) > epoch.get(("epoch_11", "adjacent"), np.nan) > epoch.get(("epoch_11", "same"), np.nan))
    _atomic_json(out / "status.json", {
        "status": "complete", "selected_cells": len(chosen), "target_genes": len(targets), "donor_seeds": donor_seeds,
        "target_ids_and_bins_unchanged": True, "distant_gt_adjacent_gt_same_epoch11": ordering,
        "missing_donor_combinations": int(donor_table.donor_index.isna().sum()), "pretraining_corpus_overlap": "unknown",
    })
    return out


def _plot_context_attribution(out: Path, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ["same", "adjacent", "distant"]
    models = summary.model.unique().tolist()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.8 / len(models)
    x = np.arange(len(categories))
    for j, model in enumerate(models):
        values = summary[summary.model.eq(model)].set_index("category").reindex(categories).mean_delta
        ax.bar(x + (j - (len(models) - 1) / 2) * width, values, width=width, label=model)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(x, ["Same stage", "Adjacent stage", "Distant stage"])
    ax.set_ylabel("Target-pair distance change versus Full")
    ax.set_title("Erythroid controlled stage-context replacement")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"01_stage_context_effect.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _attribution_models(cfg: dict):
    from .semantics_workflow import _checkpoint_path

    settings = cfg["iclr_distance"]
    epochs = settings.get("attribution_checkpoint_epochs", [0, 3, 6, 9, 11])
    result = [(f"epoch_{int(epoch)}", _checkpoint_path(cfg, int(epoch)), None) for epoch in epochs]
    result.extend((f"random_{int(seed)}", Path(cfg["checkpoint"]), int(seed)) for seed in settings.get("random_seeds", [42, 43, 44]))
    return result


def _attribution_full_profiles(cfg: dict) -> Path:
    import torch
    from .ablations import resource_preflight
    from .semantics import collate_cells, load_frozen_encoder, paired_distance
    from .joint_reconstruction import FullTokenDataset

    settings, source, root = _paths(cfg)
    out = root / "attribution_full_profiles"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cpu"))
    if str(device) == "cuda:1":
        raise ValueError("GPU 1 is excluded")
    if device.type == "cuda":
        passed, trace, reason = resource_preflight(
            int(device.index),
            samples=int(settings.get("preflight_samples", 60)),
            interval_seconds=float(settings.get("preflight_interval_seconds", 5)),
            minimum_free_mb=float(settings.get("minimum_free_mb", 60_000)),
            maximum_mean_utilization=float(settings.get("maximum_mean_utilization", 10)),
        )
        trace.to_csv(out / "resource_preflight.csv", index=False)
        if not passed:
            _atomic_json(out / "status.json", {"status": "blocked_resource_preflight", "reason": reason})
            raise RuntimeError(reason)
    cells, genes = pd.read_csv(source / "cells.csv"), pd.read_csv(source / "input_genes.csv")
    data = FullTokenDataset(Path(cfg["data"]["token_path"]), Path(cfg["vocab"]))
    if not np.array_equal(data.cells, cells.cell_id.astype(str).to_numpy()):
        raise ValueError("Token and cell metadata order differs")
    if not np.array_equal(data.mapping, genes.vocab_id.to_numpy()):
        raise ValueError("Token and distance gene order differs")
    reference = np.flatnonzero(cells["sample"].eq(settings.get("reference_sample", "10X_17_029")))
    heldout = np.flatnonzero(cells["sample"].eq(settings.get("heldout_sample", "10X_17_028")))
    reference_distance = np.load(source / "zero_shot" / "cell_gene_distance.npy", mmap_mode="r")
    panel = np.isfinite(reference_distance[reference]).sum(0) >= int(settings.get("min_gene_observations", 50))
    batch_size = int(settings.get("attribution_batch_size", 8))
    metrics = []
    lookup = {int(g): j for j, g in enumerate(data.mapping)}
    for label, checkpoint, random_seed in _attribution_models(cfg):
        final_path = out / f"{label}_final_distance.npy"
        partial = out / f"{label}_final_distance.partial.npy"
        progress_path = out / f"{label}_progress.json"
        start_cell = 0
        if final_path.exists():
            matrix = np.load(final_path, mmap_mode="r")
        else:
            if partial.exists() and progress_path.exists():
                matrix = np.lib.format.open_memmap(partial, mode="r+", dtype=np.float32, shape=(len(cells), len(genes)))
                start_cell = int(json.loads(progress_path.read_text()).get("cells_complete", 0))
            else:
                matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(cells), len(genes)))
                matrix[:] = np.nan
            model = load_frozen_encoder(checkpoint, device, random_seed)
            started = time.monotonic()
            for start in range(start_cell, len(cells), batch_size):
                stop = min(start + batch_size, len(cells))
                items = [data[i] for i in range(start, stop)]
                g, v, s, lengths = collate_cells(items, device)
                with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hidden = model._encode(g, v, s)
                for row, (item, length) in enumerate(zip(items, lengths)):
                    distance, valid = paired_distance(hidden[row], item["value_bins"][0], int(length))
                    gene = item["gene_ids"][0, ::2].numpy()
                    cols = np.fromiter((lookup[int(x)] for x in gene), dtype=np.int64, count=len(gene))
                    use = valid.cpu().numpy()
                    matrix[start + row, cols[use]] = distance[valid].float().cpu().numpy()
                matrix.flush()
                _atomic_json(progress_path, {"status": "running", "model": label, "cells_complete": stop, "cells": len(cells), "elapsed_seconds": time.monotonic() - started})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            del matrix
            partial.replace(final_path)
            _atomic_json(progress_path, {"status": "complete", "model": label, "cells_complete": len(cells), "cells": len(cells)})
            matrix = np.load(final_path, mmap_mode="r")
        if not np.array_equal(np.isfinite(matrix), np.isfinite(reference_distance)):
            raise ValueError(f"{label} missing mask differs from epoch-11 reference")
        d_train, d_test, _, common = _profile_distance_matrices(matrix, reference, heldout, panel, "rms")
        for k in settings.get("neighbors", [5, 15, 50]):
            result, _ = _knn_from_distances(d_train, d_test, common, cells, reference, heldout, int(panel.sum()), int(k))
            result.update({"model": label, "k": int(k), "random_seed": random_seed})
            metrics.append(result)
    frame = pd.DataFrame(metrics)
    frame.to_csv(out / "full_profile_metrics.csv", index=False)
    _plot_full_attribution(out, frame)
    _atomic_json(out / "status.json", {
        "status": "complete", "cells": len(cells), "genes": len(genes), "selected_genes": int(panel.sum()),
        "models": frame.model.unique().tolist(), "missing_mask_identical": True, "test_labels_used_for_fit": False,
    })
    return out


def _plot_full_attribution(out: Path, metrics: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use = metrics[metrics.k.eq(15)].copy()
    order = use.sort_values("balanced_accuracy", ascending=False).model.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, title in zip(axes, ("balanced_accuracy", "macro_f1", "macro_auprc"), ("Balanced accuracy", "Macro-F1", "Macro-AUPRC")):
        values = use.set_index("model").loc[order, key]
        colors = ["#B65C70" if name.startswith("random") else "#3977B5" for name in order]
        ax.bar(range(len(order)), values, color=colors, alpha=0.82)
        ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen-checkpoint contribution on the identical all-cell distance profile")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"01_full_profile_checkpoint_attribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_attribution(out: Path, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, part in use.groupby("model"):
        part = part.sort_values("layer")
        ax.plot(part.layer, part.separation, marker="o", lw=1.5, label=model)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Correct-vs-null standardized separation")
    ax.set_title("Layer-wise origin of paired U/S geometry")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"01_layerwise_attribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def align_pareto(cfg: dict) -> Path:
    settings, _, root = _paths(cfg)
    if settings.get("align_pareto_evaluate", False):
        return _evaluate_align_objectives(cfg)
    run = Path(settings["align_run_dir"])
    fit = run / "align-fit"
    out = root / "align_pareto"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(fit.glob("epoch_*_validation.json")):
        epoch = int(path.stem.split("_")[1])
        validation = json.loads(path.read_text())
        qc = json.loads((fit / f"epoch_{epoch:03d}_qc.json").read_text())
        eligibility = json.loads((fit / f"epoch_{epoch:03d}_eligibility.json").read_text())
        train = pd.read_csv(fit / f"epoch_{epoch:03d}_train.csv")
        row = {"epoch": epoch, **{f"validation_{k}": v for k, v in validation.items()}, **{f"qc_{k}": v for k, v in qc.items() if np.isscalar(v)}}
        for key in ("loss_adv", "loss_disc", "loss_rec", "loss_token_weighted", "accuracy_u", "accuracy_s"):
            row[f"train_{key}"] = float(train[key].mean())
        row["checkpoint_eligible_historical"] = bool(eligibility.get("checkpoint_eligible", False))
        row["knn_overlap"] = eligibility.get("knn_overlap", np.nan)
        row["distance_mean_spearman"] = eligibility.get("distance_mean_spearman", np.nan)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("epoch")
    external = Path(settings["align_external_metrics_path"])
    if external.exists():
        performance = pd.read_csv(external)
        mapping = {"epoch_029": 29, "epoch_049": 49}
        for model, epoch in mapping.items():
            part = performance[performance.model.eq(model)]
            if len(part):
                for key in ("balanced_accuracy", "macro_f1", "macro_auprc"):
                    table.loc[table.epoch.eq(epoch), f"heldout_{key}"] = float(part.iloc[0][key])
        zero = performance[performance.model.eq("zero_shot")]
        if len(zero):
            pd.DataFrame([{**{"model": "zero_shot", "epoch": -1}, **zero.iloc[0].to_dict()}]).to_csv(out / "zero_shot_external_reference.csv", index=False)
    # Pareto is descriptive: maximize held-out BA and semantic effect, minimize CE.
    comparable = table.dropna(subset=["heldout_balanced_accuracy", "qc_matched_shuffle_effect", "validation_loss_rec"]).copy()
    comparable["pareto_non_dominated"] = True
    for i, row in comparable.iterrows():
        dominated = comparable[
            (comparable.heldout_balanced_accuracy >= row.heldout_balanced_accuracy)
            & (comparable.qc_matched_shuffle_effect >= row.qc_matched_shuffle_effect)
            & (comparable.validation_loss_rec <= row.validation_loss_rec)
            & (
                (comparable.heldout_balanced_accuracy > row.heldout_balanced_accuracy)
                | (comparable.qc_matched_shuffle_effect > row.qc_matched_shuffle_effect)
                | (comparable.validation_loss_rec < row.validation_loss_rec)
            )
        ]
        comparable.loc[i, "pareto_non_dominated"] = dominated.empty
    table["pareto_non_dominated_measured"] = table.epoch.map(comparable.set_index("epoch").pareto_non_dominated if len(comparable) else {})
    table.to_csv(out / "checkpoint_metrics.csv", index=False)
    comparable.to_csv(out / "measured_pareto_points.csv", index=False)
    _plot_align_pareto(out, table)
    status = {
        "status": "complete_existing_objective",
        "epochs": int(len(table)),
        "external_performance_epochs": table.loc[table.heldout_balanced_accuracy.notna(), "epoch"].tolist() if "heldout_balanced_accuracy" in table else [],
        "objective": "Joint adversarial=0.1, reconstruction=0.3, seed=42",
        "missing_required_objectives": ["adversarial_only", "reconstruction_only", "joint_seed43", "joint_seed44"],
        "warning": "A 50-epoch trajectory is not the planned objective/seed ablation.",
    }
    _atomic_json(out / "status.json", status)
    return out


def _evaluate_align_objectives(cfg: dict) -> Path:
    import torch
    from .ablations import resource_preflight
    from .joint_batch import encoded_cells
    from .joint_reconstruction import FullTokenDataset, JointModel
    from .runtime import cosine_distance

    settings, source, root = _paths(cfg)
    out = root / "align_pareto" / "formal_objective_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cpu"))
    if str(device) == "cuda:1":
        raise ValueError("GPU 1 is excluded")
    if device.type == "cuda":
        passed, trace, reason = resource_preflight(
            int(device.index), samples=int(settings.get("preflight_samples", 60)), interval_seconds=float(settings.get("preflight_interval_seconds", 5)),
            minimum_free_mb=float(settings.get("minimum_free_mb", 60_000)), maximum_mean_utilization=float(settings.get("maximum_mean_utilization", 10)),
        )
        trace.to_csv(out / "resource_preflight.csv", index=False)
        if not passed:
            _atomic_json(out / "status.json", {"status": "blocked_resource_preflight", "reason": reason})
            raise RuntimeError(reason)
    cells, genes = pd.read_csv(source / "cells.csv"), pd.read_csv(source / "input_genes.csv")
    data = FullTokenDataset(Path(cfg["data"]["token_path"]), Path(cfg["vocab"]))
    if not np.array_equal(data.cells, cells.cell_id.astype(str).to_numpy()) or not np.array_equal(data.mapping, genes.vocab_id.to_numpy()):
        raise ValueError("Align evaluation input order differs from Zero-shot")
    bins = np.load(root / "calibration" / "forebrain" / "input_bins.npz")
    u, s = bins["u"], bins["s"]
    reference = np.flatnonzero(cells["sample"].eq(settings.get("reference_sample", "10X_17_029")))
    heldout = np.flatnonzero(cells["sample"].eq(settings.get("heldout_sample", "10X_17_028")))
    zero = np.load(source / "zero_shot" / "cell_gene_distance.npy", mmap_mode="r")
    panel = np.isfinite(zero[reference]).sum(0) >= int(settings.get("min_gene_observations", 50))
    lookup = {int(g): i for i, g in enumerate(data.mapping)}
    batch_size = int(settings.get("align_evaluation_batch_size", 32))
    epochs = [int(x) for x in settings.get("align_evaluation_epochs", [2, 29])]
    rows = []
    run_specs = settings["align_objective_runs"]
    zero_calibrated = np.load(root / "calibration" / "forebrain" / "calibrated_distance.npy", mmap_mode="r")
    joint_reference = Path(run_specs["joint_seed42"]["path"]) / "align-fit"
    raw_validation = json.loads((joint_reference / "raw_validation.json").read_text())
    raw_qc = json.loads((joint_reference / "raw_qc.json").read_text())
    for representation, values in (("aligned_raw_distance", zero), ("aligned_calibrated_distance", zero_calibrated)):
        d_train, d_test, _, common = _profile_distance_matrices(values, reference, heldout, panel, "rms")
        result, _ = _knn_from_distances(d_train, d_test, common, cells, reference, heldout, int(panel.sum()), 15)
        result.update({
            "run": "zero_shot", "objective": "zero_shot", "seed": 42, "epoch": -1, "representation": representation,
            "validation_loss_rec": raw_validation["loss_rec"], "validation_accuracy_u": raw_validation["accuracy_u"],
            "validation_accuracy_s": raw_validation["accuracy_s"], "effective_rank": raw_qc["effective_rank"],
            "distance_iqr": raw_qc["distance_iqr"], "matched_shuffle_effect": raw_qc["matched_shuffle_effect"],
            "modality_probe_ba": raw_qc["probe_balanced_accuracy"], "modality_probe_auroc": raw_qc["probe_auroc"],
        })
        rows.append(result)
    for name, spec in run_specs.items():
        run = Path(spec["path"])
        status_path = run / "align-fit" / "status.json"
        status = json.loads(status_path.read_text())
        if status.get("status") not in {"formal_fit_complete", "formal_fit_no_eligible_checkpoint"}:
            raise RuntimeError(f"Align run is incomplete: {name}: {status.get('status')}")
        for epoch in epochs:
            checkpoint = run / "align-fit" / f"epoch_{epoch:03d}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            dest = out / name / f"epoch_{epoch:03d}"
            dest.mkdir(parents=True, exist_ok=True)
            final = dest / "cell_gene_distance.npy"
            partial = dest / "cell_gene_distance.partial.npy"
            progress = dest / "progress.json"
            start_cell = 0
            if final.exists():
                matrix = np.load(final, mmap_mode="r")
            else:
                if partial.exists() and progress.exists():
                    matrix = np.lib.format.open_memmap(partial, mode="r+", dtype=np.float32, shape=zero.shape)
                    start_cell = int(json.loads(progress.read_text()).get("cells_complete", 0))
                else:
                    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=zero.shape)
                    matrix[:] = np.nan
                model = JointModel.from_checkpoint(Path(cfg["checkpoint"])).to(device).eval()
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict(payload["model"], strict=True)
                model.requires_grad_(False)
                del payload
                for offset, (cell, _, z) in enumerate(encoded_cells(model, data, list(range(start_cell, len(cells))), device, batch_size), start=start_cell):
                    value = cell["value_bins"][0]
                    gene = cell["gene_ids"][0]
                    paired = (value[::2] > 0) & (value[1::2] > 0)
                    distance = cosine_distance(z[0, ::2][paired], z[0, 1::2][paired]).float().cpu().numpy()
                    cols = np.fromiter((lookup[int(x)] for x in gene[::2][paired].cpu().numpy()), dtype=np.int64, count=int(paired.sum()))
                    matrix[offset, cols] = distance
                    if (offset + 1) % 128 == 0 or offset + 1 == len(cells):
                        matrix.flush()
                        _atomic_json(progress, {"status": "running", "cells_complete": offset + 1, "cells": len(cells)})
                del model, matrix
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                partial.replace(final)
                _atomic_json(progress, {"status": "complete", "cells_complete": len(cells), "cells": len(cells)})
                matrix = np.load(final, mmap_mode="r")
            if not np.array_equal(np.isfinite(matrix), np.isfinite(zero)):
                raise ValueError(f"Align missing mask differs: {name} epoch {epoch}")
            fitted = fit_distance_calibrator(matrix, u, s, reference, int(settings.get("min_gene_observations", 50)))
            calibrated = apply_distance_calibrator(matrix, u, s, fitted)
            for representation, values in (("aligned_raw_distance", matrix), ("aligned_calibrated_distance", calibrated)):
                d_train, d_test, _, common = _profile_distance_matrices(values, reference, heldout, panel, "rms")
                result, _ = _knn_from_distances(d_train, d_test, common, cells, reference, heldout, int(panel.sum()), 15)
                result.update({"run": name, "objective": spec["objective"], "seed": int(spec["seed"]), "epoch": epoch, "representation": representation})
                rows.append(result)
            validation = json.loads((run / "align-fit" / f"epoch_{epoch:03d}_validation.json").read_text())
            qc = json.loads((run / "align-fit" / f"epoch_{epoch:03d}_qc.json").read_text())
            for row in rows[-2:]:
                row.update({
                    "validation_loss_rec": validation["loss_rec"], "validation_accuracy_u": validation["accuracy_u"], "validation_accuracy_s": validation["accuracy_s"],
                    "effective_rank": qc["effective_rank"], "distance_iqr": qc["distance_iqr"], "matched_shuffle_effect": qc["matched_shuffle_effect"],
                    "modality_probe_ba": qc["probe_balanced_accuracy"], "modality_probe_auroc": qc["probe_auroc"],
                })
            pd.DataFrame(rows).to_csv(out / "objective_metrics.partial.csv", index=False)
            _atomic_json(out / "status.json", {"status": "running", "completed_run": name, "completed_epoch": epoch, "rows": len(rows)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "objective_metrics.csv", index=False)
    _plot_objective_pareto(out, metrics)
    _atomic_json(out / "status.json", {"status": "complete", "runs": list(run_specs), "epochs": epochs, "rows": len(metrics), "test_labels_used_for_checkpoint_selection": False})
    return out


def _plot_objective_pareto(out: Path, metrics: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use = metrics[metrics.representation.eq("aligned_calibrated_distance")]
    colors = {"zero_shot": "#555555", "adversarial_only": "#B65C70", "reconstruction_only": "#3977B5", "joint": "#2A927B"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, x, xlabel in zip(axes, ("matched_shuffle_effect", "effective_rank", "validation_loss_rec"), ("Pairing effect", "Effective rank", "Validation reconstruction CE")):
        for objective, part in use.groupby("objective"):
            ax.scatter(part[x], part.balanced_accuracy, color=colors.get(objective, "#777777"), label=objective, s=70, alpha=0.85)
            for row in part.itertuples():
                ax.annotate(f"s{row.seed}/e{row.epoch}", (getattr(row, x), row.balanced_accuracy), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Held-out sample balanced accuracy")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("scUS-Align objective ablation: task performance versus relational preservation")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"01_objective_pareto.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_align_pareto(out: Path, table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("validation_loss_rec", "Validation reconstruction CE", False),
        ("qc_matched_shuffle_effect", "Correct-vs-shuffle effect", True),
        ("qc_effective_rank", "Effective rank", True),
        ("knn_overlap", "Zero-shot kNN overlap", True),
    ]
    for ax, (key, title, _) in zip(axes.flat, panels):
        ax.plot(table.epoch, table[key], lw=1.3, color="#3977B5")
        for epoch, color in ((2, "#2A927B"), (29, "#D18B31"), (49, "#B65C70")):
            part = table[table.epoch.eq(epoch)]
            if len(part):
                ax.scatter(epoch, part.iloc[0][key], color=color, s=45, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Joint fine-tuning epoch")
        ax.grid(alpha=0.2)
    fig.suptitle("Joint scUS-Align trajectory: optimization versus relational preservation")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"01_joint_trajectory.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    measured = table.dropna(subset=["heldout_balanced_accuracy"])
    if len(measured):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        scatter = ax.scatter(measured.qc_matched_shuffle_effect, measured.heldout_balanced_accuracy, c=measured.epoch, s=90, cmap="viridis")
        for row in measured.itertuples():
            ax.annotate(f"e{row.epoch}", (row.qc_matched_shuffle_effect, row.heldout_balanced_accuracy), xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("Correct-vs-shuffle standardized effect")
        ax.set_ylabel("Held-out sample balanced accuracy")
        ax.set_title("Measured task-semantics trade-off")
        fig.colorbar(scatter, ax=ax, label="Epoch")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"02_measured_task_semantics_tradeoff.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    (out / "REPORT_CN.md").write_text(
        "# scUS-Align Pareto 审计\n\n"
        "本目录首先整理现有 Joint(adv=0.1, rec=0.3, seed=42) 的50轮轨迹。"
        "外层留出性能目前只对 Zero-shot、epoch 29和epoch 49完整测量，因此不能从本图声称已经完成objective或多seed消融。\n\n"
        "图中的关键含义是：重建损失、模态混淆、有效秩、配对效应和外层任务性能不是同一个目标；绝对distance变小也不是成功标准。\n"
    )


def validate_iclr_core(cfg: dict) -> dict:
    _, _, root = _paths(cfg)
    checks = {}
    required_stages = (
        "calibration/forebrain",
        "benchmark/forebrain",
        "benchmark/erythroid/formal",
        "attribution_full_512/layerwise/formal",
        "attribution_full_profiles",
        "context_attribution/erythroid",
        "align_pareto/formal_objective_evaluation",
    )
    for stage in required_stages:
        path = root / stage / "status.json"
        checks[stage] = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
    complete = {name: value.get("status", "").startswith("complete") for name, value in checks.items()}
    pending = [name for name, is_complete in complete.items() if not is_complete]
    result = {
        "output_dir": str(root), "checks": checks,
        "completed_required_stages": [name for name, is_complete in complete.items() if is_complete],
        "planned_core_complete": not pending,
        "pending": pending,
        "pretraining_corpus_overlap": "unknown",
    }
    _atomic_json(root / "validation.json", result)
    lines = [
        "# scUS-Distance ICLR 核心实验报告",
        "",
        "定义：`d(c,g) = 1 - cosine(h_U(c,g), h_S(c,g))`。它是 gene-resolved、context-modulated、unsigned paired-modality discrepancy；不是RNA velocity、方向、未来表达或异常概率。",
        "",
        f"当前完整状态：`{str(result['planned_core_complete']).lower()}`。预训练语料重叠：`unknown`。",
        "",
        "## 已完成的固定结果",
        "",
        "- Forebrain（held-out sample，15-NN）：direct bin difference BA 0.9052，calibrated distance 0.8925，raw distance 0.8682；B0+calibrated 没有提高BA。",
        "- Erythroid（27 samples LOSO，主15-NN）：direct bin difference stage Spearman 0.8245，raw distance 0.4992，calibrated distance 0.4501。",
        "- Erythroid 5-NN灵敏度中 raw distance Spearman 0.8362，但校准降至0.6941；这属于邻域尺度敏感的局部信号，不覆盖主设置的负结果。",
        "",
        "## 尚未完成",
        "",
    ]
    lines.extend(f"- `{name}`" for name in pending)
    if not pending:
        lines.append("- 无；所有登记核心阶段均已通过状态检查。")
    lines.extend([
        "",
        "## 当前结论",
        "",
        "distance profile明显优于distance scalar和gene-shuffle，并能表示细胞状态；但现有公平结果不支持它普遍优于直接U−S、pair mask或表达基线，也不支持把raw/calibrated distance称为跨数据异常分数。预训练归因、Erythroid受控context和Align Pareto完成前，不升级为‘learned normal U/S relationships’。",
    ])
    (root / "REPORT_CN.md").write_text("\n".join(lines) + "\n")
    return result

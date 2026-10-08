from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .runtime import provenance, stage_directory, write_json


def _distance_checks(distance: np.ndarray, chunk_size: int = 10_000_000) -> tuple[bool, bool]:
    finite, in_range = True, True
    for start in range(0, len(distance), chunk_size):
        block = np.asarray(distance[start:start + chunk_size])
        finite = finite and bool(np.isfinite(block).all())
        in_range = in_range and bool(((block >= -1e-5) & (block <= 2.00001)).all())
        if not finite and not in_range:
            break
    return finite, in_range


def _arrays_equal(left: np.ndarray, right: np.ndarray, chunk_size: int = 10_000_000) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return all(
        np.array_equal(left[start:start + chunk_size], right[start:start + chunk_size])
        for start in range(0, len(left), chunk_size)
    )


def _effective_rank(embedding: np.ndarray, maximum: int = 50_000) -> int:
    """Return a stable numerical rank from the small feature Gram matrix."""
    sample = np.asarray(embedding[:min(len(embedding), maximum)], np.float64).copy()
    if not len(sample):
        return 0
    sample -= sample.mean(0, keepdims=True)
    singular = np.sqrt(np.maximum(np.linalg.eigvalsh(sample.T @ sample), 0))[::-1]
    return int((singular > singular[0] * 1e-3).sum()) if singular[0] > 0 else 0


def _balanced_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [np.mean(predicted[true == label] == label) for label in np.unique(true)]
    return float(np.mean(recalls))


def _knn_accuracy(
    embedding: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    neighbours: int = 15,
) -> float:
    """Small, dependency-free cosine-kNN used only for validation."""
    train = np.asarray(embedding[train_index], np.float32)
    test = np.asarray(embedding[test_index], np.float32)
    train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    test /= np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
    classes, encoded = np.unique(labels, return_inverse=True)
    train_labels = encoded[train_index]
    predictions = []
    k = min(neighbours, len(train))
    for begin in range(0, len(test), 256):
        similarity = test[begin:begin + 256] @ train.T
        nearest = np.argpartition(similarity, -k, axis=1)[:, -k:]
        predictions.extend(
            np.bincount(train_labels[row], minlength=len(classes)).argmax()
            for row in nearest
        )
    return _balanced_accuracy(encoded[test_index], np.asarray(predictions))


def _knn_fidelity(
    cfg: dict,
    cells: pd.DataFrame,
    zero_shot: np.ndarray,
    aligned: np.ndarray,
) -> dict:
    example_cfg = cfg.get("example", {})
    label_column = example_cfg.get("cell_type_column", "cell_type")
    split_column = cfg.get("align", {}).get("split_group", "embryo_id")
    if label_column not in cells or split_column not in cells:
        return {"knn_fidelity_status": f"skipped: requires {label_column!r} and {split_column!r}"}

    labels = cells[label_column].astype(object).where(cells[label_column].notna(), "NA").astype(str).to_numpy()
    groups = cells[split_column].astype(object).where(cells[split_column].notna(), "NA").astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or len(np.unique(labels)) < 2:
        return {"knn_fidelity_status": "skipped: at least two labels and split groups are required"}

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    rng.shuffle(unique_groups)
    test_groups = unique_groups[:max(1, int(np.ceil(0.2 * len(unique_groups))))]
    train_pool = np.flatnonzero(~np.isin(groups, test_groups))
    test_pool = np.flatnonzero(np.isin(groups, test_groups))
    if not len(train_pool) or not len(test_pool):
        return {"knn_fidelity_status": "skipped: empty group-aware train/test split"}
    train_index = rng.choice(train_pool, min(20_000, len(train_pool)), replace=False)
    test_index = rng.choice(test_pool, min(5_000, len(test_pool)), replace=False)
    zero_shot_score = _knn_accuracy(zero_shot, labels, train_index, test_index)
    align_score = _knn_accuracy(aligned, labels, train_index, test_index)
    return {
        "zero_shot_knn_balanced_accuracy": zero_shot_score,
        "align_knn_balanced_accuracy": align_score,
        "align_knn_ge_90pct_zero_shot": align_score >= 0.9 * zero_shot_score,
        "knn_train_cells": int(len(train_index)),
        "knn_test_cells": int(len(test_index)),
        "knn_split_group": split_column,
    }


def validate(cfg: dict, tag: str | None = None) -> dict:
    raw_dir = stage_directory(cfg, "zero-shot", tag)
    indptr = np.load(raw_dir / "indptr.npy", mmap_mode="r")
    indices = np.load(raw_dir / "indices.npy", mmap_mode="r")
    distance = np.load(raw_dir / "data.npy", mmap_mode="r")
    embedding = np.load(raw_dir / "pooled_embedding.npy", mmap_mode="r")
    cells = pd.read_parquet(raw_dir / "cell_scores.parquet")
    finite, in_range = _distance_checks(distance)
    checks = {
        "finite_distances": finite,
        "distance_range": in_range,
        "csr_consistent": bool(indptr[-1] == len(distance) == len(indices)),
        "unique_cells": bool(cells.cell_id.is_unique),
    }
    checks["zero_shot_pooled_effective_rank"] = _effective_rank(embedding)
    checks["zero_shot_rank_ge_32"] = checks["zero_shot_pooled_effective_rank"] >= min(32, embedding.shape[1])
    if (raw_dir / "reservoir_hu.npy").exists():
        hu = np.asarray(np.load(raw_dir / "reservoir_hu.npy", mmap_mode="r")[:100_000], np.float32)
        hs = np.asarray(np.load(raw_dir / "reservoir_hs.npy", mmap_mode="r")[:100_000], np.float32)
        normalize = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        matched = float(np.mean(1 - np.sum(normalize(hu) * normalize(hs), axis=1)))
        shuffled = float(np.mean(1 - np.sum(normalize(hu) * normalize(np.roll(hs, 1, axis=0)), axis=1)))
        checks.update({"zero_shot_matched_distance": matched, "zero_shot_shuffled_distance": shuffled, "matched_lt_shuffle": matched < shuffled})
    aligned = stage_directory(cfg, "align-project", tag)
    if (aligned / "data.npy").exists():
        aligned_embedding = np.load(aligned / "pooled_embedding.npy", mmap_mode="r")
        aligned_indptr = np.load(aligned / "indptr.npy", mmap_mode="r")
        aligned_indices = np.load(aligned / "indices.npy", mmap_mode="r")
        checks["same_missing_mask"] = _arrays_equal(indptr, aligned_indptr) and _arrays_equal(indices, aligned_indices)
        checks["align_pooled_effective_rank"] = _effective_rank(aligned_embedding)
        checks["align_rank_ge_32"] = checks["align_pooled_effective_rank"] >= min(32, aligned_embedding.shape[1])
        checks.update(_knn_fidelity(cfg, cells, embedding, aligned_embedding))
    passed = all(value for value in checks.values() if isinstance(value, bool))
    out = provenance(cfg, "validate", tag)
    write_json(out / "validation.json", checks)
    write_json(out / "status.json", {"status": "complete" if passed else "failed", "checks": checks})
    return checks

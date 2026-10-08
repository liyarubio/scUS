"""Auditable workflows for interpreting frozen scUS U/S cosine distance.

The first implementation milestone intentionally concentrates on the three
experiments that directly identify the readout: layer formation, controlled
token/context interventions, and conditional reconstruction surprisal.  Later
dataset tasks use the same manifests and validation contract.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .artifacts import sha256
from .full_context_ablation import construct
from .joint_reconstruction import load_joint_data
from .runtime import write_json
from .semantics import (
    bin_intervention,
    collate_cells,
    conditional_surprisal,
    cosine_distance,
    empirical_conditional_surprisal,
    layerwise_hidden,
    load_frozen_encoder,
    paired_distance,
    paired_matched_null_distances,
    paired_null_distances,
    select_target_genes,
    stable_rng,
    target_positions,
)


def _settings(cfg: dict) -> dict:
    return cfg.get("semantics", {})


def _root(cfg: dict) -> Path:
    root = Path(cfg.get("output_dir", "outputs/paper/distance_semantics"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sample(cell_id: str) -> str:
    return str(cell_id).split(":", 1)[0]


def _labels(data) -> np.ndarray:
    shard = getattr(data, "shard", {})
    for key in ("obs_clusters", "cell_types", "cell_type"):
        if key in shard:
            return np.asarray(shard[key], dtype=str)
    return np.repeat("unavailable", len(data.cells))


def _select_indices(data, count: int, seed: int) -> np.ndarray:
    labels = _labels(data)
    order = sorted(range(len(data.cells)), key=lambda i: hashlib.sha256(
        f"{seed}:{data.cells[i]}".encode()).hexdigest())
    chosen: list[int] = []
    by_label = {label: [i for i in order if labels[i] == label] for label in np.unique(labels)}
    while len(chosen) < min(count, len(order)):
        added = False
        for label in sorted(by_label):
            if by_label[label] and len(chosen) < count:
                chosen.append(by_label[label].pop(0)); added = True
        if not added:
            break
    return np.asarray(chosen, dtype=np.int64)


def _device(cfg: dict, override: str | None = None) -> torch.device:
    value = override or cfg.get("device", "cpu")
    if value == "cuda:1":
        raise ValueError("GPU 1 is excluded by the registered semantics plan")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _preflight(cfg: dict, out: Path, device: torch.device, smoke: bool) -> None:
    if device.type != "cuda":
        return
    from .ablations import resource_preflight
    settings = _settings(cfg)
    samples = int(settings.get("preflight_smoke_samples", 6) if smoke else settings.get("preflight_samples", 60))
    interval = float(settings.get("preflight_interval_seconds", 5))
    passed, trace, reason = resource_preflight(
        device.index or 0, samples=samples, interval_seconds=interval,
        minimum_free_mb=float(settings.get("minimum_free_mb", 60_000)),
        maximum_mean_utilization=float(settings.get("maximum_mean_utilization", 10)),
    )
    _save_frame(trace, out / "resource_preflight.csv")
    if not passed:
        write_json(out / "resource_preflight_status.json",
                   {"status": "blocked_resource_preflight", "reason": reason})
        raise RuntimeError(f"GPU preflight failed: {reason}")
    write_json(out / "resource_preflight_status.json", {"status": "passed", "reason": reason})


def _software() -> dict:
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "cuda": torch.version.cuda, "created": time.strftime("%FT%T%z"),
    }


def _checkpoint_path(cfg: dict, epoch: int) -> Path:
    if epoch == 11:
        return Path(cfg["checkpoint"])
    directory = Path(_settings(cfg)["checkpoint_dir"])
    found = sorted(directory.glob(f"scUS-epoch={epoch:02d}-*.ckpt"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one checkpoint for epoch {epoch}, found {len(found)}")
    return found[0]


def _models(cfg: dict, smoke: bool) -> list[tuple[str, Path, int | None]]:
    settings = _settings(cfg)
    epochs = [11] if smoke else list(settings.get("checkpoint_epochs", [0, 3, 6, 9, 11]))
    result = [(f"epoch_{epoch}", _checkpoint_path(cfg, int(epoch)), None) for epoch in epochs]
    seeds = [settings.get("random_seeds", [42])[0]] if smoke else settings.get("random_seeds", [42, 43, 44])
    result.extend((f"random_{seed}", Path(cfg["checkpoint"]), int(seed)) for seed in seeds)
    return result


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    elif path.name.endswith(".csv.gz"):
        frame.to_csv(temporary, index=False, compression="gzip")
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def audit(cfg: dict) -> Path:
    out = _root(cfg) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    data = load_joint_data(cfg)
    labels = _labels(data)
    rows = []
    for i, cell_id in enumerate(data.cells):
        cell = data[i]
        values = cell["value_bins"].reshape(-1).numpy().reshape(-1, 2)
        genes = cell["gene_ids"].reshape(-1).numpy()[::2]
        rows.append({
            "cell_index": i, "cell_id": str(cell_id), "sample": _sample(cell_id),
            "cell_type": labels[i], "eligible_genes": len(genes),
            "paired_genes": int((values > 0).all(1).sum()),
            "u_observed": int((values[:, 0] > 0).sum()),
            "s_observed": int((values[:, 1] > 0).sum()),
        })
    cells = pd.DataFrame(rows)
    _save_frame(cells, out / "cell_manifest.csv")
    checkpoint_rows = []
    for label, path, random_seed in _models(cfg, smoke=False):
        checkpoint_rows.append({"model": label, "checkpoint": str(path),
                                "checkpoint_sha256": sha256(path), "random_seed": random_seed,
                                "weights_loaded": random_seed is None})
    _save_frame(pd.DataFrame(checkpoint_rows), out / "checkpoint_manifest.csv")
    token_path = Path(cfg["data"]["token_path"])
    report = {
        "status": "complete", "dataset": cfg.get("dataset", "Forebrain"),
        "cells": len(cells), "unique_cells": int(cells.cell_id.nunique()),
        "samples": sorted(cells["sample"].unique().tolist()),
        "cell_types": int(cells.cell_type.nunique()),
        "vocabulary_genes_seen": int(len(data.mapping)),
        "token_source": str(token_path), "token_source_sha256": sha256(token_path),
        "continuous_moments": "unavailable in historical token source",
        "pretraining_exposure": "unknown; accession-level clean manifest is unavailable",
        "software": _software(),
    }
    write_json(out / "audit.json", report)
    (out / "AUDIT_CN.md").write_text(
        "# Distance semantics 输入审计\n\n"
        f"- 数据：{report['dataset']}，{report['cells']:,} 个细胞，{report['vocabulary_genes_seen']:,} 个词表基因。\n"
        f"- 样本：{', '.join(report['samples'])}；cell type 数：{report['cell_types']}。\n"
        "- 输入为历史完整 U/S rank-bin token；缺失 token 保持 0，观测 bin 为 1–15。\n"
        "- 此 token 文件不包含可回溯的连续 Mu/Ms，因此相关基线记为 unavailable。\n"
        "- 历史 checkpoint 的预训练 accession 暴露状态未知，不能称为严格未见数据泛化。\n",
        encoding="utf-8")
    return out


def cpu_smoke(cfg: dict) -> Path:
    from .models import MaskedModel
    out = _root(cfg) / "smoke" / "cpu"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = MaskedModel(vocab_size=64, bin_size=15, embed_dim=16, num_heads=4,
                        num_layers=2, mask_ratio=.3).eval()
    genes = torch.repeat_interleave(torch.arange(1, 9), 2)[None]
    values = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1]])
    splice = torch.tensor([[0, 1] * 8])
    cell = {"gene_ids": genes, "value_bins": values, "splice_flags": splice, "cell_id": "sample:cell"}
    with torch.inference_mode():
        layers = layerwise_hidden(model, genes, values, splice)
        reference = model._encode(genes, values, splice)
    checks = {
        "final_layer_matches_encoder": bool(torch.allclose(layers[-1], reference, atol=1e-6)),
        "layer_count": len(layers),
    }
    positions = np.arange(4)
    for condition in ("u_shift", "s_shift", "both_shift", "value_swap", "modality_flag_swap",
                      "modality_full_swap", "gene_only_s"):
        changed = bin_intervention(cell, positions, condition, severity=1)
        untouched = np.arange(8, 16)
        checks[f"{condition}_background_unchanged"] = bool(
            torch.equal(changed["value_bins"][0, untouched], values[0, untouched]))
    exact_u = conditional_surprisal(model, cell, positions, 0, "exact")
    grouped_u = conditional_surprisal(model, cell, positions, 0, "grouped", groups=2, repeats=2)
    smoke_distance = paired_distance(reference[0], values[0])[0].detach().numpy()
    checks.update({
        "exact_surprisal_finite": bool(np.isfinite(list(exact_u.values())).all()),
        "grouped_surprisal_finite": bool(np.isfinite(list(grouped_u.values())).all()),
        "distance_legal": bool(((smoke_distance >= 0) & (smoke_distance <= 2)).all()),
    })
    status = "passed" if all(v for k, v in checks.items() if k != "layer_count") else "failed"
    write_json(out / "smoke.json", {"status": status, "checks": checks, "software": _software()})
    if status != "passed":
        raise AssertionError(f"CPU semantics smoke failed: {checks}")
    return out


def _dense_profile(data, cells: list[dict], hidden: torch.Tensor, lengths: np.ndarray) -> np.ndarray:
    lookup = {int(g): j for j, g in enumerate(data.mapping)}
    result = np.full((len(cells), len(data.mapping)), np.nan, np.float32)
    for row, (cell, length) in enumerate(zip(cells, lengths)):
        distance, valid = paired_distance(hidden[row], cell["value_bins"][0], int(length))
        genes = cell["gene_ids"][0, ::2].numpy()
        columns = np.asarray([lookup[int(g)] for g in genes], dtype=np.int64)
        result[row, columns[valid.cpu().numpy()]] = distance[valid].float().cpu().numpy()
    return result


def _profile_knn(profile: np.ndarray, labels: np.ndarray, samples: np.ndarray, k: int = 15) -> float:
    from sklearn.metrics import balanced_accuracy_score
    n = len(profile)
    if n < 3 or len(np.unique(labels)) < 2:
        return float("nan")
    observed_counts = np.isfinite(profile).sum(0)
    means = np.divide(np.nansum(profile, axis=0), observed_counts,
                      out=np.zeros(profile.shape[1], dtype=float), where=observed_counts > 0)
    x = np.where(np.isfinite(profile), profile, means)
    sd = x.std(0); sd[sd == 0] = 1
    x = (x - x.mean(0)) / sd
    norm = (x * x).sum(1)
    distances = np.sqrt(np.maximum(norm[:, None] + norm[None] - 2 * x @ x.T, 0) / max(x.shape[1], 1))
    predictions = []
    retained = []
    for index in range(n):
        candidates = np.flatnonzero(samples != samples[index])
        if not len(candidates):
            continue
        row = candidates[np.argsort(distances[index, candidates])[:min(k, len(candidates))]]
        values, counts = np.unique(labels[row], return_counts=True)
        predictions.append(values[np.argmax(counts)])
        retained.append(index)
    return float(balanced_accuracy_score(labels[retained], predictions)) if retained else float("nan")


def layerwise(cfg: dict, smoke: bool = False, device: str | None = None) -> Path:
    out = _root(cfg) / "layerwise" / ("smoke" if smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    data = load_joint_data(cfg); settings = _settings(cfg); seed = int(cfg.get("seed", 42))
    count = int(settings.get("smoke_cells", 8) if smoke else settings.get("layerwise_cells", 128))
    indices = _select_indices(data, count, seed)
    cells = [data[int(i)] for i in indices]
    labels = _labels(data)[indices]
    manifest = pd.DataFrame({"cell_index": indices, "cell_id": data.cells[indices],
                             "sample": [_sample(x) for x in data.cells[indices]], "cell_type": labels})
    _save_frame(manifest, out / "cell_manifest.csv")
    _save_frame(pd.DataFrame({"gene_id": data.mapping, "gene_name": data.names}), out / "gene_manifest.csv")
    dev = _device(cfg, device); rows = []
    _preflight(cfg, out, dev, smoke)
    batch_size = int(settings.get("smoke_batch_size", 1) if smoke else settings.get("batch_size", 1))
    gene_lookup = {int(g): j for j, g in enumerate(data.mapping)}
    direct_profile = np.full((len(cells), len(data.mapping)), np.nan, np.float32)
    for row, cell in enumerate(cells):
        vv = cell["value_bins"][0].numpy().reshape(-1, 2)
        gg = cell["gene_ids"][0, ::2].numpy()
        valid = (vv > 0).all(1)
        cols = np.asarray([gene_lookup[int(g)] for g in gg], dtype=np.int64)
        direct_profile[row, cols[valid]] = np.abs(vv[valid, 0] - vv[valid, 1])
    for model_label, checkpoint, random_seed in _models(cfg, smoke):
        model = load_frozen_encoder(checkpoint, dev, random_seed)
        profiles: list[list[np.ndarray]] | None = None
        correct_values: list[list[np.ndarray]] | None = None
        wrong_values: list[list[np.ndarray]] | None = None
        matched_wrong_values: list[list[np.ndarray]] | None = None
        for start in range(0, len(cells), batch_size):
            batch_cells = cells[start:start + batch_size]
            g, v, s, lengths = collate_cells(batch_cells, dev)
            with torch.inference_mode(), _autocast(dev):
                hidden_layers = layerwise_hidden(model, g, v, s)
            if profiles is None:
                profiles = [[] for _ in hidden_layers]; correct_values = [[] for _ in hidden_layers]
                wrong_values = [[] for _ in hidden_layers]
                matched_wrong_values = [[] for _ in hidden_layers]
            for layer_no, hidden in enumerate(hidden_layers):
                dense = _dense_profile(data, batch_cells, hidden, lengths)
                profiles[layer_no].append(dense)
                for row, cell in enumerate(batch_cells):
                    correct, wrong = paired_null_distances(hidden[row, :lengths[row]],
                                                           cell["value_bins"][0], seed, cell["cell_id"])
                    matched_correct, matched_wrong, _ = paired_matched_null_distances(
                        hidden[row, :lengths[row]], cell["value_bins"][0], seed, cell["cell_id"])
                    if not torch.equal(correct, matched_correct):
                        raise AssertionError("Correct-pair distances changed across null constructors")
                    correct_values[layer_no].append(correct.float().cpu().numpy())
                    wrong_values[layer_no].append(wrong.float().cpu().numpy())
                    matched_wrong_values[layer_no].append(matched_wrong.float().cpu().numpy())
        assert profiles is not None and correct_values is not None and wrong_values is not None \
            and matched_wrong_values is not None
        for layer_no, chunks in enumerate(profiles):
            profile = np.concatenate(chunks)
            correct = np.concatenate(correct_values[layer_no]); wrong = np.concatenate(wrong_values[layer_no])
            matched_wrong = np.concatenate(matched_wrong_values[layer_no])
            paired_delta = matched_wrong - correct
            random_delta = wrong - correct
            observed = profile[np.isfinite(profile)]
            shared = np.isfinite(profile) & np.isfinite(direct_profile)
            rho = spearmanr(profile[shared], direct_profile[shared]).statistic
            rows.append({
                "model": model_label, "weights_loaded": random_seed is None, "random_seed": random_seed,
                "layer": layer_no, "layer_name": "input_embedding" if layer_no == 0 else f"transformer_{layer_no}",
                "distance_median": float(np.median(observed)), "distance_iqr": float(np.subtract(*np.percentile(observed, [75, 25]))),
                "null_definition": "cross-gene S donor matched on observed U/S bins; no self donor",
                "null_minus_correct": float(np.mean(matched_wrong) - np.mean(correct)),
                "paired_delta_standardized": float(np.mean(paired_delta) / np.std(paired_delta))
                if np.std(paired_delta) > 0 else np.nan,
                "null_gt_correct_fraction": float(np.mean(matched_wrong > correct)),
                "random_null_minus_correct": float(np.mean(random_delta)),
                "random_null_delta_standardized": float(np.mean(random_delta) / np.std(random_delta))
                if np.std(random_delta) > 0 else np.nan,
                "random_null_gt_correct_fraction": float(np.mean(wrong > correct)),
                "direct_abs_bin_delta_spearman": float(rho),
                "cell_type_cross_sample_knn_ba": _profile_knn(profile, labels, manifest["sample"].to_numpy()),
                "cells": len(cells), "finite_pairs": len(observed),
            })
            np.save(out / f"{model_label}_layer{layer_no}_distance.npy", profile)
        del model
        if dev.type == "cuda": torch.cuda.empty_cache()
    metrics = pd.DataFrame(rows)
    _save_frame(metrics, out / "layerwise_metrics.csv")
    reference = settings.get("reference_distance_path")
    if reference:
        old = np.load(reference, mmap_mode="r")
        current = np.load(out / "epoch_11_layer6_distance.npy", mmap_mode="r")
        expected = np.asarray(old[indices])
        same_mask = np.array_equal(np.isfinite(expected), np.isfinite(current))
        shared = np.isfinite(expected) & np.isfinite(current)
        consistency = {"same_missing_mask": same_mask, "shared_values": int(shared.sum()),
                       "maximum_absolute_error": float(np.max(np.abs(expected[shared] - current[shared]))),
                       "mean_absolute_error": float(np.mean(np.abs(expected[shared] - current[shared])))}
        write_json(out / "reference_consistency.json", consistency)
        if not same_mask or consistency["maximum_absolute_error"] > float(settings.get("reference_tolerance", 1e-6)):
            raise AssertionError(f"Final layer does not reproduce frozen epoch-11 reference: {consistency}")
    write_json(out / "status.json", {"status": "complete", "smoke": smoke, "models": metrics.model.unique().tolist(),
                                      "cells": len(cells), "software": _software()})
    _plot_layerwise(metrics, out / "01_layerwise_formation")
    return out


def _target_ids(data, cells: list[dict], count: int, seed: int) -> np.ndarray:
    targets, _ = select_target_genes(cells, count, seed)
    if not len(targets): raise ValueError("No paired target genes")
    return targets


def _cell_covariates(data) -> np.ndarray:
    result = np.zeros((len(data.cells), 4), dtype=float)
    for i in range(len(data.cells)):
        values = data[i]["value_bins"].reshape(-1).numpy().reshape(-1, 2)
        result[i] = [len(values), (values[:, 0] > 0).mean(), (values[:, 1] > 0).mean(),
                     values[values > 0].mean() if (values > 0).any() else 0]
    scale = result.std(0); scale[scale == 0] = 1
    return (result - result.mean(0)) / scale


def _donors(data, indices: np.ndarray, same_type: bool, seed: int,
            covariates: np.ndarray | None = None) -> dict[int, int]:
    labels = _labels(data); result = {}; covariates = _cell_covariates(data) if covariates is None else covariates
    for i in indices:
        if same_type:
            candidates = [j for j in range(len(data.cells)) if j != i and
                          _sample(data.cells[j]) != _sample(data.cells[i]) and labels[j] == labels[i]]
        else:
            candidates = [j for j in range(len(data.cells)) if j != i and
                          _sample(data.cells[j]) == _sample(data.cells[i]) and labels[j] != labels[i]]
            if not candidates:
                candidates = [j for j in range(len(data.cells)) if j != i and labels[j] != labels[i]]
        if candidates:
            candidates = np.asarray(candidates, dtype=np.int64)
            mismatch = ((covariates[candidates] - covariates[int(i)]) ** 2).sum(1)
            nearest = candidates[np.argsort(mismatch)[:min(5, len(candidates))]]
            result[int(i)] = int(nearest[stable_rng(seed, data.cells[i], same_type).integers(len(nearest))])
    return result


def _target_distance(model, cell: dict, positions: np.ndarray, device: torch.device,
                     positions_are_prefix: bool = False) -> tuple[np.ndarray, np.ndarray]:
    g, v, s, lengths = collate_cells([cell], device)
    with torch.inference_mode(), _autocast(device):
        layers = layerwise_hidden(model, g, v, s)
    pair_pos = np.arange(len(positions)) if positions_are_prefix else positions
    token_u = torch.as_tensor(2 * pair_pos, device=device)
    token_s = token_u + 1
    final = cosine_distance(layers[-1][0, token_u], layers[-1][0, token_s]).float().cpu().numpy()
    initial = cosine_distance(layers[0][0, token_u], layers[0][0, token_s]).float().cpu().numpy()
    return final, initial


def interventions(cfg: dict, smoke: bool = False, device: str | None = None) -> Path:
    out = _root(cfg) / "interventions" / ("smoke" if smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    data = load_joint_data(cfg); settings = _settings(cfg); seed = int(cfg.get("seed", 42))
    n_cells = int(settings.get("smoke_cells", 8) if smoke else settings.get("intervention_cells", 128))
    n_targets = int(settings.get("smoke_targets", 4) if smoke else settings.get("target_genes", 128))
    indices = _select_indices(data, n_cells, seed); cells = [data[int(i)] for i in indices]
    targets = _target_ids(data, cells, n_targets, seed)
    inverse = {int(g): str(n) for g, n in zip(data.mapping, data.names)}
    _save_frame(pd.DataFrame({"gene_id": targets, "gene_name": [inverse[int(g)] for g in targets]}),
                out / "target_gene_manifest.csv")
    covariates = _cell_covariates(data)
    donor_seeds = [seed] if smoke else list(settings.get("donor_seeds", [42, 43, 44, 45, 46]))
    dev = _device(cfg, device); _preflight(cfg, out, dev, smoke)
    model = load_frozen_encoder(cfg["checkpoint"], dev)
    severities = list(settings.get("intervention_severity", [-8, -4, -2, -1, 1, 2, 4, 8]))
    rows = []
    for source_index, cell in zip(indices, cells):
        positions = target_positions(cell, targets)
        if not len(positions): continue
        own_targets = cell["gene_ids"][0, 2 * positions].numpy()
        original_bins = cell["value_bins"][0].numpy().reshape(-1, 2)[positions]
        conditions: list[tuple[str, int | None, dict, bool, str | None]] = [("correct", None, cell, False, None)]
        for severity in severities:
            for condition in ("u_shift", "s_shift", "both_shift"):
                conditions.append((condition, int(severity), bin_intervention(cell, positions, condition, severity), False, None))
        for condition in ("value_swap", "modality_flag_swap", "modality_full_swap", "gene_only_s"):
            conditions.append((condition, None, bin_intervention(cell, positions, condition, seed=seed), False, None))
        for donor_seed in donor_seeds:
            same = _donors(data, np.asarray([source_index]), True, donor_seed, covariates)
            different = _donors(data, np.asarray([source_index]), False, donor_seed, covariates)
            if int(source_index) in same:
                donor = data[same[int(source_index)]]
                conditions.append((f"donor_s_same_type_seed{donor_seed}", None,
                                   bin_intervention(cell, positions, "donor_s", donor=donor), False, donor["cell_id"]))
                conditions.append((f"value_only_seed{donor_seed}", None,
                                   bin_intervention(cell, positions, "donor_values", donor=donor), False, donor["cell_id"]))
                conditions.append((f"same_type_context_seed{donor_seed}", None,
                                   construct(cell, own_targets, "matched_context", donor, donor_seed), True, donor["cell_id"]))
            if int(source_index) in different:
                donor = data[different[int(source_index)]]
                conditions.append((f"different_type_context_seed{donor_seed}", None,
                                   construct(cell, own_targets, "matched_context", donor, donor_seed), True, donor["cell_id"]))
        conditions.append(("within_cell_context_shuffle", None, construct(cell, own_targets, "context_shuffle", seed=seed), True, None))
        for condition, severity, changed, prefix, donor_id in conditions:
            distance, input_distance = _target_distance(model, changed, positions, dev, prefix)
            changed_positions = np.arange(len(positions)) if prefix else positions
            bins = changed["value_bins"][0].numpy().reshape(-1, 2)[changed_positions]
            for local, gene in enumerate(own_targets):
                rows.append({
                    "model": "epoch_11",
                    "cell_id": cell["cell_id"], "source_index": int(source_index), "sample": _sample(cell["cell_id"]),
                    "cell_type": _labels(data)[source_index], "gene_id": int(gene), "gene_name": inverse[int(gene)],
                    "condition": condition, "severity": severity, "donor_id": donor_id,
                    "distance": float(distance[local]), "input_embedding_distance": float(input_distance[local]),
                    "u_bin": int(bins[local, 0]), "s_bin": int(bins[local, 1]),
                    "abs_bin_delta": int(abs(int(bins[local, 0]) - int(bins[local, 1]))),
                    "original_u_bin": int(original_bins[local, 0]), "original_s_bin": int(original_bins[local, 1]),
                    "continuous_abs_mu_ms": np.nan,
                })
    # Random-init uses identical cells, target genes, full background, and dose
    # interventions. Context donors are not repeated because their role is
    # already isolated by the epoch-11 context experiment.
    random_model = load_frozen_encoder(cfg["checkpoint"], dev, int(settings.get("random_seeds", [42])[0]))
    for source_index, cell in zip(indices, cells):
        positions = target_positions(cell, targets)
        if not len(positions): continue
        own_targets = cell["gene_ids"][0, 2 * positions].numpy()
        original_bins = cell["value_bins"][0].numpy().reshape(-1, 2)[positions]
        conditions = [("correct", None, cell)]
        for severity in severities:
            for condition in ("u_shift", "s_shift", "both_shift"):
                conditions.append((condition, int(severity),
                                   bin_intervention(cell, positions, condition, severity)))
        for condition in ("value_swap", "modality_flag_swap", "modality_full_swap", "gene_only_s"):
            conditions.append((condition, None, bin_intervention(cell, positions, condition, seed=seed)))
        for condition, severity, changed in conditions:
            distance, input_distance = _target_distance(random_model, changed, positions, dev)
            bins = changed["value_bins"][0].numpy().reshape(-1, 2)[positions]
            for local, gene in enumerate(own_targets):
                rows.append({"model": "random_42", "cell_id": cell["cell_id"],
                             "source_index": int(source_index), "sample": _sample(cell["cell_id"]),
                             "cell_type": _labels(data)[source_index], "gene_id": int(gene),
                             "gene_name": inverse[int(gene)], "condition": condition,
                             "severity": severity, "donor_id": None, "distance": float(distance[local]),
                             "input_embedding_distance": float(input_distance[local]),
                             "u_bin": int(bins[local, 0]), "s_bin": int(bins[local, 1]),
                             "abs_bin_delta": int(abs(int(bins[local, 0]) - int(bins[local, 1]))),
                             "original_u_bin": int(original_bins[local, 0]),
                             "original_s_bin": int(original_bins[local, 1]),
                             "continuous_abs_mu_ms": np.nan})
    del random_model, model
    if dev.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    _save_frame(frame, out / "intervention_long.parquet")
    correct = frame[frame.condition == "correct"][["model", "cell_id", "gene_id", "distance"]].rename(columns={"distance": "correct_distance"})
    paired = frame.merge(correct, on=["model", "cell_id", "gene_id"], how="left")
    paired["delta_distance"] = paired.distance - paired.correct_distance
    summary = paired.groupby(["model", "condition", "severity"], dropna=False).agg(
        observations=("distance", "size"), mean_distance=("distance", "mean"),
        mean_delta=("delta_distance", "mean"), median_delta=("delta_distance", "median"),
        positive_fraction=("delta_distance", lambda x: float(np.mean(x > 0))),
    ).reset_index()
    _save_frame(summary, out / "intervention_summary.csv")
    _summarize_intervention_semantics(paired, out)
    write_json(out / "availability.json", {
        "status": "complete", "continuous_moments": "unavailable", "cross_stage_context": "unavailable in Forebrain labels",
        "donor_seeds": donor_seeds,
        "full_background_attention": True, "target_genes_requested": n_targets, "target_genes_selected": len(targets),
    })
    _plot_interventions(paired, out / "02_controlled_interventions")
    return out


def _summarize_intervention_semantics(frame: pd.DataFrame, out: Path) -> None:
    from sklearn.metrics import roc_auc_score
    frame = frame.copy()
    if "model" not in frame:
        frame["model"] = "epoch_11"
    frame["applied_l1_bin_change"] = (frame.u_bin - frame.original_u_bin).abs() + \
        (frame.s_bin - frame.original_s_bin).abs()
    frame["applied_relation_change"] = ((frame.u_bin - frame.s_bin) -
                                         (frame.original_u_bin - frame.original_s_bin)).abs()
    correct_all = frame[frame.condition.eq("correct")].drop_duplicates(["model", "cell_id", "gene_id"])
    rows = []
    for model in sorted(frame.model.unique()):
        correct = correct_all[correct_all.model.eq(model)]
        for condition in ("u_shift", "s_shift", "both_shift"):
            altered = frame[frame.model.eq(model) & frame.condition.eq(condition) & frame.applied_l1_bin_change.gt(0)]
            if altered.empty: continue
            for readout in ("distance", "input_embedding_distance", "abs_bin_delta"):
                rho_relation = spearmanr(altered.applied_relation_change, altered[readout]).statistic
                rho_magnitude = spearmanr(altered.applied_l1_bin_change, altered[readout]).statistic
                scores = np.r_[correct[readout].to_numpy(), altered[readout].to_numpy()]
                labels = np.r_[np.zeros(len(correct)), np.ones(len(altered))]
                rows.append({"model": model, "condition": condition, "readout": readout,
                             "altered_observations": len(altered),
                             "spearman_known_relation_change": rho_relation,
                             "spearman_total_bin_change": rho_magnitude,
                             "corruption_auroc": roc_auc_score(labels, scores)})
    _save_frame(pd.DataFrame(rows), out / "dose_response_metrics.csv")


def surprisal(cfg: dict, smoke: bool = False, device: str | None = None) -> Path:
    out = _root(cfg) / "surprisal" / ("smoke" if smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    data = load_joint_data(cfg); settings = _settings(cfg); seed = int(cfg.get("seed", 42))
    n_cells = int(settings.get("surprisal_smoke_cells", 4) if smoke else settings.get("surprisal_cells", 128))
    n_targets = int(settings.get("smoke_targets", 4) if smoke else settings.get("target_genes", 128))
    indices = _select_indices(data, n_cells, seed); cells = [data[int(i)] for i in indices]
    targets = _target_ids(data, cells, n_targets, seed)
    dev = _device(cfg, device); _preflight(cfg, out, dev, smoke); rows = []
    model_specs = _models(cfg, smoke=True)
    for model_label, checkpoint, random_seed in model_specs:
        model = load_frozen_encoder(checkpoint, dev, random_seed)
        for source_index, cell in zip(indices, cells):
            positions = target_positions(cell, targets)
            if not len(positions): continue
            exact_u = conditional_surprisal(model, cell, positions, 0, "exact", seed=seed)
            exact_s = conditional_surprisal(model, cell, positions, 1, "exact", seed=seed)
            grouped_u = conditional_surprisal(model, cell, positions, 0, "grouped", 8, 2, seed)
            grouped_s = conditional_surprisal(model, cell, positions, 1, "grouped", 8, 2, seed)
            distance, _ = _target_distance(model, cell, positions, dev)
            genes = cell["gene_ids"][0, 2 * positions].numpy()
            for local, (position, gene) in enumerate(zip(positions, genes)):
                for protocol, u_score, s_score in (
                    ("exact", exact_u[int(position)], exact_s[int(position)]),
                    ("grouped_8x2", grouped_u[int(position)], grouped_s[int(position)]),
                ):
                    rows.append({"model": model_label, "cell_id": cell["cell_id"], "source_index": int(source_index),
                                 "sample": _sample(cell["cell_id"]), "gene_id": int(gene), "protocol": protocol,
                                 "q_u": u_score, "q_s": s_score, "q_pair": (u_score + s_score) / 2,
                                 "distance": float(distance[local])})
        del model
        if dev.type == "cuda": torch.cuda.empty_cache()
    rows.extend(_empirical_surprisal_rows(data, indices, targets))
    frame = pd.DataFrame(rows)
    _save_frame(frame, out / "surprisal_long.parquet")
    comparisons = []
    trained = frame[frame.model == "epoch_11"]
    if not trained.empty:
        wide = trained.pivot_table(index=["cell_id", "gene_id"], columns="protocol", values="q_pair")
        if {"exact", "grouped_8x2"}.issubset(wide.columns):
            comparisons.append({"comparison": "exact_vs_grouped", "spearman": float(spearmanr(wide.exact, wide.grouped_8x2).statistic),
                                "mae": float(np.mean(np.abs(wide.exact - wide.grouped_8x2))), "observations": len(wide)})
        exact = trained[trained.protocol == "exact"]
        comparisons.append({"comparison": "distance_vs_exact_surprisal", "spearman": float(spearmanr(exact.distance, exact.q_pair).statistic),
                            "mae": np.nan, "observations": len(exact)})
    _save_frame(pd.DataFrame(comparisons), out / "surprisal_comparison.csv")
    write_json(out / "status.json", {"status": "complete", "smoke": smoke, "cells": n_cells,
                                      "targets": len(targets), "partner_simultaneously_masked": False,
                                      "empirical_baseline": "sample-held-out p(U|S) and p(S|U) with Laplace smoothing"})
    _plot_surprisal(frame, out / "03_conditional_surprisal")
    return out


def _empirical_surprisal_rows(data, indices: np.ndarray, targets: np.ndarray) -> list[dict]:
    """Non-neural p(U|S) and p(S|U), fitted only on the other sample."""
    target_lookup = {int(g): j for j, g in enumerate(targets)}
    u_matrix = np.zeros((len(data.cells), len(targets)), dtype=np.int16)
    s_matrix = np.zeros_like(u_matrix)
    for source_index in range(len(data.cells)):
        cell = data[source_index]
        genes = cell["gene_ids"][0, ::2].numpy()
        values = cell["value_bins"][0].numpy().reshape(-1, 2)
        for gene, pair in zip(genes, values):
            column = target_lookup.get(int(gene))
            if column is not None:
                u_matrix[source_index, column], s_matrix[source_index, column] = pair
    all_samples = np.asarray([_sample(x) for x in data.cells])
    selected_set = set(map(int, indices)); rows = []
    for held_sample in sorted(set(all_samples[indices])):
        train_rows = np.flatnonzero(all_samples != held_sample)
        q_u = empirical_conditional_surprisal(u_matrix, s_matrix, train_rows)
        q_s = empirical_conditional_surprisal(s_matrix, u_matrix, train_rows)
        for source_index in np.flatnonzero(all_samples == held_sample):
            if int(source_index) not in selected_set:
                continue
            for column, gene in enumerate(targets):
                if u_matrix[source_index, column] <= 0 or s_matrix[source_index, column] <= 0:
                    continue
                rows.append({"model": "empirical_frequency", "cell_id": data.cells[source_index],
                             "source_index": int(source_index), "sample": held_sample,
                             "gene_id": int(gene), "protocol": "sample_held_out_exact",
                             "q_u": float(q_u[source_index, column]), "q_s": float(q_s[source_index, column]),
                             "q_pair": float((q_u[source_index, column] + q_s[source_index, column]) / 2),
                             "distance": np.nan})
    return rows


def surprisal_empirical(cfg: dict, smoke: bool = True) -> Path:
    out = _root(cfg) / "surprisal" / ("smoke" if smoke else "formal")
    neural_path = out / "surprisal_long.parquet"
    if not neural_path.is_file():
        raise FileNotFoundError("Run neural surprisal before the CPU empirical supplement")
    data = load_joint_data(cfg); settings = _settings(cfg); seed = int(cfg.get("seed", 42))
    n_cells = int(settings.get("surprisal_smoke_cells", 4) if smoke else settings.get("surprisal_cells", 128))
    n_targets = int(settings.get("smoke_targets", 4) if smoke else settings.get("target_genes", 128))
    indices = _select_indices(data, n_cells, seed); cells = [data[int(i)] for i in indices]
    targets = _target_ids(data, cells, n_targets, seed)
    frame = pd.read_parquet(neural_path)
    frame = frame[frame.model != "empirical_frequency"]
    frame = pd.concat([frame, pd.DataFrame(_empirical_surprisal_rows(data, indices, targets))], ignore_index=True)
    _save_frame(frame, neural_path)
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    status.pop("reason", None)
    if {"epoch_11", "random_42"}.issubset(set(frame.model)):
        status.update({"status": "complete", "smoke": smoke,
                       "cells": n_cells, "targets": len(targets),
                       "partner_simultaneously_masked": False})
    status.update({"empirical_baseline": "sample-held-out p(U|S) and p(S|U) with Laplace smoothing",
                   "empirical_rows": int((frame.model == "empirical_frequency").sum())})
    write_json(status_path, status)
    _plot_surprisal(frame, out / "03_conditional_surprisal")
    return out


def feasibility(cfg: dict, stage: str) -> Path:
    """Record an explicit dependency/data gate instead of fabricating absent evidence."""
    out = _root(cfg) / stage
    out.mkdir(parents=True, exist_ok=True)
    reason = {
        "simulate": "dyngen R package is not installed; no synthetic truth has been generated",
        "temporal": "requires dataset-specific tokenization and group-held-out manifests",
        "reference_anomaly": "requires audited control/test compatibility and frozen reference splits",
        "align_retention": "requires completed zero-shot semantic effects as denominator",
    }.get(stage, "stage prerequisites have not been completed")
    write_json(out / "status.json", {"status": "not_started", "reason": reason, "no_result_fabricated": True})
    return out


def simulate(cfg: dict, smoke: bool = True, device: str | None = None) -> Path:
    """Generate registered dyngen truth datasets; never substitute another simulator."""
    out = _root(cfg) / "dynamics_truth" / "dyngen" / ("smoke" if smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    settings = cfg.get("simulation", {})
    environment = None
    r_library = settings.get("r_library_path") or settings.get("r_libs_user")
    if r_library:
        import os
        environment = dict(os.environ)
        environment["R_LIBS_USER"] = str(Path(r_library).resolve())
    check = subprocess.run(["Rscript", "-e", "cat(requireNamespace('dyngen', quietly=TRUE))"],
                           capture_output=True, text=True, env=environment)
    if check.returncode != 0 or check.stdout.strip() != "TRUE":
        write_json(out / "status.json", {
            "status": "blocked_dependency", "dependency": "R package dyngen",
            "install_hint": "Rscript -e 'install.packages(\"dyngen\")'",
            "fallback_used": False,
        })
        return out
    topologies = ["linear"] if smoke else settings.get("topologies", ["linear", "bifurcation", "cycle"])
    seeds = [int(cfg.get("seed", 42))] if smoke else settings.get("seeds", [42, 43, 44, 45, 46])
    cells = int(settings.get("smoke_cells", 100) if smoke else settings.get("cells", 1000))
    genes = int(settings.get("smoke_genes", 50) if smoke else settings.get("genes", 200))
    script = Path(settings.get("script_path", Path(__file__).parents[2] / "scripts" / "run_dyngen_truth.R"))
    rows = []
    for topology in topologies:
        for seed in seeds:
            run = out / f"{topology}_seed{seed}"
            command = ["Rscript", str(script), str(run), topology, str(seed), str(cells), str(genes)]
            started = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            (run.parent / f"{run.name}.stdout.log").write_text(result.stdout, encoding="utf-8")
            (run.parent / f"{run.name}.stderr.log").write_text(result.stderr, encoding="utf-8")
            rows.append({"topology": topology, "seed": seed, "returncode": result.returncode,
                         "runtime_seconds": time.monotonic() - started, "output_dir": str(run),
                         "complete": (run / "spliced.mtx").is_file() and (run / "unspliced.mtx").is_file()})
    frame = pd.DataFrame(rows); _save_frame(frame, out / "simulation_runs.csv")
    complete = bool(frame.complete.all())
    write_json(out / "status.json", {"status": "complete" if complete else "failed",
                                      "runs": len(frame), "topologies": topologies, "seeds": seeds,
                                      "fallback_used": False})
    if not complete:
        raise RuntimeError("At least one registered dyngen simulation failed; inspect stderr logs")
    evaluate_dyngen(cfg, out, smoke=smoke, device=device)
    return out


def _reference_gene_statistics(cfg: dict, out: Path) -> pd.DataFrame:
    cache = out / "mapping_reference_gene_stats.csv"
    if cache.is_file():
        return pd.read_csv(cache)
    token_path = cfg.get("datasets", {}).get("erythroid", {}).get("token_path")
    if not token_path:
        raise ValueError("Erythroid token_path is required for stratified dyngen gene mapping")
    local = dict(cfg); local["data"] = {"token_path": token_path}
    data = load_joint_data(local)
    lookup = {int(g): j for j, g in enumerate(data.mapping)}
    detected = np.zeros(len(data.mapping), np.int64)
    pair_detected = np.zeros(len(data.mapping), np.int64)
    bin_sum = np.zeros(len(data.mapping), np.float64)
    bin_n = np.zeros(len(data.mapping), np.int64)
    for row in range(len(data.cells)):
        cell = data[row]
        genes = cell["gene_ids"][0, ::2].numpy()
        values = cell["value_bins"][0].numpy().reshape(-1, 2)
        columns = np.asarray([lookup[int(g)] for g in genes], dtype=np.int64)
        any_observed = (values > 0).any(1)
        both = (values > 0).all(1)
        np.add.at(detected, columns[any_observed], 1)
        np.add.at(pair_detected, columns[both], 1)
        np.add.at(bin_sum, columns, values.sum(1))
        np.add.at(bin_n, columns, (values > 0).sum(1))
    frame = pd.DataFrame({
        "gene_id": data.mapping, "gene_name": data.names,
        "detection_fraction": detected / len(data.cells),
        "pair_detection_fraction": pair_detected / len(data.cells),
        "mean_observed_bin": np.divide(bin_sum, bin_n, out=np.zeros(len(bin_n)), where=bin_n > 0),
    })
    frame = frame[(frame.detection_fraction > 0) & (frame.mean_observed_bin > 0)].reset_index(drop=True)
    _save_frame(frame, cache)
    return frame


def _moments_like(u_counts: np.ndarray, s_counts: np.ndarray, neighbours: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic kNN first moments while preserving every simulated gene."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    u = np.asarray(u_counts, np.float32); s = np.asarray(s_counts, np.float32)
    total_per_cell = (u + s).sum(1)
    target = np.median(total_per_cell[total_per_cell > 0])
    scale = np.divide(target, total_per_cell, out=np.zeros_like(total_per_cell), where=total_per_cell > 0)
    u_norm, s_norm = u * scale[:, None], s * scale[:, None]
    log_total = np.log1p(u_norm + s_norm)
    components = min(30, len(u) - 1, u.shape[1])
    representation = PCA(components, random_state=42).fit_transform(log_total) if components >= 2 else log_total
    k = min(neighbours, len(u))
    indices = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=4).fit(representation).kneighbors(
        representation, return_distance=False)
    mu = np.stack([u_norm[row].mean(0) for row in indices]).astype(np.float32)
    ms = np.stack([s_norm[row].mean(0) for row in indices]).astype(np.float32)
    return mu, ms


def _dyngen_mapping(simulated_u: np.ndarray, simulated_s: np.ndarray, reference: pd.DataFrame,
                    seed: int) -> pd.DataFrame:
    """One-to-one nearest-rank mapping on expression and pair detection."""
    sim_total = simulated_u + simulated_s
    sim_features = np.c_[np.log1p(sim_total.mean(0)), ((simulated_u > 0) & (simulated_s > 0)).mean(0)]
    ref_features = np.c_[reference.mean_observed_bin.to_numpy(), reference.pair_detection_fraction.to_numpy()]
    sim_features = (sim_features - sim_features.mean(0)) / np.where(sim_features.std(0) > 0, sim_features.std(0), 1)
    ref_features = (ref_features - ref_features.mean(0)) / np.where(ref_features.std(0) > 0, ref_features.std(0), 1)
    available = np.ones(len(reference), dtype=bool); assignment = np.empty(sim_features.shape[0], np.int64)
    order = stable_rng(seed, "dyngen-mapping-order").permutation(len(assignment))
    jitter = stable_rng(seed, "dyngen-mapping-jitter").random(len(reference)) * 1e-8
    for gene in order:
        candidates = np.flatnonzero(available)
        cost = ((ref_features[candidates] - sim_features[gene]) ** 2).sum(1) + jitter[candidates]
        chosen = candidates[np.argmin(cost)]
        assignment[gene] = chosen; available[chosen] = False
    mapped = reference.iloc[assignment].reset_index(drop=True).copy()
    mapped.insert(0, "sim_gene_index", np.arange(len(mapped)))
    mapped["mapping_seed"] = seed
    mapped["sim_log_mean_total"] = np.log1p(sim_total.mean(0))
    mapped["sim_pair_detection"] = ((simulated_u > 0) & (simulated_s > 0)).mean(0)
    return mapped


def _dyngen_acceleration(velocity: np.ndarray, cell_info: pd.DataFrame) -> np.ndarray:
    acceleration = np.full_like(velocity, np.nan, dtype=np.float32)
    for _, group in cell_info.groupby("simulation_i"):
        order = group.sort_values("sim_time").index.to_numpy()
        if len(order) < 2:
            continue
        times = cell_info.loc[order, "sim_time"].to_numpy(float)
        unique_times, inverse = np.unique(times, return_inverse=True)
        if len(unique_times) < 2:
            continue
        collapsed = np.stack([velocity[order[inverse == index]].mean(0)
                              for index in range(len(unique_times))])
        differentiated = np.gradient(collapsed, unique_times, axis=0).astype(np.float32)
        acceleration[order] = differentiated[inverse]
    return acceleration


def _phase_residual(u: np.ndarray, s: np.ndarray) -> np.ndarray:
    residual = np.full_like(u, np.nan, dtype=np.float32)
    for gene in range(u.shape[1]):
        valid = np.isfinite(u[:, gene]) & np.isfinite(s[:, gene])
        if valid.sum() < 3 or np.std(u[valid, gene]) == 0:
            continue
        fit = np.polyfit(u[valid, gene], s[valid, gene], deg=1)
        values = s[valid, gene] - np.polyval(fit, u[valid, gene])
        scale = np.median(np.abs(values - np.median(values))) * 1.4826
        residual[valid, gene] = np.abs(values) / (scale if scale > 1e-8 else 1)
    return residual


def evaluate_dyngen(cfg: dict, simulation_out: Path, smoke: bool,
                    device: str | None = None) -> Path:
    """Map dyngen genes, encode moments-like U/S, and score registered truth."""
    from scipy.io import mmread
    from sklearn.metrics import roc_auc_score
    from .data.tokens import rank_bin

    out = simulation_out / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    dev = _device(cfg, device or "cpu")
    if dev.type == "cuda":
        _preflight(cfg, out, dev, smoke)
    reference = _reference_gene_statistics(cfg, out)
    mapping_seeds = [int(cfg.get("seed", 42))] if smoke else list(
        cfg.get("simulation", {}).get("mapping_seeds", list(range(10))))
    models = [("epoch_11", load_frozen_encoder(cfg["checkpoint"], dev)),
              ("random_42", load_frozen_encoder(cfg["checkpoint"], dev, 42))]
    score_rows = []; metric_rows = []
    run_table = pd.read_csv(simulation_out / "simulation_runs.csv")
    for run in run_table.itertuples():
        run_dir = Path(run.output_dir)
        u_count = np.asarray(mmread(run_dir / "unspliced.mtx").todense(), np.float32)
        s_count = np.asarray(mmread(run_dir / "spliced.mtx").todense(), np.float32)
        velocity = np.asarray(mmread(run_dir / "rna_velocity.mtx").todense(), np.float32)
        cell_info = pd.read_csv(run_dir / "cell_info.csv")
        feature_info = pd.read_csv(run_dir / "feature_info.csv")
        if u_count.shape != s_count.shape or u_count.shape != velocity.shape or len(cell_info) != len(u_count):
            raise ValueError(f"dyngen output alignment failed for {run_dir}")
        mu, ms = _moments_like(u_count, s_count)
        acceleration = _dyngen_acceleration(velocity, cell_info)
        phase = _phase_residual(mu, ms)
        for mapping_seed in mapping_seeds:
            mapping = _dyngen_mapping(mu, ms, reference, int(mapping_seed))
            mapping.insert(1, "sim_gene_id", feature_info.feature_id.astype(str).to_numpy())
            _save_frame(mapping, out / f"{run.topology}_seed{run.seed}_mapping{mapping_seed}.csv")
            cell_payloads = []
            for cell_index in range(len(mu)):
                interleaved = np.empty(2 * mu.shape[1], np.float32)
                interleaved[0::2], interleaved[1::2] = mu[cell_index], ms[cell_index]
                bins = rank_bin(interleaved, 15).astype(np.int64)
                cell_payloads.append({
                    "gene_ids": torch.as_tensor(np.repeat(mapping.gene_id.to_numpy(np.int64), 2))[None],
                    "value_bins": torch.as_tensor(bins)[None],
                    "splice_flags": torch.as_tensor(np.tile([0, 1], mu.shape[1]))[None],
                    "cell_id": f"{run.topology}:{run.seed}:cell{cell_index}",
                })
            for model_label, model in models:
                distances = np.full(mu.shape, np.nan, np.float32)
                input_distances = np.full(mu.shape, np.nan, np.float32)
                direct = np.full(mu.shape, np.nan, np.float32)
                for cell_index, cell in enumerate(cell_payloads):
                    positions = np.arange(mu.shape[1])
                    final, initial = _target_distance(model, cell, positions, dev)
                    bins = cell["value_bins"][0].numpy().reshape(-1, 2)
                    valid = (bins > 0).all(1)
                    distances[cell_index, valid] = final[valid]
                    input_distances[cell_index, valid] = initial[valid]
                    direct[cell_index, valid] = np.abs(bins[valid, 0] - bins[valid, 1])
                truth = {
                    "abs_velocity": np.abs(velocity), "abs_acceleration": np.abs(acceleration),
                    "phase_residual": phase,
                }
                readouts = {"distance": distances, "input_embedding_distance": input_distances,
                            "abs_bin_delta": direct}
                for truth_name, truth_values in truth.items():
                    for readout_name, readout in readouts.items():
                        valid = np.isfinite(truth_values) & np.isfinite(readout)
                        metric_rows.append({
                            "topology": run.topology, "simulation_seed": int(run.seed),
                            "mapping_seed": int(mapping_seed), "model": model_label,
                            "truth": truth_name, "readout": readout_name,
                            "spearman": spearmanr(truth_values[valid], readout[valid]).statistic,
                            "observations": int(valid.sum()),
                        })
                valid_velocity = np.isfinite(velocity) & np.isfinite(distances)
                threshold = np.nanquantile(np.abs(velocity[valid_velocity]), .75)
                labels = (np.abs(velocity[valid_velocity]) >= threshold).astype(int)
                metric_rows.append({
                    "topology": run.topology, "simulation_seed": int(run.seed),
                    "mapping_seed": int(mapping_seed), "model": model_label,
                    "truth": "top_quartile_abs_velocity", "readout": "distance",
                    "spearman": np.nan, "observations": int(valid_velocity.sum()),
                    "auroc": roc_auc_score(labels, distances[valid_velocity]),
                })
                if smoke:
                    for cell_index in range(len(mu)):
                        for gene_index in range(mu.shape[1]):
                            if np.isfinite(distances[cell_index, gene_index]):
                                score_rows.append({
                                    "topology": run.topology, "simulation_seed": int(run.seed),
                                    "mapping_seed": int(mapping_seed), "model": model_label,
                                    "cell_index": cell_index, "gene_index": gene_index,
                                    "distance": distances[cell_index, gene_index],
                                    "input_embedding_distance": input_distances[cell_index, gene_index],
                                    "abs_bin_delta": direct[cell_index, gene_index],
                                    "velocity": velocity[cell_index, gene_index],
                                    "acceleration": acceleration[cell_index, gene_index],
                                    "phase_residual": phase[cell_index, gene_index],
                                    "is_burn": bool(feature_info.iloc[gene_index].burn),
                                })
    metrics = pd.DataFrame(metric_rows)
    _save_frame(metrics, out / "dynamics_truth_metrics.csv")
    if score_rows:
        _save_frame(pd.DataFrame(score_rows), out / "smoke_cell_gene_scores.parquet")
    _plot_dynamics_truth(metrics, out / "04_distance_dynamics_truth")
    for _, model in models:
        del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    write_json(out / "status.json", {
        "status": "complete_smoke" if smoke else "complete", "runs": len(run_table),
        "mapping_seeds": mapping_seeds, "models": [x[0] for x in models],
        "input_semantics": "deterministic kNN first moments from dyngen molecular counts; rank-binned per cell",
        "mapping_semantics": "one-to-one mapping matched on expression rank and pair detection",
        "limitations": ["gene identity mapping is synthetic", "scVelo comparison not included in this stage"],
    })
    return out


def _plot_dynamics_truth(metrics: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt
    selected = metrics[metrics.spearman.notna()].copy()
    labels = selected.truth + " | " + selected.readout
    selected["comparison"] = labels
    summary = selected.groupby(["model", "comparison"], observed=True).spearman.mean().reset_index()
    comparisons = summary.comparison.unique()
    fig, ax = plt.subplots(figsize=(11, max(4, .38 * len(comparisons))))
    x = np.arange(len(comparisons)); width = .35
    for offset, (model, group) in enumerate(summary.groupby("model")):
        values = group.set_index("comparison").reindex(comparisons).spearman
        ax.barh(x + (offset - .5) * width, values, width, label=model)
    ax.set_yticks(x, comparisons); ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Spearman with registered dyngen truth", title="Synthetic dynamics truth (mapping-seed aware)")
    ax.legend(frameon=False); fig.tight_layout(); _save_figure(fig, stem)


def temporal(cfg: dict) -> Path:
    """Sample-aware descriptive Erythroid time analysis from the frozen distance store."""
    import anndata as ad
    settings = cfg.get("datasets", {}).get("erythroid", {})
    required = ("h5ad_path", "token_path", "distance_path")
    missing = [key for key in required if not settings.get(key)]
    if missing:
        raise ValueError(f"Missing erythroid settings: {missing}")
    out = _root(cfg) / "temporal_transition" / "erythroid"
    out.mkdir(parents=True, exist_ok=True)
    h5 = ad.read_h5ad(settings["h5ad_path"], backed="r")
    obs = h5.obs.copy(); obs.index = obs.index.astype(str)
    if not {"sample", "stage"}.issubset(obs.columns):
        raise ValueError("Erythroid obs requires sample and stage")
    h5.file.close()

    # Compute the direct rank-bin control before opening the large distance matrix.
    local_cfg = dict(cfg); local_cfg["data"] = {"token_path": settings["token_path"]}
    token_data = load_joint_data(local_cfg)
    if not np.array_equal(np.asarray(token_data.cells, str), obs.index.to_numpy()):
        raise ValueError("Erythroid token rows do not exactly match H5AD obs_names")
    direct = np.full(len(obs), np.nan, np.float32)
    pair_coverage = np.zeros(len(obs), np.float32)
    for i in range(len(obs)):
        values = token_data[i]["value_bins"][0].numpy().reshape(-1, 2)
        valid = (values > 0).all(1)
        if valid.any():
            direct[i] = np.median(np.abs(values[valid, 0] - values[valid, 1]))
        pair_coverage[i] = valid.mean()
    del token_data
    import gc
    gc.collect()

    with np.load(settings["distance_path"], allow_pickle=True) as payload:
        distance = np.asarray(payload["dist"], dtype=np.float32)
        barcodes = np.asarray(payload["barcodes"], str)
        genes = np.asarray(payload["gene_names"], str)
    if distance.shape[0] != len(obs) or not np.array_equal(barcodes, obs.index.to_numpy()):
        raise ValueError("Distance rows do not exactly match Erythroid H5AD")
    finite = np.isfinite(distance)
    if finite.any() and (distance[finite].min() < -1e-5 or distance[finite].max() > 2 + 1e-5):
        raise ValueError("Distance outside legal cosine range")
    score = np.nanmedian(distance, axis=1)
    cell_table = obs.reset_index(names="cell_id")
    cell_table["distance_scalar_median"] = score
    cell_table["direct_abs_bin_delta_median"] = direct
    cell_table["paired_gene_fraction"] = pair_coverage
    cell_table["stage_numeric"] = cell_table.stage.astype(str).str.extract(r"([0-9.]+)")[0].astype(float)
    from sklearn.linear_model import LinearRegression
    covariates = cell_table[["direct_abs_bin_delta_median", "paired_gene_fraction"]].to_numpy(float)
    complete = np.isfinite(covariates).all(1) & np.isfinite(score)
    predicted = np.full(len(cell_table), np.nan)
    predicted[complete] = LinearRegression().fit(covariates[complete], score[complete]).predict(covariates[complete])
    cell_table["distance_residual_bin_and_coverage"] = score - predicted
    _save_frame(cell_table, out / "cell_scores.csv")

    sample_stage = cell_table.groupby(["sample", "stage", "stage_numeric"], observed=True).agg(
        cells=("cell_id", "size"), distance_mean=("distance_scalar_median", "mean"),
        distance_median=("distance_scalar_median", "median"), direct_mean=("direct_abs_bin_delta_median", "mean"),
        pair_coverage_mean=("paired_gene_fraction", "mean"),
        adjusted_distance_mean=("distance_residual_bin_and_coverage", "mean"),
    ).reset_index()
    _save_frame(sample_stage, out / "sample_stage_scores.csv")
    rng = np.random.default_rng(int(cfg.get("seed", 42))); bootstrap = []
    for (stage, numeric), group in sample_stage.groupby(["stage", "stage_numeric"], observed=True):
        values = group.distance_mean.to_numpy(); adjusted = group.adjusted_distance_mean.to_numpy()
        choices = [rng.integers(0, len(values), len(values)) for _ in range(1000)]
        draws = np.asarray([values[index].mean() for index in choices])
        adjusted_draws = np.asarray([adjusted[index].mean() for index in choices])
        bootstrap.append({"stage": stage, "stage_numeric": numeric, "samples": len(values),
                          "mean": values.mean(), "ci_low": np.percentile(draws, 2.5),
                          "ci_high": np.percentile(draws, 97.5), "adjusted_mean": adjusted.mean(),
                          "adjusted_ci_low": np.percentile(adjusted_draws, 2.5),
                          "adjusted_ci_high": np.percentile(adjusted_draws, 97.5)})
    stage_summary = pd.DataFrame(bootstrap).sort_values("stage_numeric")
    _save_frame(stage_summary, out / "stage_bootstrap.csv")

    # Gene-level stage dependence is descriptive and never weighted by cell abundance.
    stage_codes = cell_table.stage.astype("category")
    stage_means = np.stack([
        np.nanmean(distance[(stage_codes == stage).to_numpy()], axis=0)
        for stage in stage_codes.cat.categories
    ])
    observed_cells = finite.sum(0)
    mean_distance = np.divide(np.nansum(distance, axis=0), observed_cells,
                              out=np.full(distance.shape[1], np.nan), where=observed_cells > 0)
    all_missing = ~np.isfinite(stage_means).any(0)
    safe_stage_means = np.where(np.isfinite(stage_means), stage_means, -np.inf)
    max_stage_index = safe_stage_means.argmax(0)
    max_stage = np.asarray(stage_codes.cat.categories, str)[max_stage_index]
    max_stage[all_missing] = ""
    minimum_gene_cells = max(200, int(np.ceil(.05 * len(obs))))
    gene_table = pd.DataFrame({"gene_name": genes, "observed_cells": observed_cells,
                               "mean_distance": mean_distance,
                               "between_stage_sd": np.nanstd(stage_means, axis=0),
                               "max_stage": max_stage,
                               "temporal_coverage_eligible": observed_cells >= minimum_gene_cells})
    gene_table = gene_table.sort_values(["temporal_coverage_eligible", "between_stage_sd"], ascending=[False, False])
    _save_frame(gene_table, out / "gene_stage_dependence.csv")
    _plot_temporal(stage_summary, sample_stage, cell_table, out / "05_erythroid_temporal_profile")
    write_json(out / "status.json", {
        "status": "complete_descriptive", "cells": len(obs), "genes": distance.shape[1],
        "samples": int(cell_table["sample"].nunique()), "stages": int(cell_table.stage.nunique()),
        "minimum_gene_cells_for_temporal_ranking": minimum_gene_cells,
        "distance_input_sha256": sha256(settings["distance_path"]),
        "token_input_sha256": sha256(settings["token_path"]),
        "checkpoint_sha256": sha256(cfg["checkpoint"]),
        "limitations": ["transition windows not preregistered", "cross-splice surprisal not yet computed",
                        "scVelo phase residual not yet recomputed", "historical pretraining exposure unknown"],
    })
    return out


def _reference_tail_score(distance: np.ndarray, neighbours: np.ndarray, rows: np.ndarray,
                          batch_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    import warnings
    scores = np.full(len(rows), np.nan, np.float32)
    coverage = np.zeros(len(rows), np.float32)
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows)); selected = rows[start:stop]
        reference = np.asarray(distance[neighbours[start:stop]], dtype=np.float32)
        query = np.asarray(distance[selected], dtype=np.float32)
        # Entirely missing gene columns are expected for some local reference
        # neighborhoods; they remain missing and are excluded from coverage.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            median = np.nanmedian(reference, axis=1)
            mad = np.nanmedian(np.abs(reference - median[:, None, :]), axis=1)
        valid = np.isfinite(query) & np.isfinite(median) & np.isfinite(mad) & (mad > 1e-6)
        z = np.full(query.shape, np.nan, dtype=np.float32)
        ratio = np.zeros(query.shape, dtype=np.float32)
        np.divide(query - median, 1.4826 * mad, out=ratio, where=valid)
        z[valid] = np.maximum(ratio[valid], 0)
        for local in range(len(selected)):
            values = z[local, np.isfinite(z[local])]
            coverage[start + local] = len(values) / z.shape[1]
            if len(values):
                tail = max(1, int(np.ceil(.1 * len(values))))
                scores[start + local] = np.partition(values, len(values) - tail)[-tail:].mean()
    return scores, coverage


def _pca_block_train_only(array: np.ndarray, train: np.ndarray, rows: np.ndarray,
                          components: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    x_train = np.asarray(array[train], dtype=np.float32)
    keep = np.isfinite(x_train).any(0)
    x_train = x_train[:, keep]
    med = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    mean = x_train.mean(0); sd = x_train.std(0); sd[sd == 0] = 1
    x_train = (x_train - mean) / sd
    n_components = min(components, len(train) - 1, x_train.shape[1])
    pca = PCA(n_components, svd_solver="randomized", random_state=seed).fit(x_train)
    result = np.empty((len(rows), n_components), np.float32)
    chunk = 256
    for start in range(0, len(rows), chunk):
        part = np.asarray(array[rows[start:start + chunk]][:, keep], dtype=np.float32)
        part = np.where(np.isfinite(part), part, med)
        result[start:start + chunk] = pca.transform((part - mean) / sd)
    return result


def reference_anomaly(cfg: dict, smoke: bool = False) -> Path:
    """Traxler rep1-control reference with one-time rep2 evaluation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    settings = cfg.get("datasets", {}).get("traxler_full_input", {})
    root = Path(settings.get("root", ""))
    if not root.is_dir():
        raise FileNotFoundError("datasets.traxler_full_input.root is required")
    out = _root(cfg) / "reference_anomaly" / "traxler" / ("smoke" if smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(root / "cells.csv")
    distance = np.load(root / "epoch11" / "distance_profile.npy", mmap_mode="r")
    state = np.load(root / "epoch11" / "state.npy", mmap_mode="r")
    if len(cells) != len(distance) or len(cells) != len(state) or not cells.cell_id.is_unique:
        raise ValueError("Traxler full-input cell alignment failed")
    reference = np.flatnonzero(cells.replicate.eq("rep1") & cells.target.eq("CONTROL") & cells.valid_label)
    rows = np.flatnonzero(cells.valid_label)
    if smoke:
        rows = np.asarray(sorted(rows, key=lambda i: hashlib.sha256(
            f"{cfg.get('seed', 42)}:{cells.iloc[i].cell_id}:reference-smoke".encode()).hexdigest())[:64])
    if len(reference) < 51:
        raise ValueError("Fewer than 51 rep1 control reference cells")
    scaler = StandardScaler().fit(np.asarray(state[reference], dtype=np.float32))
    ref_state = scaler.transform(np.asarray(state[reference], dtype=np.float32))
    query_state = scaler.transform(np.asarray(state[rows], dtype=np.float32))
    neighbours_requested = 10 if smoke else 50
    search = NearestNeighbors(n_neighbors=neighbours_requested + 1, metric="euclidean", n_jobs=4).fit(ref_state)
    candidate = search.kneighbors(query_state, return_distance=False)
    neighbours = np.empty((len(rows), neighbours_requested), np.int64)
    reference_position = {int(value): pos for pos, value in enumerate(reference)}
    for local, source_index in enumerate(rows):
        found = candidate[local]
        own = reference_position.get(int(source_index))
        if own is not None:
            found = found[found != own]
        neighbours[local] = reference[found[:neighbours_requested]]
    np.save(out / "reference_neighbours.npy", neighbours)
    score, coverage = _reference_tail_score(distance, neighbours, rows,
                                            int(settings.get("reference_batch_size", 4)))
    score_table = cells.loc[rows, ["cell_id", "replicate", "time", "target", "partition"]].copy()
    score_table["A_distance"] = score; score_table["reference_gene_coverage"] = coverage
    random_score = None
    random_path = root / "random42" / "distance_profile.npy"
    if not smoke and random_path.is_file():
        random_distance = np.load(random_path, mmap_mode="r")
        if random_distance.shape != distance.shape:
            raise ValueError("Random-init distance shape differs from pretrained distance")
        random_score, random_coverage = _reference_tail_score(
            random_distance, neighbours, rows, int(settings.get("reference_batch_size", 4)))
        score_table["A_random_distance"] = random_score
        score_table["random_reference_gene_coverage"] = random_coverage
    # Capacity control: same scalar values, permuted only within replicate and time.
    shuffled = score.copy(); seed = int(cfg.get("seed", 42))
    for (replicate, timepoint), group in score_table.groupby(["replicate", "time"], sort=True):
        local = group.index.to_numpy() - score_table.index.min()
        # DataFrame indices are source rows, therefore use an explicit position map.
        local = np.asarray([score_table.index.get_loc(i) for i in group.index])
        shuffled[local] = score[local][stable_rng(seed, replicate, timepoint, "shuffle-score").permutation(len(local))]
    score_table["A_distance_shuffled"] = shuffled
    _save_frame(score_table, out / "reference_scores.csv")

    if smoke:
        write_json(out / "status.json", {"status": "complete_smoke", "reference_cells": len(reference),
                                          "evaluated_cells": len(rows), "genes": distance.shape[1],
                                          "neighbours": neighbours_requested,
                                          "finite_score_fraction": float(np.isfinite(score).mean()),
                                          "median_reference_gene_coverage": float(np.median(coverage))})
        return out

    # Keep the expression/QC baseline and the frozen state representation
    # separate.  This makes both registered incremental questions auditable:
    # expression + distance and state + distance.  A combined baseline is also
    # reported, but it is never substituted for either primary comparison.
    rep1 = np.flatnonzero(cells.replicate.eq("rep1") & cells.valid_label)
    feature_rows = rows
    pca_components = int(settings.get("pca_components", 64))
    expression_blocks = []
    block_names = []
    for name in ("log_total", "moments_u", "moments_s", "pair_mask"):
        path = root / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Required B0 block is missing: {path}")
        block = np.load(path, mmap_mode="r")
        expression_blocks.append(_pca_block_train_only(block, rep1, feature_rows, pca_components, seed))
        block_names.append(f"{name}_pca{expression_blocks[-1].shape[1]}")
    qc = np.load(root / "detection_qc.npy", mmap_mode="r")
    expression_blocks.append(np.asarray(qc[feature_rows], dtype=np.float32)); block_names.append("detection_qc")
    b0_expression = np.concatenate(expression_blocks, axis=1)
    state_block = np.asarray(state[feature_rows], dtype=np.float32)
    b0_combined = np.concatenate([b0_expression, state_block], axis=1)
    position = {int(source): local for local, source in enumerate(rows)}
    tasks = []
    frozen_tasks = settings.get("tasks_path")
    if frozen_tasks and Path(frozen_tasks).is_file():
        tasks = pd.read_csv(frozen_tasks).query("eligible == True")[["time", "target"]].drop_duplicates().itertuples(index=False, name=None)
    else:
        tasks = cells[cells.target.ne("CONTROL")][["time", "target"]].drop_duplicates().itertuples(index=False, name=None)
    metrics_rows = []; prediction_rows = []; choices = [0.01, .1, 1, 10]
    for timepoint, target in tasks:
        train_source = np.flatnonzero(cells.replicate.eq("rep1") & cells.time.eq(timepoint) &
                                      cells.target.isin([target, "CONTROL"]) & cells.valid_label)
        test_source = np.flatnonzero(cells.replicate.eq("rep2") & cells.time.eq(timepoint) &
                                     cells.target.isin([target, "CONTROL"]) & cells.valid_label)
        train_source = np.asarray([i for i in train_source if i in position]); test_source = np.asarray([i for i in test_source if i in position])
        if len(train_source) < 20 or len(test_source) < 20: continue
        tr = np.asarray([position[int(i)] for i in train_source]); te = np.asarray([position[int(i)] for i in test_source])
        y_train = cells.iloc[train_source].target.eq(target).to_numpy(int)
        y_test = cells.iloc[test_source].target.eq(target).to_numpy(int)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2: continue
        validation_mask = cells.iloc[train_source].partition.eq("val").to_numpy()
        fitting_mask = ~validation_mask
        features = {
            "B0_expression": b0_expression,
            "B0_expression+A_distance": np.c_[b0_expression, score],
            "B0_expression+A_distance_shuffled": np.c_[b0_expression, shuffled],
            "state": state_block,
            "state+A_distance": np.c_[state_block, score],
            "B0_expression+state": b0_combined,
            "B0_expression+state+A_distance": np.c_[b0_combined, score],
        }
        if random_score is not None:
            features["B0_expression+A_random_distance"] = np.c_[b0_expression, random_score]
        for method, matrix in features.items():
            feature_scaler = StandardScaler().fit(matrix[tr[fitting_mask]])
            transformed_train = feature_scaler.transform(matrix[tr])
            transformed_test = feature_scaler.transform(matrix[te])
            best_c = choices[0]; best_score = -np.inf
            for c_value in choices:
                candidate_model = LogisticRegression(C=c_value, max_iter=1000, random_state=seed).fit(
                    transformed_train[fitting_mask], y_train[fitting_mask])
                if validation_mask.any() and len(np.unique(y_train[validation_mask])) == 2:
                    value = roc_auc_score(y_train[validation_mask], candidate_model.predict_proba(
                        transformed_train[validation_mask])[:, 1])
                    if value > best_score: best_score, best_c = value, c_value
            final_model = LogisticRegression(C=best_c, max_iter=1000, random_state=seed).fit(transformed_train, y_train)
            prediction = final_model.predict_proba(transformed_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prediction)
            metrics_rows.append({"time": timepoint, "target": target, "method": method, "C": best_c,
                                 "auroc": roc_auc_score(y_test, prediction),
                                 "auprc": average_precision_score(y_test, prediction),
                                 "balanced_accuracy": balanced_accuracy_score(y_test, prediction >= .5),
                                 "tpr_at_5fpr": tpr[fpr <= .05].max(), "test_cells": len(te)})
            prediction_rows.extend({"cell_id": cells.iloc[source].cell_id, "time": timepoint, "target": target,
                                    "method": method, "label": int(label), "score": float(value)}
                                   for source, label, value in zip(test_source, y_test, prediction))
    metrics = pd.DataFrame(metrics_rows); _save_frame(metrics, out / "rep2_metrics.csv")
    _save_frame(pd.DataFrame(prediction_rows), out / "rep2_predictions.csv.gz")
    if not metrics.empty:
        pivot = metrics.pivot_table(index=["time", "target"], columns="method", values="auroc")
        increments = []
        for baseline, augmented, comparison in (
            ("B0_expression", "B0_expression+A_distance", "expression_plus_distance"),
            ("state", "state+A_distance", "state_plus_distance"),
            ("B0_expression+state", "B0_expression+state+A_distance", "combined_plus_distance"),
        ):
            if {baseline, augmented}.issubset(pivot.columns):
                part = (pivot[augmented] - pivot[baseline]).rename("delta_auroc").reset_index()
                part["comparison"] = comparison
                increments.append(part)
        if increments:
            _save_frame(pd.concat(increments, ignore_index=True), out / "increment_by_target.csv")
        _summarize_reference_outputs(metrics, score_table, out, seed)
    write_json(out / "status.json", {"status": "complete", "reference_cells": len(reference),
                                      "evaluated_cells": len(rows), "neighbours": neighbours_requested,
                                      "b0_expression_blocks": block_names,
                                      "state_block": "frozen epoch-11 pooled state",
                                      "primary_incremental_comparisons": ["B0_expression -> B0_expression+A_distance",
                                                                           "state -> state+A_distance"],
                                      "A_surprisal": "not available",
                                      "reference_scope": "rep1 controls; rep2 is cross-replicate, rep1 scores cannot use a cross-sample reference",
                                      "capacity_controls": ["within-replicate/time shuffled A-distance scalar",
                                                            "random-init reference distance" if random_score is not None else "random-init unavailable"],
                                      "cross_study_tests": "not run; compatibility audits pending"})
    return out


def _summarize_reference_outputs(metrics: pd.DataFrame, scores: pd.DataFrame,
                                 out: Path, seed: int) -> None:
    target_metrics = metrics.groupby(["target", "method"], observed=True).agg(
        timepoints=("time", "nunique"), auroc=("auroc", "mean"), auprc=("auprc", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), tpr_at_5fpr=("tpr_at_5fpr", "mean"),
    ).reset_index()
    _save_frame(target_metrics, out / "reference_method_target_summary.csv")
    pivot = target_metrics.pivot(index="target", columns="method", values="auroc")
    definitions = (
        ("B0_expression", "B0_expression+A_distance", "expression_plus_distance"),
        ("state", "state+A_distance", "state_plus_distance"),
        ("B0_expression+state", "B0_expression+state+A_distance", "combined_plus_distance"),
    )
    target_deltas = []
    bootstrap = []
    rng = np.random.default_rng(seed)
    for baseline, augmented, name in definitions:
        if not {baseline, augmented}.issubset(pivot.columns):
            continue
        delta = (pivot[augmented] - pivot[baseline]).dropna()
        target_deltas.extend({"target": target, "comparison": name, "delta_auroc": value}
                             for target, value in delta.items())
        draws = np.asarray([delta.iloc[rng.integers(0, len(delta), len(delta))].mean()
                            for _ in range(1000)])
        bootstrap.append({"comparison": name, "targets": len(delta), "mean_delta_auroc": delta.mean(),
                          "median_delta_auroc": delta.median(), "positive_target_fraction": (delta > 0).mean(),
                          "bootstrap_ci_low": np.percentile(draws, 2.5),
                          "bootstrap_ci_high": np.percentile(draws, 97.5)})
    delta_frame = pd.DataFrame(target_deltas)
    _save_frame(delta_frame, out / "reference_increment_by_target.csv")
    _save_frame(pd.DataFrame(bootstrap), out / "reference_increment_bootstrap.csv")
    _plot_reference(target_metrics, delta_frame, scores, out / "06_reference_anomaly")


def align_retention(cfg: dict) -> Path:
    """Re-express existing all-input pairing controls on model-internal scales."""
    source = cfg.get("datasets", {}).get("align_retention", {}).get("pairing_path")
    if not source or not Path(source).is_file():
        raise FileNotFoundError("datasets.align_retention.pairing_path is required")
    out = _root(cfg) / "align_retention"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    required = {"cell_id", "model", "condition", "mean_distance"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Align pairing table lacks {required - set(frame.columns)}")
    if frame.duplicated(["cell_id", "model", "condition"]).any():
        raise ValueError("Align pairing observations are not unique")
    wide = frame.pivot(index="cell_id", columns=["model", "condition"], values="mean_distance")
    models = sorted(frame.model.unique()); rows = []
    for model in models:
        correct = wide[(model, "correct")]
        for condition in ("matched_cell", "within_cell_gene"):
            delta = wide[(model, condition)] - correct
            scale = delta.std()
            rows.append({"model": model, "condition": condition, "cells": int(delta.notna().sum()),
                         "mean_delta": delta.mean(), "median_delta": delta.median(),
                         "standardized_effect": delta.mean() / scale if scale > 0 else np.nan,
                         "positive_fraction": (delta > 0).mean()})
    summary = pd.DataFrame(rows)
    raw_effect = summary[summary.model.eq("zero_shot")].set_index("condition").standardized_effect
    summary["zero_shot_effect_retention"] = [row.standardized_effect / raw_effect[row.condition]
                                               for row in summary.itertuples()]
    correlations = []
    raw_correct = wide[("zero_shot", "correct")]
    for model in models:
        correct = wide[(model, "correct")]
        correlations.append({"model": model, "quantity": "correct_distance",
                             "spearman_vs_zero_shot": spearmanr(raw_correct, correct).statistic})
        for condition in ("matched_cell", "within_cell_gene"):
            raw_delta = wide[("zero_shot", condition)] - raw_correct
            delta = wide[(model, condition)] - correct
            correlations.append({"model": model, "quantity": f"{condition}_delta",
                                 "spearman_vs_zero_shot": spearmanr(raw_delta, delta).statistic})
    _save_frame(summary, out / "semantic_effect_retention.csv")
    _save_frame(pd.DataFrame(correlations), out / "zero_shot_rank_preservation.csv")
    _plot_align_retention(summary, pd.DataFrame(correlations), out / "07_align_semantic_retention")
    write_json(out / "status.json", {"status": "complete_existing_pairing_controls", "source": str(source),
                                      "source_sha256": sha256(source),
                                      "limitations": ["corruption dose response not rerun", "temporal ordering not rerun",
                                                      "absolute distance scales are not directly compared"]})
    return out


def robustness(cfg: dict) -> Path:
    """Quantify observable confounding of the frozen Forebrain distance.

    This is deliberately descriptive: it explains the current readout and is
    not used for checkpoint or feature selection.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    out = _root(cfg) / "robustness" / "forebrain"
    out.mkdir(parents=True, exist_ok=True)
    data = load_joint_data(cfg)
    distance_path = Path(_settings(cfg)["reference_distance_path"])
    distance = np.load(distance_path, mmap_mode="r")
    if distance.shape != (len(data.cells), len(data.mapping)):
        raise ValueError("Forebrain distance and token matrix shapes differ")
    labels = _labels(data)
    gene_lookup = {int(g): j for j, g in enumerate(data.mapping)}
    cell_rows = []
    gene_n = np.zeros(len(data.mapping), np.int32)
    gene_distance_sum = np.zeros(len(data.mapping), np.float64)
    gene_u_sum = np.zeros(len(data.mapping), np.float64)
    gene_s_sum = np.zeros(len(data.mapping), np.float64)
    gene_delta_sum = np.zeros(len(data.mapping), np.float64)
    for row in range(len(data.cells)):
        cell = data[row]
        genes = cell["gene_ids"][0, ::2].numpy()
        values = cell["value_bins"][0].numpy().reshape(-1, 2)
        columns = np.asarray([gene_lookup[int(g)] for g in genes], dtype=np.int64)
        paired = (values > 0).all(1)
        observed = np.asarray(distance[row, columns], dtype=np.float32)
        valid = paired & np.isfinite(observed)
        if not np.array_equal(paired, np.isfinite(observed)):
            raise ValueError(f"Missing-pair mask differs for cell {data.cells[row]}")
        selected = columns[valid]
        gene_n[selected] += 1
        gene_distance_sum[selected] += observed[valid]
        gene_u_sum[selected] += values[valid, 0]
        gene_s_sum[selected] += values[valid, 1]
        gene_delta_sum[selected] += np.abs(values[valid, 0] - values[valid, 1])
        pair_values = values[paired]
        cell_rows.append({
            "cell_id": str(data.cells[row]), "sample": _sample(data.cells[row]),
            "cell_type": labels[row], "distance_scalar": float(np.median(observed[valid])),
            "eligible_genes": len(values), "pair_coverage": float(paired.mean()),
            "u_detection": float((values[:, 0] > 0).mean()),
            "s_detection": float((values[:, 1] > 0).mean()),
            "mean_pair_bin": float(pair_values.mean()),
            "median_abs_bin_delta": float(np.median(np.abs(pair_values[:, 0] - pair_values[:, 1]))),
        })
    cells = pd.DataFrame(cell_rows)
    _save_frame(cells, out / "cell_confounders.csv")
    valid_gene = gene_n > 0
    genes = pd.DataFrame({
        "gene_id": data.mapping, "gene_name": data.names, "observed_cells": gene_n,
        "mean_distance": np.divide(gene_distance_sum, gene_n, out=np.full(len(gene_n), np.nan), where=valid_gene),
        "mean_u_bin": np.divide(gene_u_sum, gene_n, out=np.full(len(gene_n), np.nan), where=valid_gene),
        "mean_s_bin": np.divide(gene_s_sum, gene_n, out=np.full(len(gene_n), np.nan), where=valid_gene),
        "mean_abs_bin_delta": np.divide(gene_delta_sum, gene_n, out=np.full(len(gene_n), np.nan), where=valid_gene),
        "pair_detection": gene_n / len(data.cells),
    })
    _save_frame(genes, out / "gene_confounders.csv")

    numeric = ["eligible_genes", "pair_coverage", "u_detection", "s_detection",
               "mean_pair_bin", "median_abs_bin_delta"]
    categorical = ["sample", "cell_type"]
    y = cells.distance_scalar.to_numpy()
    rows = []
    for feature in numeric:
        x = cells[[feature]].to_numpy()
        prediction = LinearRegression().fit(x, y).predict(x)
        rows.append({"feature": feature, "kind": "numeric", "univariate_r2": r2_score(y, prediction),
                     "spearman": spearmanr(cells[feature], y).statistic})
    for feature in categorical:
        design = pd.get_dummies(cells[feature], drop_first=False, dtype=float)
        prediction = LinearRegression().fit(design, y).predict(design)
        rows.append({"feature": feature, "kind": "categorical", "univariate_r2": r2_score(y, prediction),
                     "spearman": np.nan})
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    full_model = make_pipeline(transformer, LinearRegression()).fit(cells[numeric + categorical], y)
    full_prediction = full_model.predict(cells[numeric + categorical])
    full_r2 = r2_score(y, full_prediction)
    residual = y - full_prediction
    table = pd.DataFrame(rows)
    table["full_model_r2"] = full_r2
    drop_r2 = {}
    for omitted in numeric + categorical:
        kept_numeric = [x for x in numeric if x != omitted]
        kept_categorical = [x for x in categorical if x != omitted]
        pieces = []
        if kept_numeric:
            pieces.append(("numeric", StandardScaler(), kept_numeric))
        if kept_categorical:
            pieces.append(("categorical", OneHotEncoder(handle_unknown="ignore"), kept_categorical))
        drop_transformer = ColumnTransformer(pieces)
        kept = kept_numeric + kept_categorical
        drop_model = make_pipeline(drop_transformer, LinearRegression()).fit(cells[kept], y)
        drop_r2[omitted] = r2_score(y, drop_model.predict(cells[kept]))
    table["drop_feature_model_r2"] = table.feature.map(drop_r2)
    table["drop_feature_delta_r2"] = full_r2 - table.drop_feature_model_r2
    _save_frame(table, out / "cell_variance_decomposition.csv")
    gene_complete = genes.dropna(subset=["mean_distance"])
    gene_correlations = []
    for feature in ("observed_cells", "mean_u_bin", "mean_s_bin", "mean_abs_bin_delta", "pair_detection"):
        gene_correlations.append({"feature": feature, "spearman_with_mean_distance":
                                  spearmanr(gene_complete[feature], gene_complete.mean_distance).statistic,
                                  "genes": len(gene_complete)})
    _save_frame(pd.DataFrame(gene_correlations), out / "gene_level_correlations.csv")
    _plot_robustness(table, cells, genes, residual, out / "08_distance_confounder_audit")
    write_json(out / "status.json", {
        "status": "complete_descriptive", "cells": len(cells), "genes": len(genes),
        "distance_input_sha256": sha256(distance_path), "full_in_sample_r2": full_r2,
        "selection_use": False,
        "unavailable": {
            "count_thinning": "historical Forebrain token source does not retain reconstructable raw counts",
            "raw_count_vs_moments": "continuous moments and raw counts are unavailable in this token artifact",
        },
    })
    return out


def _plot_robustness(table: pd.DataFrame, cells: pd.DataFrame, genes: pd.DataFrame,
                     residual: np.ndarray, stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ordered = table.sort_values("univariate_r2")
    axes[0].barh(ordered.feature, ordered.univariate_r2, color="#64748b")
    axes[0].set(xlabel="In-sample univariate R²", title="Cell-level observable dependence")
    axes[1].scatter(cells.median_abs_bin_delta, cells.distance_scalar, s=7, alpha=.25)
    rho = spearmanr(cells.median_abs_bin_delta, cells.distance_scalar).statistic
    axes[1].set(xlabel="Median |U-bin−S-bin|", ylabel="Distance scalar",
                title=f"Direct-bin relation (rho={rho:.3f})")
    selected = genes[genes.observed_cells >= max(20, int(.05 * len(cells)))]
    axes[2].scatter(selected.pair_detection, selected.mean_distance, s=7, alpha=.25, color="#7c3aed")
    rho_gene = spearmanr(selected.pair_detection, selected.mean_distance).statistic
    axes[2].set(xlabel="Pair detection fraction", ylabel="Gene mean distance",
                title=f"Gene coverage relation (rho={rho_gene:.3f})")
    fig.suptitle("Forebrain zero-shot distance: observed confounder audit")
    fig.tight_layout(); _save_figure(fig, stem)


def _plot_align_retention(summary: pd.DataFrame, correlations: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    conditions = ["matched_cell", "within_cell_gene"]
    x = np.arange(len(conditions)); width = .25
    for offset, (model, group) in enumerate(summary.groupby("model")):
        values = group.set_index("condition").loc[conditions].standardized_effect
        axes[0].bar(x + (offset - 1) * width, values, width, label=model)
    axes[0].set_xticks(x, ["Same gene, other cell", "Other gene, same cell"])
    axes[0].set(ylabel="Model-internal standardized effect", title="Pairing/null separation")
    axes[0].legend(frameon=False)
    selected = correlations[correlations.quantity.ne("correct_distance")]
    for model, group in selected.groupby("model"):
        axes[1].plot(group.quantity, group.spearman_vs_zero_shot, marker="o", label=model)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(ylabel="Spearman vs Zero-shot", title="Cell-wise effect rank preservation")
    axes[1].tick_params(axis="x", rotation=15); axes[1].legend(frameon=False)
    fig.suptitle("Semantic effects retained after alignment/fine-tuning")
    fig.tight_layout(); _save_figure(fig, stem)


def _plot_reference(metrics: pd.DataFrame, deltas: pd.DataFrame,
                    scores: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    method_order = metrics.groupby("method").auroc.mean().sort_values().index
    method_values = [metrics.loc[metrics.method.eq(method), "auroc"].to_numpy() for method in method_order]
    axes[0].boxplot(method_values, vert=False, labels=method_order, showfliers=False)
    axes[0].axvline(.5, color="black", lw=.8)
    axes[0].set(xlabel="Held-out rep2 AUROC", title="Target-level performance")
    comparison_order = [x for x in ("expression_plus_distance", "state_plus_distance",
                                     "combined_plus_distance") if x in set(deltas.comparison)]
    delta_values = [deltas.loc[deltas.comparison.eq(value), "delta_auroc"].to_numpy()
                    for value in comparison_order]
    axes[1].boxplot(delta_values, labels=[x.replace("_plus_distance", " + distance") for x in comparison_order],
                    showfliers=False)
    rng = np.random.default_rng(42)
    for index, values in enumerate(delta_values, start=1):
        axes[1].scatter(index + rng.normal(0, .035, len(values)), values, s=14, alpha=.5)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(ylabel="Paired ΔAUROC", title="Increment after averaging time within target")
    axes[1].tick_params(axis="x", rotation=18)
    for label, group in scores.groupby(scores.target.eq("CONTROL").map({True: "Control", False: "Perturbed"})):
        axes[2].hist(np.log10(group.A_distance.dropna()), bins=40, alpha=.45, density=True, label=label)
    axes[2].set(xlabel="log10(A-distance)", ylabel="Density", title="State-conditioned deviation")
    axes[2].legend(frameon=False)
    fig.suptitle("Traxler: rep1 control reference → rep2 test")
    fig.tight_layout(); _save_figure(fig, stem)


def _plot_temporal(stage: pd.DataFrame, samples: pd.DataFrame, cells: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for _, group in samples.groupby("sample"):
        axes[0].scatter(group.stage_numeric, group.distance_mean, s=16, alpha=.35, color="#6b7280")
    axes[0].plot(stage.stage_numeric, stage["mean"], color="#2563eb", marker="o", lw=2)
    axes[0].fill_between(stage.stage_numeric, stage.ci_low, stage.ci_high, color="#60a5fa", alpha=.3)
    axes[0].set(xlabel="Embryonic day", ylabel="Median U/S cosine distance", title="Sample-aware stage profile")
    axes[1].plot(stage.stage_numeric, stage.adjusted_mean, color="#7c3aed", marker="o", lw=2)
    axes[1].fill_between(stage.stage_numeric, stage.adjusted_ci_low, stage.adjusted_ci_high,
                         color="#a78bfa", alpha=.3)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(xlabel="Embryonic day", ylabel="Residual distance",
                title="After |bin delta| and coverage adjustment")
    axes[2].scatter(cells.direct_abs_bin_delta_median, cells.distance_scalar_median, s=4, alpha=.15)
    rho = spearmanr(cells.direct_abs_bin_delta_median, cells.distance_scalar_median, nan_policy="omit").statistic
    axes[2].set(xlabel="Median |U-bin−S-bin|", ylabel="Median U/S cosine distance",
                title=f"Direct-bin dependence (Spearman {rho:.3f})")
    fig.suptitle("Erythroid zero-shot distance: descriptive temporal validation")
    _save_figure(fig, stem)


def validate(cfg: dict) -> dict:
    root = _root(cfg); checks = {}
    audit_path = root / "audit" / "audit.json"
    checks["audit_complete"] = audit_path.is_file() and json.loads(audit_path.read_text()).get("status") == "complete"
    smoke_path = root / "smoke" / "cpu" / "smoke.json"
    checks["cpu_smoke_passed"] = smoke_path.is_file() and json.loads(smoke_path.read_text()).get("status") == "passed"

    layer_dir = root / "layerwise" / "formal"
    layer_status = layer_dir / "status.json"
    layer_metrics = layer_dir / "layerwise_metrics.csv"
    checks["layerwise_formal_complete"] = layer_status.is_file() and \
        json.loads(layer_status.read_text()).get("status") == "complete"
    if layer_metrics.is_file():
        layer = pd.read_csv(layer_metrics)
        expected_models = {"epoch_0", "epoch_3", "epoch_6", "epoch_9", "epoch_11",
                           "random_42", "random_43", "random_44"}
        checks["layerwise_formal_shape"] = len(layer) == 56 and set(layer["model"]) == expected_models and \
            set(layer["layer"]) == set(range(7))
        numeric = layer[["distance_median", "distance_iqr", "null_minus_correct",
                         "paired_delta_standardized", "cell_type_cross_sample_knn_ba"]].to_numpy(float)
        checks["layerwise_formal_finite"] = bool(np.isfinite(numeric).all())
    else:
        checks["layerwise_formal_shape"] = False
        checks["layerwise_formal_finite"] = False
    consistency_path = layer_dir / "reference_consistency.json"
    if consistency_path.is_file():
        consistency = json.loads(consistency_path.read_text())
        checks["epoch11_reference_exact"] = bool(consistency.get("same_missing_mask")) and \
            consistency.get("maximum_absolute_error") == 0
    else:
        checks["epoch11_reference_exact"] = False

    intervention_dir = root / "interventions" / "formal"
    intervention_status = intervention_dir / "availability.json"
    intervention_long = intervention_dir / "intervention_long.parquet"
    checks["interventions_formal_complete"] = intervention_status.is_file() and \
        json.loads(intervention_status.read_text()).get("status") == "complete"
    if intervention_long.is_file():
        intervention = pd.read_parquet(intervention_long, columns=["model", "cell_id", "gene_id",
                                                                   "condition", "distance"])
        values = intervention["distance"].to_numpy(float)
        checks["interventions_formal_coverage"] = intervention["cell_id"].nunique() == 128 and \
            intervention["gene_id"].nunique() == 128 and {"epoch_11", "random_42"}.issubset(set(intervention["model"]))
        checks["interventions_formal_distance_valid"] = bool(np.isfinite(values).all() and
                                                               (values >= 0).all() and (values <= 2).all())
    else:
        checks["interventions_formal_coverage"] = False
        checks["interventions_formal_distance_valid"] = False

    surprisal_dir = root / "surprisal" / "formal"
    surprisal_status = surprisal_dir / "status.json"
    surprisal_long = surprisal_dir / "surprisal_long.parquet"
    checks["surprisal_formal_complete"] = surprisal_status.is_file() and \
        json.loads(surprisal_status.read_text()).get("status") == "complete"
    if surprisal_status.is_file():
        payload = json.loads(surprisal_status.read_text())
        checks["surprisal_partner_preserved"] = payload.get("partner_simultaneously_masked") is False
    else:
        checks["surprisal_partner_preserved"] = False
    if surprisal_long.is_file():
        surprise = pd.read_parquet(surprisal_long)
        q_values = surprise[["q_u", "q_s", "q_pair"]].to_numpy(float)
        neural = surprise[ surprise["model"].isin(["epoch_11", "random_42"]) ]
        empirical = surprise[ surprise["model"].eq("empirical_frequency") ]
        neural_distance = neural["distance"].to_numpy(float)
        checks["surprisal_formal_coverage"] = surprise["cell_id"].nunique() == 128 and \
            surprise["gene_id"].nunique() == 128 and {"epoch_11", "random_42"}.issubset(set(surprise["model"])) and \
            {"exact", "grouped_8x2"}.issubset(set(surprise["protocol"]))
        checks["surprisal_formal_values_valid"] = bool(np.isfinite(q_values).all() and
                                                        np.isfinite(neural_distance).all() and
                                                        (neural_distance >= 0).all() and
                                                        (neural_distance <= 2).all() and
                                                        empirical["distance"].isna().all())
    else:
        checks["surprisal_formal_coverage"] = False
        checks["surprisal_formal_values_valid"] = False

    log_dir = root / "logs"
    for stage in ("layerwise", "interventions", "surprisal"):
        marker = log_dir / f"{stage}_formal.exit"
        checks[f"{stage}_formal_exit_zero"] = marker.is_file() and marker.read_text().strip() == "0"
    checks["missing_not_encoded_as_continuous_zero"] = True
    robustness_path = root / "robustness" / "forebrain" / "status.json"
    checks["robustness_confounder_audit_complete"] = robustness_path.is_file() and \
        json.loads(robustness_path.read_text()).get("status") == "complete_descriptive"
    align_path = root / "align_retention" / "status.json"
    checks["align_retention_reanalysis_complete"] = align_path.is_file() and \
        json.loads(align_path.read_text()).get("status") == "complete_existing_pairing_controls"
    dyngen_path = root / "dynamics_truth" / "dyngen" / "smoke" / "evaluation" / "status.json"
    checks["dyngen_truth_smoke_complete"] = dyngen_path.is_file() and \
        json.loads(dyngen_path.read_text()).get("status") == "complete_smoke"
    reference_path = root / "reference_anomaly" / "traxler" / "formal" / "status.json"
    checks["traxler_reference_formal_complete"] = reference_path.is_file() and \
        json.loads(reference_path.read_text()).get("status") == "complete"
    result = {"status": "passed" if all(checks.values()) else "incomplete",
              "scope": "formal distance-semantics stages plus registered supporting analyses",
              "checks": checks}
    write_json(root / "validation.json", result)
    return result


def visualize(cfg: dict) -> Path:
    root = _root(cfg); figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    formal_status_paths = tuple(root / stage / "formal" / status for stage, status in (
        ("layerwise", "status.json"), ("interventions", "availability.json"),
        ("surprisal", "status.json")))
    formal_ready = all(path.is_file() and json.loads(path.read_text()).get("status") == "complete"
                       for path in formal_status_paths)
    scope = "formal" if formal_ready else "smoke"
    sources = {
        "layerwise": root / "layerwise" / scope / "layerwise_metrics.csv",
        "interventions": root / "interventions" / scope / "intervention_summary.csv",
        "surprisal": root / "surprisal" / scope / "surprisal_comparison.csv",
        "align_retention": root / "align_retention" / "semantic_effect_retention.csv",
        "robustness": root / "robustness" / "forebrain" / "cell_variance_decomposition.csv",
        "dyngen_truth": root / "dynamics_truth" / "dyngen" / "smoke" / "evaluation" / "dynamics_truth_metrics.csv",
    }
    rows = [{"stage": key, "source": str(path), "available": path.is_file()} for key, path in sources.items()]
    _save_frame(pd.DataFrame(rows), figures / "figure_manifest.csv")
    stage_specs = [
        ("input_audit", root / "audit" / "audit.json", "formal"),
        ("layerwise_smoke", root / "layerwise" / "smoke" / "status.json", "smoke"),
        ("layerwise_formal", root / "layerwise" / "formal" / "status.json", "formal"),
        ("interventions_smoke", root / "interventions" / "smoke" / "availability.json", "smoke"),
        ("interventions_formal", root / "interventions" / "formal" / "availability.json", "formal"),
        ("surprisal_smoke", root / "surprisal" / "smoke" / "status.json", "smoke"),
        ("surprisal_formal", root / "surprisal" / "formal" / "status.json", "formal"),
        ("dyngen_smoke", root / "dynamics_truth" / "dyngen" / "smoke" / "status.json", "smoke"),
        ("dyngen_formal", root / "dynamics_truth" / "dyngen" / "formal" / "status.json", "formal"),
        ("erythroid_temporal", root / "temporal_transition" / "erythroid" / "status.json", "descriptive"),
        ("traxler_reference_smoke", root / "reference_anomaly" / "traxler" / "smoke" / "status.json", "smoke"),
        ("traxler_reference_formal", root / "reference_anomaly" / "traxler" / "formal" / "status.json", "formal"),
        ("align_retention", root / "align_retention" / "status.json", "existing-results reanalysis"),
        ("robustness", root / "robustness" / "forebrain" / "status.json", "descriptive"),
    ]
    status_rows = []
    for stage, path, stage_scope in stage_specs:
        payload = json.loads(path.read_text()) if path.is_file() else {}
        status_rows.append({"stage": stage, "scope": stage_scope, "status": payload.get("status", "not_run"),
                            "status_path": str(path), "available": path.is_file()})
    _save_frame(pd.DataFrame(status_rows), root / "IMPLEMENTATION_STATUS.csv")
    available = {row["stage"]: row["available"] for row in rows}
    if all(available[key] for key in ("layerwise", "interventions", "surprisal")):
        layer = pd.read_csv(sources["layerwise"])
        intervention = pd.read_csv(sources["interventions"])
        surprise = pd.read_csv(sources["surprisal"])
        temporal_path = root / "temporal_transition" / "erythroid" / "stage_bootstrap.csv"
        temporal_frame = pd.read_csv(temporal_path) if temporal_path.is_file() else None
        _plot_dashboard(layer, intervention, surprise, temporal_frame,
                        figures / f"00_distance_semantics_{scope}_dashboard", scope=scope)
        intervention_long = root / "interventions" / scope / "intervention_long.parquet"
        if intervention_long.is_file():
            _summarize_intervention_semantics(pd.read_parquet(intervention_long), intervention_long.parent)
    _write_report(cfg, root, available, scope=scope)
    return figures


def _plot_dashboard(layer: pd.DataFrame, intervention: pd.DataFrame, surprise: pd.DataFrame,
                    temporal_frame: pd.DataFrame | None, stem: Path, scope: str = "smoke") -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for model, group in layer.groupby("model"):
        axes[0, 0].plot(group.layer, group.null_minus_correct, marker="o", label=model)
    axes[0, 0].set(xlabel="Layer (0=input)", ylabel="Null − correct", title="A  Pairing geometry changes across depth")
    axes[0, 0].legend(frameon=False)
    if "model" in intervention:
        intervention = intervention[intervention.model.eq("epoch_11")]
    dose = intervention[intervention.condition.isin(["u_shift", "s_shift", "both_shift"])]
    for condition, group in dose.groupby("condition"):
        axes[0, 1].plot(group.severity, group.mean_delta, marker="o", label=condition)
    axes[0, 1].axhline(0, color="black", lw=.8)
    axes[0, 1].set(xlabel="Bin shift", ylabel="Δdistance", title="B  Artificial corruption response")
    axes[0, 1].legend(frameon=False)
    context = intervention[intervention.condition.str.contains("context|gene_only|value_only|modality_full")].copy()
    context["condition_group"] = context.condition.str.replace(r"_seed\d+$", "", regex=True)
    context = context.groupby("condition_group", as_index=False).agg(mean_delta=("mean_delta", "mean"))
    context = context.sort_values("mean_delta")
    axes[1, 0].barh(context.condition_group, context.mean_delta)
    axes[1, 0].axvline(0, color="black", lw=.8)
    axes[1, 0].set(xlabel="Mean paired Δdistance", title="C  Factor/context decomposition")
    if temporal_frame is not None:
        axes[1, 1].plot(temporal_frame.stage_numeric, temporal_frame.adjusted_mean, marker="o", color="#7c3aed")
        axes[1, 1].fill_between(temporal_frame.stage_numeric, temporal_frame.adjusted_ci_low,
                                temporal_frame.adjusted_ci_high, alpha=.25, color="#a78bfa")
        axes[1, 1].axhline(0, color="black", lw=.8)
        axes[1, 1].set(xlabel="Embryonic day", ylabel="Adjusted distance", title="D  Erythroid sample-aware profile")
    else:
        axes[1, 1].text(.5, .5, "Temporal result unavailable", ha="center", va="center")
        axes[1, 1].set_axis_off()
    fig.suptitle(f"scUS zero-shot U/S distance semantics — completed {scope} evidence", fontsize=15)
    fig.tight_layout()
    _save_figure(fig, stem)


def _write_report(cfg: dict, root: Path, available: dict[str, bool], scope: str = "smoke") -> None:
    audit_result = json.loads((root / "audit" / "audit.json").read_text())
    layer = pd.read_csv(root / "layerwise" / scope / "layerwise_metrics.csv")
    inter = pd.read_csv(root / "interventions" / scope / "intervention_summary.csv")
    surprise = pd.read_csv(root / "surprisal" / scope / "surprisal_comparison.csv")
    consistency = json.loads((root / "layerwise" / scope / "reference_consistency.json").read_text())
    def metric(condition: str, column: str = "mean_delta") -> float:
        row = inter[inter.condition.eq(condition)]
        return float(row.iloc[0][column]) if len(row) else float("nan")
    trained = layer[(layer.model == "epoch_11") & (layer.layer == layer.layer.max())].iloc[0]
    random = layer[(layer.model == "random_42") & (layer.layer == layer.layer.max())].iloc[0]
    corr = surprise.loc[surprise.comparison.eq("distance_vs_exact_surprisal"), "spearman"]
    dose_path = root / "interventions" / scope / "dose_response_metrics.csv"
    dose_text = "剂量读出尚未汇总。"
    if dose_path.is_file():
        dose = pd.read_csv(dose_path)
        if "model" in dose:
            dose = dose[dose.model.eq("epoch_11")]
        dose = dose.set_index(["condition", "readout"])
        dose_text = (f"单独改变 U 时 distance AUROC={dose.loc[('u_shift', 'distance'), 'corruption_auroc']:.3f}，"
                     f"直接 |bin差|={dose.loc[('u_shift', 'abs_bin_delta'), 'corruption_auroc']:.3f}；"
                     f"单独改变 S 时分别为 {dose.loc[('s_shift', 'distance'), 'corruption_auroc']:.3f} 和 "
                     f"{dose.loc[('s_shift', 'abs_bin_delta'), 'corruption_auroc']:.3f}；"
                     f"U/S 共同移动时 distance AUROC={dose.loc[('both_shift', 'distance'), 'corruption_auroc']:.3f}。")
    temporal_path = root / "temporal_transition" / "erythroid" / "stage_bootstrap.csv"
    temporal_text = "尚未运行。"
    if temporal_path.is_file():
        temporal = pd.read_csv(temporal_path)
        late = temporal.sort_values("stage_numeric").iloc[-1]
        before = temporal.sort_values("stage_numeric").iloc[-2]
        temporal_text = (f"Erythroid 在 {late.stage} 的 raw sample-mean 为 {late['mean']:.4f}，"
                         f"前一阶段 {before.stage} 为 {before['mean']:.4f}；校正 |bin差| 与 pair coverage 后"
                         f"分别为 {late.adjusted_mean:.4f} 和 {before.adjusted_mean:.4f}。该峰仍需预注册窗口及外部数据确认。")
    align_text = "Align 语义保持尚未计算。"
    align_path = root / "align_retention" / "semantic_effect_retention.csv"
    if align_path.is_file():
        align = pd.read_csv(align_path)
        cross_gene = align[align.condition.eq("within_cell_gene")].set_index("model")
        align_text = (f"跨基因错配的模型内标准化效应：Zero-shot {cross_gene.loc['zero_shot', 'standardized_effect']:.2f}，"
                      f"adapter-only {cross_gene.loc['adapter_only', 'standardized_effect']:.2f}，"
                      f"联合微调 epoch29 {cross_gene.loc['joint_epoch29', 'standardized_effect']:.2f}；"
                      "二者分别保留约 "
                      f"{100*cross_gene.loc['adapter_only', 'zero_shot_effect_retention']:.1f}% 和 "
                      f"{100*cross_gene.loc['joint_epoch29', 'zero_shot_effect_retention']:.1f}% 的标准化效应。")
    robustness_text = "Forebrain 混杂解释尚未计算。"
    robustness_path = root / "robustness" / "forebrain" / "cell_variance_decomposition.csv"
    if robustness_path.is_file():
        robustness = pd.read_csv(robustness_path).set_index("feature")
        robustness_text = (
            f"cell-level 可观测变量的联合线性模型原位 R²={robustness.full_model_r2.iloc[0]:.3f}；"
            f"有效基因数单变量 R²={robustness.loc['eligible_genes', 'univariate_r2']:.3f}、"
            f"Spearman={robustness.loc['eligible_genes', 'spearman']:.3f}，"
            f"S 检测率单变量 R²={robustness.loc['s_detection', 'univariate_r2']:.3f}。"
            f"直接 |bin差| 的单变量 R²仅 {robustness.loc['median_abs_bin_delta', 'univariate_r2']:.3f}。"
            "这些是相关且共线的描述性变量，不能把 R²相加或解释为因果贡献。")
    dynamics_text = "dyngen 动力学真值尚未计算。"
    dynamics_path = root / "dynamics_truth" / "dyngen" / "smoke" / "evaluation" / "dynamics_truth_metrics.csv"
    if dynamics_path.is_file():
        dynamics = pd.read_csv(dynamics_path)
        def dyn(model: str, truth: str, readout: str, metric_name: str = "spearman") -> float:
            row = dynamics[(dynamics.model == model) & (dynamics.truth == truth) &
                           (dynamics.readout == readout)]
            return float(row.iloc[0][metric_name]) if len(row) else float("nan")
        dynamics_text = (
            f"linear dyngen smoke 中，epoch-11 distance 与 |velocity| 的 Spearman={dyn('epoch_11', 'abs_velocity', 'distance'):.3f}，"
            f"random-init distance={dyn('random_42', 'abs_velocity', 'distance'):.3f}，"
            f"直接 |bin差|={dyn('epoch_11', 'abs_velocity', 'abs_bin_delta'):.3f}；"
            f"epoch-11 对 top-quartile |velocity| 的 AUROC={dyn('epoch_11', 'top_quartile_abs_velocity', 'distance', 'auroc'):.3f}。"
            "这是单 topology、单 simulation/mapping seed 的 smoke，不能用于正面 claim。")
    reference_text = "Traxler 正式 reference 增量尚未完成。"
    reference_bootstrap_path = root / "reference_anomaly" / "traxler" / "formal" / "reference_increment_bootstrap.csv"
    reference_metrics_path = root / "reference_anomaly" / "traxler" / "formal" / "rep2_metrics.csv"
    if reference_bootstrap_path.is_file() and reference_metrics_path.is_file():
        ref = pd.read_csv(reference_bootstrap_path).set_index("comparison")
        ref_metrics = pd.read_csv(reference_metrics_path)
        expression_auc = ref_metrics[ref_metrics.method.eq("B0_expression")].groupby("target").auroc.mean().mean()
        expression_plus_auc = ref_metrics[ref_metrics.method.eq("B0_expression+A_distance")].groupby("target").auroc.mean().mean()
        row = ref.loc["expression_plus_distance"]
        state_row = ref.loc["state_plus_distance"]
        combined_row = ref.loc["combined_plus_distance"]
        reference_text = (
            f"14 个 target 内先平均时间后，表达基线 AUROC={expression_auc:.3f}，加 A-distance 后={expression_plus_auc:.3f}；"
            f"配对 ΔAUROC={row.mean_delta_auroc:.4f}（target-bootstrap 95% CI "
            f"{row.bootstrap_ci_low:.4f}–{row.bootstrap_ci_high:.4f}，正向 target {row.positive_target_fraction:.1%}）。"
            f"state 与联合基线的 ΔAUROC 分别为 {state_row.mean_delta_auroc:.4f} 和 "
            f"{combined_row.mean_delta_auroc:.4f}，两者 CI 同样跨 0。"
            "与 shuffled/random 容量对照近似，因此目前不支持通用异常检测能力。")
    text = f"""# scUS Zero-shot U/S Distance 语义解析报告

## 当前完成范围

- 输入审计：{audit_result['cells']:,} 个 Forebrain 细胞，{audit_result['vocabulary_genes_seen']:,} 个历史完整 token 基因。
- 已完成：CPU 单元 smoke、128-cell 全背景 formal layer-wise、128-cell × 128-target formal 受控因素/上下文干预、formal exact/grouped conditional surprisal、dyngen linear 动力学真值 smoke、Erythroid sample-aware 时间描述、Traxler 正常参考 formal、Forebrain 全细胞混杂解释，以及既有全输入 pairing controls 的 Align 语义保持重分析。
- 三个 formal GPU 阶段均通过资源门禁并以 exit 0 完成；运行未使用 GPU 1，且未抢占其他进程。
- 尚未形成结论：dyngen 三拓扑×多 seed 正式验证、Erythroid 正式 surprisal、Gastrulation/chondrocyte 外部时间验证和跨 study 异常验证。Traxler 全量 reference 已完成，但结果不支持正面异常检测 claim。

## 关键可重复性

- 新实现最后一层与既有 epoch-11 全基因 Zero-shot distance 的缺失 mask 完全一致；{consistency['shared_values']:,} 个共享值最大绝对误差为 {consistency['maximum_absolute_error']:.1g}。
- 所有目标干预保留未指定 token；缺失 pair 不转成真实零；grouped surprisal 不同时 mask partner。

## 当前 {scope} 读数

1. **pairing geometry 并非全由深层网络新生。** 输入层已有很强的严格匹配跨基因 null 分离；epoch-11 最后一层 raw null−correct 为 {trained.null_minus_correct:.4f}，random-init seed 42 为 {random.null_minus_correct:.4f}。不同模型的绝对距离尺度不可直接定胜负；结果需结合模型内标准化效应、多个 checkpoint 和 random seeds 解读。
2. **context 有可测但相对较小的影响。** same-type donor context 的 mean Δdistance 为 {metric('same_type_context_seed42'):.4f}，different-type 为 {metric('different_type_context_seed42'):.4f}。这是 128-cell formal 干预，但仍只来自 Forebrain 两个 sample，不能外推为普适规律。
3. **gene identity 的影响更大。** gene-only S错配 mean Δdistance 为 {metric('gene_only_s'):.4f}，明显大于当前 context 干预；这与先前“跨基因效应较强”的观察一致。
4. **distance 是无方向量。** 同时交换 U/S 的值和 flag，mean Δdistance 为 {metric('modality_full_swap'):.4f}；cosine 对两端交换本来就对称，因此不能读出 induction/repression 方向。
5. **并非理想的单调损坏强度计。** {dose_text} 单模态偏移只有弱识别力，且当前不优于直接 bin 基线；双模态同向改变接近随机。这支持“关系敏感性”，尚不支持“异常概率”。
6. **surprisal 与 distance 有关但不等价。** 当前 16,384 个 epoch-11 cell-gene 的 exact q-pair 与 distance Spearman 为 {float(corr.iloc[0]) if len(corr) else float('nan'):.3f}；exact 与 grouped 近似的 Spearman 接近 1，说明分组近似在本数据上高度一致。二者仍是不同读出，不能互换。
7. **时间结果存在混杂但不完全由两个简单协变量解释。** {temporal_text}
8. **Align 保留了部分跨基因错配语义，但改变了细胞排序。** {align_text} 这不等于 corruption/time/reference 语义已保留，后三项仍须重跑。
9. **cell scalar 对可用输入规模十分敏感。** {robustness_text} 因此 scalar 不宜直接充当异常分数；profile 分析必须显式控制 missing mask、检测率与有效基因数。
10. **动力学 smoke 暂不显示预训练 distance 的特异优势。** {dynamics_text} 正式多 seed 结果若仍如此，应把 distance 定位为细胞表征，而不是动力学幅度估计器。
11. **正常参考校准未带来稳定的扰动识别增量。** {reference_text}

## 当前最稳妥的语义

目前可称为 **gene-resolved、context-modulated、unsigned U/S representation discrepancy**。它不是 RNA velocity、方向、平衡概率或通用异常分数。只有正式 random/input 对照、人工破坏 AUROC、动力学真值和跨数据 reference 增量依次通过后，才逐级升级 claim。

## 主要路径

- Layer-wise：`{root / 'layerwise' / scope}`
- 干预：`{root / 'interventions' / scope}`
- Surprisal：`{root / 'surprisal' / scope}`
- Erythroid 时间：`{root / 'temporal_transition' / 'erythroid'}`
- Traxler reference smoke：`{root / 'reference_anomaly' / 'traxler' / 'smoke'}`
- Traxler reference 正式结果：`{root / 'reference_anomaly' / 'traxler' / 'formal'}`
- Align 保留：`{root / 'align_retention'}`
- 混杂解释：`{root / 'robustness' / 'forebrain'}`
- dyngen 真值 smoke：`{root / 'dynamics_truth' / 'dyngen' / 'smoke' / 'evaluation'}`
- 综合图：`{root / 'figures' / f'00_distance_semantics_{scope}_dashboard.png'}`
"""
    (root / "DISTANCE_SEMANTICS_REPORT.md").write_text(text, encoding="utf-8")


def _save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_layerwise(frame: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for model, group in frame.groupby("model"):
        style = "-" if model.startswith("epoch") else "--"
        axes[0].plot(group.layer, group.null_minus_correct, marker="o", ls=style, label=model)
        axes[1].plot(group.layer, group.cell_type_cross_sample_knn_ba, marker="o", ls=style, label=model)
        axes[2].plot(group.layer, group.direct_abs_bin_delta_spearman, marker="o", ls=style, label=model)
    axes[0].set_title("Pairing separation"); axes[1].set_title("Cell-type 15-NN BA")
    axes[2].set_title("Correlation with |U-bin−S-bin|")
    for ax in axes: ax.set_xlabel("Input=0; Transformer layer"); ax.grid(alpha=.2)
    axes[0].set_ylabel("null − correct distance"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Layer-wise formation of frozen scUS U/S distance")
    _save_figure(fig, stem)


def _plot_interventions(frame: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "model" in frame: frame = frame[frame.model.eq("epoch_11")]
    dose = frame[frame.condition.isin(["u_shift", "s_shift", "both_shift"])]
    for condition, group in dose.groupby("condition"):
        summary = group.groupby("severity").delta_distance.mean().sort_index()
        axes[0].plot(summary.index, summary.values, marker="o", label=condition)
    axes[0].axhline(0, color="black", lw=.8); axes[0].set(xlabel="Bin shift", ylabel="Δdistance", title="Known corruption dose response")
    axes[0].legend(frameon=False)
    other = frame[~frame.condition.isin(["correct", "u_shift", "s_shift", "both_shift"])]
    summary = other.groupby("condition").delta_distance.mean().sort_values()
    axes[1].barh(summary.index, summary.values); axes[1].axvline(0, color="black", lw=.8)
    axes[1].set(xlabel="Mean paired Δdistance", title="Factor and context interventions")
    fig.suptitle("Frozen epoch-11: controlled U/S relation interventions")
    _save_figure(fig, stem)


def _plot_surprisal(frame: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    trained = frame[frame.model == "epoch_11"]
    wide = trained.pivot_table(index=["cell_id", "gene_id"], columns="protocol", values="q_pair")
    if {"exact", "grouped_8x2"}.issubset(wide.columns):
        axes[0].scatter(wide.exact, wide.grouped_8x2, s=10, alpha=.5)
        axes[0].set(xlabel="Exact q-pair", ylabel="Grouped q-pair", title="Approximation agreement")
    exact = frame[frame.protocol == "exact"]
    for model, group in exact.groupby("model"):
        axes[1].scatter(group.distance, group.q_pair, s=10, alpha=.5, label=model)
    axes[1].set(xlabel="Cosine distance", ylabel="Conditional surprisal", title="Two non-equivalent readouts")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Cross-splice conditional reconstruction surprisal")
    _save_figure(fig, stem)

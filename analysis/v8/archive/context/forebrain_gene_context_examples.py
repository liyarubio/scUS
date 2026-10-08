#!/usr/bin/env python3
"""Fixed-token, gene-resolved contextual hidden-state audit for Forebrain.

This workflow never trains a model.  It encodes every cell with its complete
historical token context, stores full 128-D U/S vectors for a biology-defined
20-gene panel, and stores scalar readouts for input-matched null genes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "4"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from threadpoolctl import threadpool_limits

from scus.artifacts import sha256
from scus.config import load_config


DEFAULT_CONFIG = ROOT / "configs/forebrain_gene_context_examples.yaml"
MODEL_ORDER = ["input_embedding", "random_42", "random_43", "random_44", "epoch_11"]
MODEL_LABELS = {
    "input_embedding": "Input embedding",
    "random_42": "Random 42",
    "random_43": "Random 43",
    "random_44": "Random 44",
    "epoch_11": "Pretrained epoch 11",
}
CELL_COLORS = {
    "Radial Glia": "#3A7D70", "Neuroblast": "#E49A3A",
    "Immature Neuron": "#557DBB", "Neuron": "#C85A5A",
}


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npy")
    np.save(tmp, value)
    tmp.replace(path)


def _npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **values)
    tmp.replace(path)


def _out(cfg: dict) -> Path:
    return Path(cfg["output_dir"])


def _software() -> dict:
    import torch
    return {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "created": time.strftime("%FT%T%z"),
    }


def _panel(cfg: dict) -> pd.DataFrame:
    rows = []
    for category, genes in cfg["panel"].items():
        rows.extend({"gene_name": str(g), "category": category, "panel_order": len(rows)} for g in genes)
    frame = pd.DataFrame(rows)
    if len(frame) != 20 or frame.gene_name.duplicated().any():
        raise ValueError("The biological panel must contain 20 unique genes")
    return frame


def _spotlights(cfg: dict) -> pd.DataFrame:
    frame = pd.DataFrame(cfg["spotlights"])
    required = {"gene", "u_bin", "s_bin", "type_a", "type_b"}
    if len(frame) != 6 or set(frame.columns) != required or frame.gene.duplicated().any():
        raise ValueError("Expected six unique, fully specified spotlight examples")
    return frame


def _eligible_strata_counts(u: np.ndarray, s: np.ndarray, meta: pd.DataFrame, minimum: int) -> np.ndarray:
    """Count exact-bin/type contrasts meeting the rule in both samples."""
    samples = sorted(meta["sample"].unique())
    types = sorted(meta["cell_type"].unique())
    if len(samples) != 2:
        raise ValueError("This audit requires exactly two Forebrain samples")
    sample_masks = [meta["sample"].eq(x).to_numpy() for x in samples]
    type_masks = [meta.cell_type.eq(x).to_numpy() for x in types]
    result = np.zeros(u.shape[1], np.int32)
    for gene in range(u.shape[1]):
        code = u[:, gene].astype(np.int16) * 16 + s[:, gene].astype(np.int16)
        observed = (u[:, gene] > 0) & (s[:, gene] > 0)
        counts = np.zeros((2, len(types), 256), np.int32)
        for si, sm in enumerate(sample_masks):
            for ti, tm in enumerate(type_masks):
                counts[si, ti] = np.bincount(code[observed & sm & tm], minlength=256)
        total = 0
        for a, b in itertools.combinations(range(len(types)), 2):
            total += int(np.all(counts[:, [a, b], :] >= minimum, axis=(0, 1)).sum())
        result[gene] = total
    return result


def _input_features(u: np.ndarray, s: np.ndarray, ref: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    pair = (u > 0) & (s > 0)
    count = pair.sum(0)
    def stat(x: np.ndarray, mode: str) -> np.ndarray:
        masked = np.where(pair, x, np.nan)
        return np.nanmean(masked, axis=0) if mode == "mean" else np.nanstd(masked, axis=0)
    return pd.DataFrame({
        "paired_observation_fraction": pair.mean(0),
        "u_bin_mean": stat(u, "mean"), "u_bin_std": stat(u, "std"),
        "s_bin_mean": stat(s, "mean"), "s_bin_std": stat(s, "std"),
        "reference_observed": pair[ref].sum(0),
        "eligible_exact_strata": strata,
        "paired_observations": count,
    })


def _match_candidates(features: pd.DataFrame, genes: pd.DataFrame, panel_names: list[str], count: int) -> pd.DataFrame:
    columns = ["paired_observation_fraction", "u_bin_mean", "u_bin_std", "s_bin_mean", "s_bin_std",
               "reference_observed", "eligible_exact_strata"]
    x = features[columns].to_numpy(float)
    x[:, -1] = np.log1p(x[:, -1])
    mean, sd = x.mean(0), x.std(0); sd[sd == 0] = 1
    z = (x - mean) / sd
    names = genes.gene_name.astype(str).to_numpy()
    panel_set = set(panel_names)
    pool = np.asarray([i for i, name in enumerate(names) if name not in panel_set], np.int64)
    rows = []
    for panel_gene in panel_names:
        pi = int(np.flatnonzero(names == panel_gene)[0])
        d = np.sqrt(np.mean((z[pool] - z[pi]) ** 2, axis=1))
        order = np.lexsort((genes.vocab_id.to_numpy()[pool], d))[:count]
        for rank, local in enumerate(order, 1):
            ci = int(pool[local])
            rows.append({"panel_gene": panel_gene, "candidate_gene": names[ci],
                         "candidate_vocab_id": int(genes.vocab_id.iloc[ci]),
                         "match_rank": rank, "input_feature_distance": float(d[local])})
    frame = pd.DataFrame(rows)
    if frame.groupby("panel_gene").size().ne(count).any():
        raise AssertionError("Incomplete matched candidate list")
    return frame


def audit(cfg: dict) -> Path:
    out = _out(cfg); out.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg["data"]; panel = _panel(cfg); spot = _spotlights(cfg)
    meta = pd.read_csv(data_cfg["cells_path"])
    genes = pd.read_csv(data_cfg["genes_path"])
    selection = pd.read_csv(data_cfg["gene_selection_path"])
    input_genes = pd.read_csv(data_cfg["input_genes_path"])
    input_audit = pd.read_csv(data_cfg["input_audit_path"])
    if len(meta) != 1720 or not meta.cell_id.is_unique or len(genes) != 10692:
        raise ValueError("Unexpected Forebrain cell/gene identity")
    if not np.array_equal(meta.cell_id, input_audit.cell_id):
        raise ValueError("Cell order differs from audited full-token input")
    if not np.array_equal(genes[["gene_name", "vocab_id"]], selection[["gene_name", "vocab_id"]]):
        raise ValueError("Gene selection and hidden-cache panels differ")
    if not set(panel.gene_name).issubset(set(genes.gene_name)):
        raise ValueError(f"Missing panel genes: {set(panel.gene_name) - set(genes.gene_name)}")
    if not set(spot.gene).issubset(set(panel.gene_name)):
        raise ValueError("Spotlights must be a subset of the biological panel")
    eligible = selection.eligible.to_numpy(bool)
    if int(eligible.sum()) != 10011:
        raise ValueError("Expected 10,011 reference-eligible paired genes")
    eligible_genes = genes.loc[eligible].reset_index(drop=True)
    bins = np.load(data_cfg["input_bins_path"])
    lookup = {int(v): i for i, v in enumerate(input_genes.vocab_id)}
    columns = np.asarray([lookup[int(v)] for v in eligible_genes.vocab_id], np.int64)
    u, s = bins["u"][:, columns].astype(np.uint8), bins["s"][:, columns].astype(np.uint8)
    pair_mask = np.load(data_cfg["pair_mask_path"])[:, eligible]
    if not np.array_equal(pair_mask, (u > 0) & (s > 0)):
        raise ValueError("Input-bin and historical hidden-cache pair masks differ")
    ref = meta["sample"].eq("10X_17_029").to_numpy()
    minimum = int(cfg["analysis"]["minimum_cells_per_type_sample"])
    strata = _eligible_strata_counts(u, s, meta, minimum)
    features = _input_features(u, s, ref, strata)
    features = pd.concat([eligible_genes, features], axis=1)
    candidates = _match_candidates(features, eligible_genes, panel.gene_name.tolist(),
                                   int(cfg["analysis"]["matched_candidates_per_gene"]))
    encoded_names = panel.gene_name.tolist() + sorted(set(candidates.candidate_gene) - set(panel.gene_name))
    encoded = features.set_index("gene_name").loc[encoded_names].reset_index()
    encoded["is_panel"] = encoded.gene_name.isin(panel.gene_name)
    encoded = encoded.merge(panel[["gene_name", "category", "panel_order"]], on="gene_name", how="left")
    full_index = {name: i for i, name in enumerate(eligible_genes.gene_name)}
    selected_columns = np.asarray([full_index[x] for x in encoded.gene_name], np.int64)
    panel_info = panel.merge(features, on="gene_name", validate="one_to_one")
    if not panel_info.eligible_exact_strata.ge(0).all():
        raise AssertionError("Invalid stratum counts")
    _csv(out / "cells.csv", meta)
    _csv(out / "gene_panel.csv", panel_info)
    _csv(out / "spotlight_manifest.csv", spot)
    _csv(out / "eligible_gene_input_features.csv", features)
    _csv(out / "matched_candidates.csv", candidates)
    _csv(out / "encoded_genes.csv", encoded)
    _npz(out / "encoded_gene_bins.npz", u=u[:, selected_columns], s=s[:, selected_columns])
    source_files = {key: str(Path(value)) for key, value in data_cfg.items() if key.endswith("_path")}
    source_hashes = {key: {"path": value, "sha256": sha256(value)} for key, value in source_files.items()}
    source_hashes.update({"checkpoint": {"path": cfg["checkpoint"], "sha256": sha256(cfg["checkpoint"])},
                          "config": {"path": cfg["_config_path"], "sha256": sha256(cfg["_config_path"])}})
    _json(out / "input_hashes.json", source_hashes)
    _json(out / "audit.json", {
        "status": "passed", "cells": len(meta), "samples": sorted(meta["sample"].unique()),
        "cell_types": sorted(meta.cell_type.unique()), "source_genes": len(genes),
        "eligible_genes": len(eligible_genes), "panel_genes": len(panel),
        "encoded_scalar_genes": len(encoded), "full_vector_genes": int(encoded.is_panel.sum()),
        "minimum_cells_per_type_sample": minimum, "continuous_expression_equality": False,
        "interpretation": "Exact equality of the discrete gene/value/modality target tokens, not raw-count or moment equality",
        "software": _software(),
    })
    _json(out / "status.json", {"status": "audit_complete", "updated": time.time()})
    return out


def _load_data(cfg: dict):
    from scus.joint_reconstruction import load_joint_data
    return load_joint_data(cfg)


def _new_model(cfg: dict, device, random_seed: int | None):
    import torch
    from scus.joint_reconstruction import JointModel
    from scus.models import MaskedModel
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    hp = dict(payload["hyper_parameters"])
    if random_seed is None:
        model = JointModel.from_checkpoint(cfg["checkpoint"])
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(random_seed))
            model = JointModel(MaskedModel(**hp))
    model.activation_checkpointing = False
    return model.eval().requires_grad_(False).to(device)


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.encoder.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _resource_gate(cfg: dict, out: Path, smoke: bool, device: str) -> None:
    import torch
    from scus.ablations import resource_preflight
    if not device.startswith("cuda:"):
        raise ValueError("Real full-context encoding requires an allowed CUDA device")
    gpu = int(device.split(":", 1)[1])
    if gpu not in (0, 2, 3) or os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Use physical GPU 0, 2, or 3 without CUDA_VISIBLE_DEVICES remapping; GPU 1 is excluded")
    settings = cfg["analysis"]
    samples = int(settings["preflight_smoke_samples"] if smoke else settings["preflight_formal_samples"])
    passed, trace, reason = resource_preflight(
        gpu, samples=samples, interval_seconds=float(settings["preflight_interval_seconds"]),
        minimum_free_mb=float(settings["minimum_free_mb"]),
        maximum_mean_utilization=float(settings["maximum_mean_utilization"]),
    )
    _csv(out / "resource_preflight.csv", trace)
    _json(out / "resource_preflight_status.json", {"status": "passed" if passed else "blocked", "reason": reason})
    if not passed:
        raise RuntimeError(f"GPU resource gate failed: {reason}")
    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(settings["memory_budget_gib"]) * 2**30 / total), torch.device(device))


def _select_smoke(meta: pd.DataFrame, input_audit: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    longest = int(input_audit.genes.idxmax())
    order = sorted(range(len(meta)), key=lambda i: hashlib.sha256(f"{seed}:{meta.cell_id.iloc[i]}".encode()).hexdigest())
    selected = [longest]
    for _, group in meta.groupby(["sample", "cell_type"], sort=True):
        choices = [i for i in order if i in set(group.index)]
        selected.extend(choices[: max(1, count // 8)])
    selected = list(dict.fromkeys(selected))
    for i in order:
        if len(selected) >= count: break
        if i not in selected: selected.append(i)
    return np.asarray(selected[:count], np.int64)


def _extract_batch(model, batch, encoded_ids: np.ndarray, panel_mask: np.ndarray, include_input: bool):
    import torch
    from scus.joint_batch import move
    from scus.runtime import cosine_distance
    device = next(model.parameters()).device
    batch = move(batch, device)
    g, v, splice = batch["gene_ids"], batch["value_bins"], batch["splice_flags"]
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        input_hidden = model.encoder.embed(g, v, splice) if include_input else None
        hidden, _ = model.hidden(g, v, splice)
    outputs = []
    for tensor in ([input_hidden, hidden] if include_input else [hidden]):
        name = "input_embedding" if tensor is input_hidden else "hidden"
        scalar_u = np.full((len(g), len(encoded_ids)), np.nan, np.float32)
        scalar_s = np.full_like(scalar_u, np.nan)
        distance = np.full_like(scalar_u, np.nan)
        full_u = np.full((len(g), int(panel_mask.sum()), tensor.shape[-1]), np.nan, np.float32)
        full_s = np.full_like(full_u, np.nan)
        for row, length in enumerate(batch["lengths"]):
            genes = g[row, :length:2].detach().cpu().numpy()
            values = v[row, :length].detach().cpu().numpy().reshape(-1, 2)
            local = {int(x): i for i, x in enumerate(genes)}
            for col, gene_id in enumerate(encoded_ids):
                position = local.get(int(gene_id))
                if position is None or not (values[position] > 0).all():
                    continue
                uvec = tensor[row, 2 * position].float(); svec = tensor[row, 2 * position + 1].float()
                scalar_u[row, col] = float(uvec.mean())
                scalar_s[row, col] = float(svec.mean())
                distance[row, col] = float(cosine_distance(uvec[None], svec[None])[0])
                if panel_mask[col]:
                    pcol = int(np.flatnonzero(np.flatnonzero(panel_mask) == col)[0])
                    full_u[row, pcol] = uvec.cpu().numpy(); full_s[row, pcol] = svec.cpu().numpy()
        outputs.append((name, dict(scalar_u=scalar_u, scalar_s=scalar_s, distance=distance,
                                   panel_u=full_u, panel_s=full_s)))
    return outputs


def _encode_indices(cfg: dict, indices: np.ndarray, out: Path, models: list[tuple[str, int | None]], batch_size: int) -> dict:
    import torch
    from scus.joint_batch import collate
    data = _load_data(cfg); encoded = pd.read_csv(_out(cfg) / "encoded_genes.csv")
    encoded_ids = encoded.vocab_id.to_numpy(np.int64); panel_mask = encoded.is_panel.to_numpy(bool)
    summaries = {}
    for label, random_seed in models:
        model = _new_model(cfg, torch.device(cfg["device"]), random_seed)
        state_hash = _state_hash(model)
        chunk_root = out / "chunks" / label; chunk_root.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            path = chunk_root / f"{start:05d}.npz"
            if path.exists(): continue
            batch = collate([data[int(i)] for i in selected])
            results = _extract_batch(model, batch, encoded_ids, panel_mask, include_input=label == "epoch_11")
            for kind, arrays in results:
                target_label = "input_embedding" if kind == "input_embedding" else label
                target = out / "chunks" / target_label / f"{start:05d}.npz"
                _npz(target, indices=selected, **arrays)
            _json(out / "progress.json", {"status": "encoding", "model": label,
                                           "cells_completed": int(start + len(selected)), "updated": time.time()})
        summaries[label] = {"random_seed": random_seed, "weights_loaded": random_seed is None,
                            "encoder_state_sha256": state_hash}
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summaries


def _merge_chunks(out: Path, labels: list[str], expected_indices: np.ndarray) -> None:
    for label in labels:
        destination = out / "models" / label
        complete = destination / "complete.json"
        if complete.exists(): continue
        parts = []
        for path in sorted((out / "chunks" / label).glob("*.npz")):
            with np.load(path) as value: parts.append({k: value[k] for k in value.files})
        if not parts:
            raise FileNotFoundError(f"No chunks for {label}")
        indices = np.concatenate([x["indices"] for x in parts])
        if not np.array_equal(indices, expected_indices):
            raise ValueError(f"Incomplete or misordered chunks for {label}")
        for key in ("scalar_u", "scalar_s", "distance", "panel_u", "panel_s"):
            _npy(destination / f"{key}.npy", np.concatenate([x[key] for x in parts]))
        _json(complete, {"complete": True, "cells": len(indices), "cell_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})


def smoke(cfg: dict) -> Path:
    out = _out(cfg) / "smoke"; out.mkdir(parents=True, exist_ok=True)
    if not (_out(cfg) / "audit.json").exists(): audit(cfg)
    meta = pd.read_csv(_out(cfg) / "cells.csv"); ia = pd.read_csv(cfg["data"]["input_audit_path"])
    indices = _select_smoke(meta, ia, int(cfg["analysis"]["smoke_cells"]), int(cfg["seed"]))
    _csv(out / "cell_manifest.csv", meta.iloc[indices].assign(source_index=indices))
    _resource_gate(cfg, out, True, cfg["device"])
    summaries = _encode_indices(cfg, indices, out, [("random_42", 42), ("epoch_11", None)], min(4, int(cfg["analysis"]["inference_batch_size"])))
    _merge_chunks(out, ["input_embedding", "random_42", "epoch_11"], indices)
    values = np.load(out / "models/epoch_11/distance.npy")
    checks = {
        "cells_complete": values.shape[0] == len(indices),
        "longest_cell_included": int(ia.genes.idxmax()) in set(indices),
        "finite_observed": bool(np.isfinite(values[np.isfinite(values)]).all()),
        "distance_legal": bool(np.nanmin(values) >= -1e-6 and np.nanmax(values) <= 2 + 1e-6),
        "random_differs_from_pretrained": not np.allclose(values, np.load(out / "models/random_42/distance.npy"), equal_nan=True),
    }
    status = "passed" if all(checks.values()) else "failed"
    _json(out / "smoke.json", {"status": status, "checks": checks, "models": summaries})
    if status != "passed": raise AssertionError(checks)
    return out


def encode(cfg: dict) -> Path:
    out = _out(cfg); out.mkdir(parents=True, exist_ok=True)
    if not (out / "audit.json").exists(): audit(cfg)
    _resource_gate(cfg, out, False, cfg["device"])
    meta = pd.read_csv(out / "cells.csv"); indices = np.arange(len(meta), dtype=np.int64)
    models = [(f"random_{seed}", int(seed)) for seed in cfg["analysis"]["random_seeds"]] + [("epoch_11", None)]
    summaries = _encode_indices(cfg, indices, out, models, int(cfg["analysis"]["inference_batch_size"]))
    _merge_chunks(out, MODEL_ORDER, indices)
    encoded = pd.read_csv(out / "encoded_genes.csv")
    source_genes = pd.read_csv(cfg["data"]["genes_path"])
    source_lookup = {int(v): i for i, v in enumerate(source_genes.vocab_id)}
    source_cols = np.asarray([source_lookup[int(v)] for v in encoded.vocab_id], np.int64)
    ref_u = np.load(cfg["data"]["zero_shot_u_mean_path"], mmap_mode="r")[:, source_cols]
    ref_s = np.load(cfg["data"]["zero_shot_s_mean_path"], mmap_mode="r")[:, source_cols]
    new_u = np.load(out / "models/epoch_11/scalar_u.npy", mmap_mode="r")
    new_s = np.load(out / "models/epoch_11/scalar_s.npy", mmap_mode="r")
    if not np.array_equal(np.isfinite(ref_u), np.isfinite(new_u)) or not np.array_equal(np.isfinite(ref_s), np.isfinite(new_s)):
        raise ValueError("Epoch11 hidden missingness differs from audited cache")
    max_error = max(float(np.nanmax(abs(ref_u - new_u))), float(np.nanmax(abs(ref_s - new_s))))
    tolerance = float(cfg["analysis"]["reference_tolerance"])
    if max_error > tolerance: raise ValueError(f"Epoch11 hidden mean drift {max_error} > {tolerance}")
    model_rows = []
    for label in MODEL_ORDER:
        item = summaries.get(label, {"random_seed": None, "weights_loaded": True,
                                     "encoder_state_sha256": summaries["epoch_11"]["encoder_state_sha256"]})
        model_rows.append({"model": label, **item})
    _csv(out / "model_manifest.csv", pd.DataFrame(model_rows))
    _json(out / "encoding_complete.json", {
        "complete": True, "cells": len(meta), "panel_genes": 20, "encoded_scalar_genes": len(encoded),
        "hidden_dimensions": 128, "models": MODEL_ORDER, "batch_size": int(cfg["analysis"]["inference_batch_size"]),
        "all_source_tokens_used_as_context": True, "no_topk": True, "no_training": True,
        "epoch11_reference_max_error": max_error, "reference_tolerance": tolerance,
    })
    _json(out / "status.json", {"status": "encoding_complete", "updated": time.time()})
    return out


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    denominator = len(a) + len(b) - 2
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / denominator)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _vector_effect(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(a, float).mean(0) - np.asarray(b, float).mean(0)
    rms = float(np.sqrt(np.mean(delta ** 2)))
    pooled = np.sqrt(((len(a) - 1) * np.asarray(a).var(0, ddof=1) +
                      (len(b) - 1) * np.asarray(b).var(0, ddof=1)) / (len(a) + len(b) - 2))
    standardized = float(np.sqrt(np.mean((delta / np.where(pooled > 0, pooled, 1)) ** 2)))
    return rms, standardized


def _bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float); result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid): return result
    order = valid[np.argsort(p[valid])]; ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1)
    return result


def _valid_strata(meta: pd.DataFrame, encoded: pd.DataFrame, bins: dict[str, np.ndarray],
                  minimum: int) -> list[dict]:
    """Resolve every eligible exact-bin/type contrast once for all models.

    The original implementation rebuilt length-1,720 boolean masks inside the
    five-model loop.  That was scientifically correct but needlessly repeated
    the same input-only work.  These indices depend only on metadata and input
    bins, so caching them cannot use hidden effects or change model comparisons.
    """
    samples = sorted(meta["sample"].unique())
    types = sorted(meta.cell_type.unique())
    sample_masks = {sample: meta["sample"].eq(sample).to_numpy() for sample in samples}
    type_masks = {cell_type: meta.cell_type.eq(cell_type).to_numpy() for cell_type in types}
    specifications: list[dict] = []
    for col, gene in enumerate(encoded.gene_name):
        u, s = bins["u"][:, col], bins["s"][:, col]
        code = u.astype(np.int16) * 16 + s.astype(np.int16)
        observed = (u > 0) & (s > 0)
        for sample in samples:
            groups: dict[tuple[str, int], np.ndarray] = {}
            sample_observed = observed & sample_masks[sample]
            for cell_type in types:
                selected = np.flatnonzero(sample_observed & type_masks[cell_type])
                if not len(selected):
                    continue
                values = code[selected]
                for exact_code in np.unique(values):
                    members = selected[values == exact_code]
                    if len(members) >= minimum:
                        groups[(cell_type, int(exact_code))] = members
            for type_a, type_b in itertools.combinations(types, 2):
                shared = sorted(
                    {value for kind, value in groups if kind == type_a}
                    & {value for kind, value in groups if kind == type_b}
                )
                for exact_code in shared:
                    ub, sb = divmod(int(exact_code), 16)
                    specifications.append({
                        "col": col, "gene_name": gene, "u_bin": ub, "s_bin": sb,
                        "sample": sample, "type_a": type_a, "type_b": type_b,
                        "ia": groups[(type_a, exact_code)], "ib": groups[(type_b, exact_code)],
                    })
    return specifications


def _sample_effects(encoded: pd.DataFrame, specifications: list[dict], model: str,
                    arrays: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    scalar_rows, vector_rows = [], []
    panel_columns = np.flatnonzero(encoded.is_panel.to_numpy(bool))
    panel_local = {int(col): i for i, col in enumerate(panel_columns)}
    for specification in specifications:
        col = int(specification["col"]); ia, ib = specification["ia"], specification["ib"]
        common = {key: value for key, value in specification.items() if key not in {"col", "ia", "ib"}}
        for readout, key in (("u_hidden_mean", "scalar_u"), ("s_hidden_mean", "scalar_s"),
                             ("us_cosine_distance", "distance")):
            a, b = arrays[key][ia, col], arrays[key][ib, col]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Missing value inside valid stratum")
            # Constant context-free inputs correctly have zero effect and no
            # meaningful t-test p-value; avoid noisy precision-loss warnings.
            if float(np.ptp(a)) == 0.0 and float(np.ptp(b)) == 0.0:
                pvalue = np.nan
            else:
                pvalue = float(ttest_ind(a, b, equal_var=False).pvalue)
            scalar_rows.append({"model": model, **common, "readout": readout,
                                "n_a": len(a), "n_b": len(b),
                                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                                "mean_difference": float(a.mean() - b.mean()),
                                "cohen_d": _cohen_d(a, b), "welch_p": pvalue})
        if col in panel_local:
            pcol = panel_local[col]
            for side, key in (("u_hidden_128d", "panel_u"), ("s_hidden_128d", "panel_s")):
                a, b = arrays[key][ia, pcol], arrays[key][ib, pcol]
                rms, srms = _vector_effect(a, b)
                vector_rows.append({"model": model, **common, "readout": side,
                                    "n_a": len(a), "n_b": len(b),
                                    "centroid_rms": rms, "standardized_centroid_rms": srms})
    return scalar_rows, vector_rows


def _replicate(frame: pd.DataFrame, effect: str) -> pd.DataFrame:
    samples = sorted(frame["sample"].unique())
    keys = ["model", "gene_name", "u_bin", "s_bin", "type_a", "type_b", "readout"]
    left = frame[frame["sample"].eq(samples[0])].copy(); right = frame[frame["sample"].eq(samples[1])].copy()
    left = left.rename(columns={effect: f"effect_{samples[0]}", "n_a": f"n_a_{samples[0]}", "n_b": f"n_b_{samples[0]}"})
    right = right.rename(columns={effect: f"effect_{samples[1]}", "n_a": f"n_a_{samples[1]}", "n_b": f"n_b_{samples[1]}"})
    result = left[keys + [f"effect_{samples[0]}", f"n_a_{samples[0]}", f"n_b_{samples[0]}"]].merge(
        right[keys + [f"effect_{samples[1]}", f"n_a_{samples[1]}", f"n_b_{samples[1]}"]], on=keys, validate="one_to_one")
    a, b = result[f"effect_{samples[0]}"], result[f"effect_{samples[1]}"]
    result["direction_consistent"] = np.sign(a) == np.sign(b)
    result["replicated_abs_effect"] = np.minimum(abs(a), abs(b))
    result["minimum_group_size"] = result[[c for c in result if c.startswith("n_")]].min(axis=1)
    return result


def _draw_matched_panels(cfg: dict, scores: pd.DataFrame, candidates: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"])); draws = int(cfg["analysis"]["matched_panel_draws"])
    lookup = {g: group.candidate_gene.tolist() for g, group in candidates.groupby("panel_gene", sort=False)}
    score = scores.set_index("gene_name")
    records = []
    def summarize(names: list[str], draw: int, kind: str):
        values = score.reindex(names)
        records.append({"draw": draw, "kind": kind, "genes": len(names),
                        "genes_with_context_score": int(values.context_effect.notna().sum()),
                        "median_context_effect": float(values.context_effect.median()),
                        "median_pretraining_increment": float(values.pretraining_increment.median())})
    summarize(panel, -1, "forebrain_panel")
    for draw in range(draws):
        chosen = None
        for _ in range(100):
            used, current = set(), []
            for gene in rng.permutation(panel):
                available = [x for x in lookup[gene] if x not in used]
                if not available: break
                pick = available[int(rng.integers(len(available)))]; used.add(pick); current.append(pick)
            if len(current) == len(panel): chosen = current; break
        if chosen is None: raise RuntimeError("Could not construct a without-replacement matched panel")
        summarize(chosen, draw, "matched_random_panel")
    return pd.DataFrame(records)


def analyze(cfg: dict) -> Path:
    out = _out(cfg)
    if not (out / "encoding_complete.json").exists(): raise FileNotFoundError("Run encode first")
    meta = pd.read_csv(out / "cells.csv"); encoded = pd.read_csv(out / "encoded_genes.csv")
    panel = pd.read_csv(out / "gene_panel.csv"); candidates = pd.read_csv(out / "matched_candidates.csv")
    bins_npz = np.load(out / "encoded_gene_bins.npz"); bins = {"u": bins_npz["u"], "s": bins_npz["s"]}
    minimum = int(cfg["analysis"]["minimum_cells_per_type_sample"])
    specifications = _valid_strata(meta, encoded, bins, minimum)
    _csv(out / "eligible_strata_manifest.csv", pd.DataFrame([
        {key: value for key, value in row.items() if key not in {"ia", "ib"}}
        | {"n_a": len(row["ia"]), "n_b": len(row["ib"])}
        for row in specifications
    ]))
    scalar_rows, vector_rows = [], []
    for model in MODEL_ORDER:
        folder = out / "models" / model
        arrays = {key: np.load(folder / f"{key}.npy", mmap_mode="r") for key in
                  ("scalar_u", "scalar_s", "distance", "panel_u", "panel_s")}
        scalar, vector = _sample_effects(encoded, specifications, model, arrays)
        scalar_rows.extend(scalar); vector_rows.extend(vector)
    scalar = pd.DataFrame(scalar_rows); vector = pd.DataFrame(vector_rows)
    scalar["bh_q"] = scalar.groupby(["model", "readout"])["welch_p"].transform(lambda x: _bh(x))
    replicated = _replicate(scalar, "cohen_d")
    vector_replicated = _replicate(vector.rename(columns={"standardized_centroid_rms": "vector_effect"}), "vector_effect")
    _csv(out / "scalar_effects_by_sample.csv", scalar)
    _csv(out / "vector_effects_by_sample.csv", vector)
    _csv(out / "replicated_scalar_effects.csv", replicated)
    _csv(out / "replicated_vector_effects.csv", vector_replicated)
    # Fixed examples, including cell-level values for the main plot.
    spotlight = _spotlights(cfg); epoch = out / "models/epoch_11"
    values = {key: np.load(epoch / f"{key}.npy", mmap_mode="r") for key in ("scalar_u", "scalar_s", "distance")}
    gene_lookup = {name: i for i, name in enumerate(encoded.gene_name)}
    rows = []
    for record in spotlight.to_dict("records"):
        col = gene_lookup[record["gene"]]
        use = (bins["u"][:, col] == record["u_bin"]) & (bins["s"][:, col] == record["s_bin"]) & meta.cell_type.isin([record["type_a"], record["type_b"]]).to_numpy()
        for i in np.flatnonzero(use):
            rows.append({"cell_id": meta.cell_id.iloc[i], "sample": meta["sample"].iloc[i],
                         "cell_type": meta.cell_type.iloc[i], **record,
                         "u_hidden_mean": float(values["scalar_u"][i, col]),
                         "s_hidden_mean": float(values["scalar_s"][i, col]),
                         "us_cosine_distance": float(values["distance"][i, col])})
    examples = pd.DataFrame(rows); _csv(out / "spotlight_cell_values.csv", examples)
    # Context residuals for common-coordinate UMAP overlays.
    residual_rows = []
    for record in spotlight.to_dict("records"):
        col = gene_lookup[record["gene"]]
        frame = meta[["cell_id", "sample", "cell_type"]].copy()
        frame["gene_name"] = record["gene"]; frame["u_bin"] = bins["u"][:, col]; frame["s_bin"] = bins["s"][:, col]
        frame["u_hidden_mean"] = values["scalar_u"][:, col]; frame["s_hidden_mean"] = values["scalar_s"][:, col]
        for side in ("u", "s"):
            source = frame[f"{side}_hidden_mean"]
            center = frame.assign(value=source).groupby(["sample", "u_bin", "s_bin"], dropna=False).value.transform("mean")
            frame[f"{side}_context_residual"] = source - center
        residual_rows.append(frame)
    _csv(out / "spotlight_context_residuals.csv", pd.concat(residual_rows, ignore_index=True))
    # Per-gene scores and matched-panel null using the scalar U readout.
    urep = replicated[replicated.readout.eq("u_hidden_mean")].copy()
    consistent = urep[urep.direction_consistent]
    score_rows = []
    for gene in encoded.gene_name:
        values_by_model = {}
        for model in ["epoch_11", "random_42", "random_43", "random_44"]:
            x = consistent[(consistent.gene_name == gene) & (consistent.model == model)].replicated_abs_effect
            values_by_model[model] = float(x.median()) if len(x) else np.nan
        random_values = np.asarray([values_by_model[x] for x in ("random_42", "random_43", "random_44")], float)
        random_mean = float(np.nanmean(random_values)) if np.isfinite(random_values).any() else np.nan
        score_rows.append({"gene_name": gene, "context_effect": values_by_model["epoch_11"],
                           "random_mean_context_effect": random_mean,
                           "pretraining_increment": values_by_model["epoch_11"] - random_mean,
                           "eligible_epoch11_contrasts": int(((urep.gene_name == gene) & (urep.model == "epoch_11")).sum())})
    scores = pd.DataFrame(score_rows); _csv(out / "gene_context_scores.csv", scores)
    null = _draw_matched_panels(cfg, scores, candidates, panel.gene_name.tolist())
    _csv(out / "matched_panel_null.csv", null)
    observed = null.iloc[0]
    random = null[null.kind.eq("matched_random_panel")]
    epoch_panel = replicated[(replicated.model.eq("epoch_11")) & replicated.gene_name.isin(panel.gene_name)]
    summary = {
        "panel_median_context_effect": float(observed.median_context_effect),
        "matched_null_context_percentile": float((random.median_context_effect <= observed.median_context_effect).mean()),
        "panel_median_pretraining_increment": float(observed.median_pretraining_increment),
        "matched_null_increment_percentile": float((random.median_pretraining_increment <= observed.median_pretraining_increment).mean()),
        "panel_genes_with_context_score": int(observed.genes_with_context_score),
        "epoch11_direction_consistency": float(urep[urep.model.eq("epoch_11")].direction_consistent.mean()),
        "panel_direction_consistency_u": float(epoch_panel[epoch_panel.readout.eq("u_hidden_mean")].direction_consistent.mean()),
        "panel_direction_consistency_s": float(epoch_panel[epoch_panel.readout.eq("s_hidden_mean")].direction_consistent.mean()),
        "panel_direction_consistency_distance": float(epoch_panel[epoch_panel.readout.eq("us_cosine_distance")].direction_consistent.mean()),
        "panel_replicated_contrasts_per_readout": int(len(epoch_panel[epoch_panel.readout.eq("u_hidden_mean")])),
    }
    _json(out / "analysis_summary.json", summary)
    _json(out / "status.json", {"status": "analysis_complete", "updated": time.time()})
    return out


def _savefig(fig, out: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")


def plot(cfg: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    out = _out(cfg); sns.set_theme(style="whitegrid", context="talk", font_scale=.78)
    examples = pd.read_csv(out / "spotlight_cell_values.csv")
    spot = _spotlights(cfg); sample_order = sorted(examples["sample"].unique())
    for readout, suffix, ylabel in (("u_hidden_mean", "u", "U hidden mean over 128D"),
                                    ("s_hidden_mean", "s", "S hidden mean over 128D"),
                                    ("us_cosine_distance", "distance", "U–S cosine distance")):
        fig, axes = plt.subplots(len(spot), 2, figsize=(14, 20), sharex=False)
        for row, record in enumerate(spot.to_dict("records")):
            for col, sample in enumerate(sample_order):
                ax = axes[row, col]; d = examples[(examples.gene == record["gene"]) & (examples["sample"] == sample)]
                order = [record["type_a"], record["type_b"]]
                sns.violinplot(data=d, x="cell_type", y=readout, order=order, inner=None, cut=0,
                               palette=CELL_COLORS, alpha=.35, ax=ax)
                sns.boxplot(data=d, x="cell_type", y=readout, order=order, width=.28, showfliers=False,
                            whis=(10, 90), palette=CELL_COLORS, ax=ax)
                sns.stripplot(data=d, x="cell_type", y=readout, order=order, color="#20252B", size=2.4,
                              alpha=.42, jitter=.18, ax=ax)
                ax.set_title(f'{record["gene"]} | U={record["u_bin"]}, S={record["s_bin"]} | {sample}\n'
                             f'n={len(d[d.cell_type.eq(order[0])])}/{len(d[d.cell_type.eq(order[1])])}')
                ax.set_xlabel(""); ax.set_ylabel(ylabel if col == 0 else "")
                ax.tick_params(axis="x", rotation=15)
        figure_title = (
            "Identical target gene/value tokens, different whole-cell contexts\n"
            "Frozen epoch 11; full original token context"
        )
        if readout == "us_cosine_distance":
            figure_title += "; distance = 1 − cosine(hU, hS)"
        fig.suptitle(figure_title, fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, .97)); _savefig(fig, out, f"01_fixed_token_hidden_examples_{suffix}"); plt.close(fig)

    # Extended fixed-token distance view.  The bin panels intentionally use
    # every cell from the two compared cell types in the same sample, whereas
    # the distance panel uses only the exact target U/S-bin stratum.  Keeping
    # these populations explicit prevents the additional cells from silently
    # entering the controlled distance comparison.
    meta = pd.read_csv(out / "cells.csv")
    encoded = pd.read_csv(out / "encoded_genes.csv")
    encoded_lookup = {name: i for i, name in enumerate(encoded.gene_name)}
    encoded_bins = np.load(out / "encoded_gene_bins.npz")
    distance_lookup = examples.set_index(["cell_id", "gene"])["us_cosine_distance"]
    bin_rows = []
    for record in spot.to_dict("records"):
        gene_col = encoded_lookup[record["gene"]]
        for sample in sample_order:
            selected = meta["sample"].eq(sample) & meta.cell_type.isin([record["type_a"], record["type_b"]])
            for index in np.flatnonzero(selected.to_numpy()):
                u_bin = int(encoded_bins["u"][index, gene_col])
                s_bin = int(encoded_bins["s"][index, gene_col])
                key = (meta.cell_id.iloc[index], record["gene"])
                fixed = u_bin == int(record["u_bin"]) and s_bin == int(record["s_bin"])
                bin_rows.append({
                    "cell_id": meta.cell_id.iloc[index], "sample": sample,
                    "cell_type": meta.cell_type.iloc[index], "gene": record["gene"],
                    "u_bin": u_bin, "s_bin": s_bin,
                    "target_u_bin": int(record["u_bin"]), "target_s_bin": int(record["s_bin"]),
                    "is_fixed_stratum": fixed,
                    "us_cosine_distance": float(distance_lookup.get(key, np.nan)) if fixed else np.nan,
                })
    same_sample_bins = pd.DataFrame(bin_rows)
    _csv(out / "01B_same_sample_bin_plot_data.csv", same_sample_bins)
    fig, axes = plt.subplots(len(spot), 6, figsize=(28, 22), squeeze=False)
    for row, record in enumerate(spot.to_dict("records")):
        order = [record["type_a"], record["type_b"]]
        for sample_index, sample in enumerate(sample_order):
            block = 3 * sample_index
            data = same_sample_bins[
                same_sample_bins.gene.eq(record["gene"]) & same_sample_bins["sample"].eq(sample)
            ]
            for offset, (field, label, target) in enumerate((
                ("u_bin", "U bin", int(record["u_bin"])),
                ("s_bin", "S bin", int(record["s_bin"])),
            )):
                ax = axes[row, block + offset]
                sns.boxplot(data=data, x="cell_type", y=field, order=order, width=.45,
                            showfliers=False, whis=(10, 90), palette=CELL_COLORS, ax=ax)
                sns.stripplot(data=data, x="cell_type", y=field, order=order, color="#20252B",
                              size=1.6, alpha=.22, jitter=.22, ax=ax)
                ax.axhline(target, color="#C64B4B", ls="--", lw=1.4,
                           label=f"fixed target bin = {target}")
                ax.set_ylim(-.6, 15.7); ax.set_yticks([0, 5, 10, 15])
                ax.set_title(f"{record['gene']} | {sample}\n{label}, all compared cells")
                ax.set_xlabel(""); ax.set_ylabel("Token bin\n(0 = unobserved)" if offset == 0 else "")
                ax.tick_params(axis="x", rotation=20, labelsize=8)
                ax.legend(frameon=False, fontsize=7, loc="lower right")
            ax = axes[row, block + 2]
            fixed_data = data[data.is_fixed_stratum & data.us_cosine_distance.notna()]
            sns.violinplot(data=fixed_data, x="cell_type", y="us_cosine_distance", order=order,
                           inner=None, cut=0, palette=CELL_COLORS, alpha=.35, ax=ax)
            sns.boxplot(data=fixed_data, x="cell_type", y="us_cosine_distance", order=order,
                        width=.28, showfliers=False, whis=(10, 90), palette=CELL_COLORS, ax=ax)
            sns.stripplot(data=fixed_data, x="cell_type", y="us_cosine_distance", order=order,
                          color="#20252B", size=2.1, alpha=.38, jitter=.18, ax=ax)
            counts = [int(fixed_data.cell_type.eq(cell_type).sum()) for cell_type in order]
            ax.set_title(f"{record['gene']} | {sample}\nDistance at U={record['u_bin']}, S={record['s_bin']} | n={counts[0]}/{counts[1]}")
            ax.set_xlabel(""); ax.set_ylabel("U–S cosine distance")
            ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.suptitle(
        "Same-sample input-bin context and fixed-token U–S distance\n"
        "Bin panels: all cells from the compared cell types; distance panels: exact U/S-bin subset only",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, .975))
    _savefig(fig, out, "01B_fixed_token_distance_with_same_sample_bins"); plt.close(fig)

    replicated = pd.read_csv(out / "replicated_scalar_effects.csv")
    samples = sorted(pd.read_csv(out / "cells.csv")["sample"].unique())
    xcol, ycol = f"effect_{samples[0]}", f"effect_{samples[1]}"
    p = replicated[(replicated.model == "epoch_11") & (replicated.readout == "u_hidden_mean")]
    panel_names = set(pd.read_csv(out / "gene_panel.csv").gene_name)
    p = p[p.gene_name.isin(panel_names)]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(p[xcol], p[ycol], s=18 + 2 * p.minimum_group_size, c=np.where(p.direction_consistent, "#3A7D70", "#B8BDC4"), alpha=.72)
    for gene in spot.gene:
        q = p[p.gene_name.eq(gene)].sort_values("replicated_abs_effect", ascending=False).head(1)
        if len(q): ax.annotate(gene, (q[xcol].iloc[0], q[ycol].iloc[0]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    limit = max(abs(p[[xcol, ycol]].to_numpy()).max(), .1); ax.axhline(0, color="#777", lw=1); ax.axvline(0, color="#777", lw=1)
    ax.plot([-limit, limit], [-limit, limit], ls="--", color="#777", lw=1)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel=f"Cohen's d | {samples[0]}", ylabel=f"Cohen's d | {samples[1]}",
           title="Cross-sample replication of fixed-token U-hidden effects\n20-gene biology-defined panel; all eligible bin/type contrasts")
    fig.tight_layout(); _savefig(fig, out, "02_panel_cross_sample_replicability"); plt.close(fig)
    vector = pd.read_csv(out / "replicated_vector_effects.csv")
    scalar = replicated[(replicated.gene_name.isin(spot.gene)) & (replicated.readout == "u_hidden_mean")].copy()
    fixed = spot.rename(columns={"gene": "gene_name"})
    scalar = scalar.merge(fixed, on=["gene_name", "u_bin", "s_bin", "type_a", "type_b"])
    vec = vector[(vector.gene_name.isin(spot.gene)) & (vector.readout == "u_hidden_128d")].merge(
        fixed, on=["gene_name", "u_bin", "s_bin", "type_a", "type_b"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    gene_order = spot.gene.tolist()
    def attribution_panel(ax, frame, title, ylabel):
        for position, gene in enumerate(gene_order):
            values = frame[frame.gene_name.eq(gene)].set_index("model").replicated_abs_effect
            random_values = np.asarray([values.get(f"random_{seed}", np.nan) for seed in (42, 43, 44)], float)
            valid_random = random_values[np.isfinite(random_values)]
            if len(valid_random):
                ax.vlines(position, valid_random.min(), valid_random.max(), color="#C58C37", lw=2.2, alpha=.8, zorder=2)
                ax.scatter(position + np.asarray([-.055, 0, .055])[:len(valid_random)], valid_random,
                           s=45, color="#D5A13D", edgecolor="white", lw=.6, zorder=3)
            if np.isfinite(values.get("input_embedding", np.nan)):
                ax.scatter(position - .18, values["input_embedding"], s=58, color="#777777", edgecolor="white", lw=.7, zorder=4)
            if np.isfinite(values.get("epoch_11", np.nan)):
                ax.scatter(position + .18, values["epoch_11"], s=72, color="#2E7D6D", edgecolor="white", lw=.7, zorder=4)
        ax.set_xticks(range(len(gene_order)), gene_order, rotation=20)
        ax.set(title=title, ylabel=ylabel, xlabel="")
    attribution_panel(axes[0], scalar, "A  Fixed all-ones hidden projection", "Replicated |Cohen's d|")
    attribution_panel(axes[1], vec, "B  Full 128-D hidden vector", "Replicated standardized centroid RMS")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=8, label="Input embedding"),
        Line2D([0], [0], marker="o", color="#C58C37", markerfacecolor="#D5A13D", markeredgecolor="white", markersize=8, label="Random-init seeds 42/43/44 (range)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2E7D6D", markeredgecolor="white", markersize=9, label="Pretrained epoch 11"),
    ]
    axes[0].legend(handles=legend, bbox_to_anchor=(0, -.2), loc="upper left", ncol=3, frameon=False)
    fig.suptitle("Input, random architecture, and pretrained contextual effects"); fig.tight_layout(rect=(0, .08, 1, .95))
    _savefig(fig, out, "03_input_random_pretrained_attribution"); plt.close(fig)
    coords = pd.read_csv(cfg["data"]["umap_coordinates_path"]).query("representation == 'mu_profile'")
    residual = pd.read_csv(out / "spotlight_context_residuals.csv").merge(coords[["cell_id", "UMAP1", "UMAP2"]], on="cell_id", validate="many_to_one")
    fig, axes = plt.subplots(len(spot), 3, figsize=(16, 24))
    for row, gene in enumerate(spot.gene):
        d = residual[residual.gene_name.eq(gene)]
        for col, (field, title, cmap) in enumerate((("u_bin", "U bin", "viridis"), ("s_bin", "S bin", "viridis"),
                                                   ("u_context_residual", "U hidden context residual", "coolwarm"))):
            ax = axes[row, col]; values = d[field].to_numpy(float)
            if "residual" in field:
                limit = np.nanpercentile(abs(values), 98); norm = matplotlib.colors.TwoSlopeNorm(0, -max(limit, 1e-8), max(limit, 1e-8))
                scatter = ax.scatter(d.UMAP1, d.UMAP2, c=values, s=5, cmap=cmap, norm=norm, linewidths=0)
            else: scatter = ax.scatter(d.UMAP1, d.UMAP2, c=values, s=5, cmap=cmap, vmin=0, vmax=15, linewidths=0)
            ax.set(xticks=[], yticks=[], xlabel="UMAP1", ylabel="UMAP2" if col == 0 else "", title=f"{gene} | {title}")
            fig.colorbar(scatter, ax=ax, fraction=.035, pad=.02)
    fig.suptitle("Common Mu-profile UMAP coordinates: input bins versus contextual hidden residual\nUMAP is descriptive; residual removes sample × exact U/S-bin mean")
    fig.tight_layout(rect=(0, 0, 1, .97)); _savefig(fig, out, "04_gene_context_umap_overlays"); plt.close(fig)
    null = pd.read_csv(out / "matched_panel_null.csv"); observed = null[null.kind.eq("forebrain_panel")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, field, title in ((axes[0], "median_context_effect", "Context effect"),
                             (axes[1], "median_pretraining_increment", "Epoch11 − random-init")):
        sns.histplot(null[null.kind.eq("matched_random_panel")][field], bins=35, color="#9CA5B0", ax=ax)
        ax.axvline(observed[field], color="#C64B4B", lw=2.5, label="Forebrain panel")
        ax.set(title=title, xlabel="Panel median", ylabel="Matched panel draws"); ax.legend(frameon=False)
    fig.suptitle("Biology-defined Forebrain panel versus 1,000 input-matched random panels")
    fig.tight_layout(); _savefig(fig, out, "05_forebrain_panel_vs_matched_null"); plt.close(fig)
    p = replicated[(replicated.model == "epoch_11") & (replicated.gene_name.isin(spot.gene))].merge(
        fixed, on=["gene_name", "u_bin", "s_bin", "type_a", "type_b"])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=p, x="gene_name", y="replicated_abs_effect", hue="readout",
                hue_order=["u_hidden_mean", "s_hidden_mean", "us_cosine_distance"],
                palette=["#557DBB", "#E49A3A", "#3A7D70"], ax=ax)
    ax.set(title="Which side drives the fixed-token contextual effect?", xlabel="", ylabel="Replicated |Cohen's d|")
    ax.legend(title="Readout", frameon=False); fig.tight_layout(); _savefig(fig, out, "06_u_s_and_distance_context_effects"); plt.close(fig)
    summary = json.loads((out / "analysis_summary.json").read_text())
    report = f"""# Forebrain固定Token的逐基因Contextual Hidden审计

## 结果入口

- [固定token实例：U hidden](01_fixed_token_hidden_examples_u.png)；[S hidden](01_fixed_token_hidden_examples_s.png)；[U-S distance](01_fixed_token_hidden_examples_distance.png)
- [同sample全部待比较细胞的U/S bins＋固定token distance](01B_fixed_token_distance_with_same_sample_bins.png)
- [跨sample复现](02_panel_cross_sample_replicability.png)
- [Input／random-init／epoch11归因](03_input_random_pretrained_attribution.png)
- [共同Mu-UMAP上的输入与hidden residual](04_gene_context_umap_overlays.png)
- [前脑panel与匹配随机panel](05_forebrain_panel_vs_matched_null.png)
- [U、S与cosine distance对照](06_u_s_and_distance_context_effects.png)

## 固定口径

- 1,720个Forebrain细胞；完整原始token context；无top-k、无重新训练。
- 20个生物学定义基因全部报告；6个实例只用于提高图的可读性。
- 固定的是模型实际接收的gene ID、U/S bins和modality token，不是连续Mu/Ms或原始count逐值等同。
- hidden mean与hidden sum只相差常数128；完整128维结果用于检查固定投影是否造成误导。

## 当前汇总

- 全部可评估基因中，Epoch11 U-hidden合格对比的跨sample方向一致率：**{summary['epoch11_direction_consistency']:.3f}**。
- 前脑panel的{summary['panel_replicated_contrasts_per_readout']}个可复现exact-bin对比中，U hidden／S hidden／U-S distance方向一致率分别为 **{summary['panel_direction_consistency_u']:.3f}／{summary['panel_direction_consistency_s']:.3f}／{summary['panel_direction_consistency_distance']:.3f}**。
- 前脑panel median context effect：**{summary['panel_median_context_effect']:.4f}**；位于匹配随机panel的 **{summary['matched_null_context_percentile']:.1%}** 分位。
- 前脑panel median epoch11−random增量：**{summary['panel_median_pretraining_increment']:.4f}**；位于匹配随机panel的 **{summary['matched_null_increment_percentile']:.1%}** 分位。
- 20个panel基因中有 **{summary['panel_genes_with_context_score']}** 个满足严格的双sample exact-bin比较条件；其余不足条件的基因保留为NA。

## 逐图解读

1. `01`：在目标gene ID和U/S bins完全相同的前提下，六个实例的epoch11 U hidden、S hidden及U-S distance仍按cell type分离，并在两个sample中方向一致。这是context dependence的直观证据。Distance图与U/S图使用完全相同的barcode集合。`01B`额外展示同sample、相同两个cell type的全部细胞bin分布；这些额外细胞不进入右侧fixed-token distance统计。
2. `02`：绝大多数panel exact-bin效应落在同号象限并靠近对角线，表明效应不是单个sample偶然产生；灰点是方向不一致的反例，继续保留。
3. `03`：Input embedding为零效应；random-init已经产生context effect，而epoch11只在部分基因／读出上更强。完整128维与hidden mean的排名并不完全相同，说明单一固定投影会改变结论。
4. `04`：在同一Mu-profile UMAP坐标上，hidden residual在扣除sample与exact-bin均值后仍呈现空间结构。该图只作描述，不参与效应检验。
5. `05`：红线没有进入匹配随机panel右尾，故不支持前脑基因集合特异增强；contextualization更像是广泛的逐基因现象。
6. `06`：U hidden、S hidden和U-S cosine distance均可对context变化作出响应，但不同基因主要由哪一侧驱动并不一致；distance不能被简化为单侧hidden变化。

## 解释边界

若epoch11不超过三个random-init，则只能说明Transformer结构和tokenization产生context sensitivity。若前脑panel不超过匹配随机panel，则contextualization是广泛现象而非前脑基因特异。任何正结果都不等价于表达独立性、因果调控或RNA动力学方向。
"""
    (out / "REPORT_CN.md").write_text(report, encoding="utf-8")
    _json(out / "status.json", {"status": "plots_complete", "updated": time.time()})
    return out


def validate(cfg: dict) -> Path:
    from PIL import Image
    out = _out(cfg); checks = {}
    meta = pd.read_csv(out / "cells.csv"); panel = pd.read_csv(out / "gene_panel.csv")
    encoded = pd.read_csv(out / "encoded_genes.csv"); bins = np.load(out / "encoded_gene_bins.npz")
    checks["cell_identity"] = len(meta) == 1720 and meta.cell_id.is_unique
    checks["panel_identity"] = len(panel) == 20 and panel.gene_name.is_unique and int(encoded.is_panel.sum()) == 20
    checks["bin_shape"] = bins["u"].shape == bins["s"].shape == (1720, len(encoded))
    checks["all_models_complete"] = all((out / "models" / model / "complete.json").exists() for model in MODEL_ORDER)
    if checks["all_models_complete"]:
        masks = []
        for model in MODEL_ORDER:
            folder = out / "models" / model
            u = np.load(folder / "scalar_u.npy", mmap_mode="r"); s = np.load(folder / "scalar_s.npy", mmap_mode="r")
            d = np.load(folder / "distance.npy", mmap_mode="r")
            checks[f"{model}_shape"] = u.shape == s.shape == d.shape == (1720, len(encoded))
            checks[f"{model}_finite"] = bool(np.isfinite(u[np.isfinite(u)]).all() and np.isfinite(s[np.isfinite(s)]).all() and np.isfinite(d[np.isfinite(d)]).all())
            checks[f"{model}_distance_legal"] = bool(np.nanmin(d) >= -1e-6 and np.nanmax(d) <= 2 + 1e-6)
            masks.append(np.isfinite(d))
        checks["missing_masks_identical"] = all(np.array_equal(masks[0], x) for x in masks[1:])
        inp = np.load(out / "models/input_embedding/scalar_u.npy")
        max_range = 0.0
        for col in range(len(encoded)):
            code = bins["u"][:, col].astype(int) * 16 + bins["s"][:, col].astype(int)
            for sample in meta["sample"].unique():
                for value in np.unique(code[meta["sample"].eq(sample)]):
                    idx = np.flatnonzero(meta["sample"].eq(sample).to_numpy() & (code == value) & np.isfinite(inp[:, col]))
                    if len(idx): max_range = max(max_range, float(np.ptp(inp[idx, col])))
        checks["input_embedding_exact_token_invariant"] = max_range <= float(cfg["analysis"]["input_invariance_tolerance"])
        checks["input_embedding_max_exact_token_range"] = max_range
    figure_stems = ["01_fixed_token_hidden_examples_u", "01_fixed_token_hidden_examples_s",
                    "01_fixed_token_hidden_examples_distance",
                    "01B_fixed_token_distance_with_same_sample_bins",
                    "02_panel_cross_sample_replicability", "03_input_random_pretrained_attribution",
                    "04_gene_context_umap_overlays", "05_forebrain_panel_vs_matched_null",
                    "06_u_s_and_distance_context_effects"]
    checks["figures_complete"] = all((out / f"{x}.png").exists() and (out / f"{x}.pdf").exists() for x in figure_stems)
    if checks["figures_complete"]:
        for stem in figure_stems:
            with Image.open(out / f"{stem}.png") as image: image.verify()
            if not (out / f"{stem}.pdf").read_bytes().startswith(b"%PDF"): raise ValueError(f"Invalid PDF {stem}")
    boolean = [v for v in checks.values() if isinstance(v, bool)]
    status = "passed" if all(boolean) else "failed"
    _json(out / "validation.json", {"status": status, "checks": checks})
    _json(out / "status.json", {"status": "complete" if status == "passed" else "validation_failed", "updated": time.time()})
    if status != "passed": raise AssertionError(checks)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "smoke", "encode", "analyze", "plot", "validate", "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Physical cuda:0/cuda:2/cuda:3; GPU1 is forbidden")
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.device: cfg["device"] = args.device
    out = _out(cfg); out.mkdir(parents=True, exist_ok=True)
    cache = out / ".cache"; cache.mkdir(exist_ok=True)
    for key, sub in (("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")):
        path = cache / sub; path.mkdir(exist_ok=True); os.environ[key] = str(path)
    stages = ["audit", "smoke", "encode", "analyze", "plot", "validate"] if args.stage == "all" else [args.stage]
    try:
        with threadpool_limits(limits=4):
            for stage in stages:
                globals()[stage](cfg)
    except Exception as exc:
        _json(out / "status.json", {"status": "failed", "stage": stage, "error": repr(exc), "updated": time.time()})
        raise


if __name__ == "__main__":
    main()

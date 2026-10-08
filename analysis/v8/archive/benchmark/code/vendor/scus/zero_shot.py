from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .data.tokens import tokenize_pair
from .runtime import cosine_distance, provenance, stage_directory, write_json


def load_encoder(cfg: dict, device: torch.device):
    from .models import MaskedModel

    with Path(cfg["data"]["vocab"]).open() as handle:
        vocab_size = len(json.load(handle))
    model_cfg = cfg.get("model", {})
    model = MaskedModel.load_from_checkpoint(
        model_cfg["checkpoint"], vocab_size=vocab_size,
        bin_size=int(model_cfg.get("bin_size", 15)), mask_ratio=0.0,
        predict_mode="gene", map_location=device,
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _collate(records, device):
    length = max(1, max(len(record[1]) for record in records))
    shape = (len(records), length)
    gene = np.zeros(shape, np.int64)
    value = np.zeros(shape, np.int64)
    splice = np.zeros(shape, np.int64)
    for row, record in enumerate(records):
        n = len(record[1])
        gene[row, :n], value[row, :n], splice[row, :n] = record[1], record[2], record[3]
    return tuple(torch.as_tensor(array, device=device) for array in (gene, value, splice))


def encode(cfg: dict, tag: str | None = None, start: int = 0, stop: int | None = None,
           device_name: str | None = None, projector=None, stage: str = "zero-shot",
           prepare_tag: str | None = None) -> Path:
    import anndata as ad

    device = torch.device(device_name or cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model = load_encoder(cfg, device)
    data_cfg = cfg["data"]
    adata = ad.read_h5ad(data_cfg["input"], backed="r")
    prep = stage_directory(cfg, "prepare", prepare_tag or tag)
    mapping = pd.read_csv(prep / "gene_mapping.csv")
    cells = pd.read_parquet(prep / "cells.parquet")
    columns = mapping.source_column.to_numpy(int)
    vocab_ids = mapping.vocab_id.to_numpy(int)
    stop = min(adata.n_obs, stop if stop is not None else adata.n_obs)
    start = max(0, start)
    out = provenance(cfg, stage, tag, {"range": [start, stop], "device": str(device), "encoder_frozen": True, "prepare_tag": prepare_tag or tag})
    settings = cfg.get("zero_shot", {})
    max_genes = int(settings.get("max_genes", 1000))
    batch_size = int(settings.get("batch_size", 4))
    bin_size = int(cfg.get("model", {}).get("bin_size", 15))
    seed = int(cfg.get("seed", 42))
    reservoir_path = prep / "reservoir_cells.npy"
    reservoir = set(np.load(reservoir_path).tolist()) if reservoir_path.exists() and projector is None else set()
    pairs_per_cell = int(cfg.get("align", {}).get("reservoir_pairs_per_cell", 512))

    indptr, indices, distances = [0], [], []
    pooled = np.zeros((stop - start, int(cfg.get("model", {}).get("embed_dim", 128))), np.float32)
    scores, cache_u, cache_s, cache_cell = [], [], [], []
    mu_layer, ms_layer = data_cfg.get("mu_layer", "Mu"), data_cfg.get("ms_layer", "Ms")
    for batch_start in range(start, stop, batch_size):
        batch_stop = min(stop, batch_start + batch_size)
        mu_block = adata.layers[mu_layer][batch_start:batch_stop, :]
        ms_block = adata.layers[ms_layer][batch_start:batch_stop, :]
        if hasattr(mu_block, "toarray"): mu_block = mu_block.toarray()
        if hasattr(ms_block, "toarray"): ms_block = ms_block.toarray()
        mu = np.asarray(mu_block, np.float32)[:, columns]
        ms = np.asarray(ms_block, np.float32)[:, columns]
        records = [tokenize_pair(mu[j], ms[j], vocab_ids, max_genes, bin_size, str(cells.iloc[i].cell_id), seed)
                   for j, i in enumerate(range(batch_start, batch_stop))]
        gene, value, splice = _collate(records, device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model._encode(gene, value, splice, return_all=True).float()
            if projector is not None:
                hidden = projector(hidden)
        for row, (selected, _, value_bins, _, valid) in enumerate(records):
            cell_index = batch_start + row
            n_tokens = 2 * len(selected)
            token_hidden = hidden[row, :n_tokens]
            present = torch.as_tensor(value_bins > 0, device=device)
            if present.any():
                pooled[cell_index - start] = token_hidden[present].mean(0).cpu().numpy()
            valid_index = np.flatnonzero(valid)
            hu, hs = token_hidden[0:n_tokens:2][valid_index], token_hidden[1:n_tokens:2][valid_index]
            distance = cosine_distance(hu, hs).cpu().numpy().astype(np.float32)
            indices.extend(selected[valid_index].tolist())
            distances.extend(distance.tolist())
            indptr.append(len(distances))
            scores.append(float(np.median(distance)) if len(distance) else np.nan)
            if cell_index in reservoir and len(valid_index):
                take = valid_index[:pairs_per_cell]
                cache_u.append(token_hidden[0:n_tokens:2][take].cpu().numpy().astype(np.float16))
                cache_s.append(token_hidden[1:n_tokens:2][take].cpu().numpy().astype(np.float16))
                cache_cell.extend([cell_index] * len(take))

    np.save(out / "indptr.npy", np.asarray(indptr, np.int64))
    np.save(out / "indices.npy", np.asarray(indices, np.int32))
    np.save(out / "data.npy", np.asarray(distances, np.float32))
    np.save(out / "pooled_embedding.npy", pooled)
    result_cells = cells.iloc[start:stop].copy()
    result_cells["cell_score"] = scores
    result_cells.to_parquet(out / "cell_scores.parquet", index=False)
    if cache_u:
        np.save(out / "reservoir_hu.npy", np.concatenate(cache_u))
        np.save(out / "reservoir_hs.npy", np.concatenate(cache_s))
        np.save(out / "reservoir_cell.npy", np.asarray(cache_cell, np.int64))
    write_json(out / "status.json", {"status": "complete", "cells": stop-start, "pairs": len(distances), "range": [start, stop], "encoder_frozen": True, "distance": "1-cosine"})
    return out


def merge_parts(cfg: dict, stage: str, output_tag: str, part_tags: list[str]) -> Path:
    parts = []
    for tag in part_tags:
        path = stage_directory(cfg, stage, tag)
        status = json.loads((path / "status.json").read_text())
        parts.append((status["range"], path, status))
    parts.sort(key=lambda item: item[0][0])
    expected = parts[0][0][0]
    for limits, _, _ in parts:
        if limits[0] != expected:
            raise ValueError(f"Non-contiguous ranges: expected {expected}, got {limits}")
        expected = limits[1]
    out = provenance(cfg, stage, output_tag, {"merged_parts": part_tags})
    indptr, total, indices, data, pooled, cells = [0], 0, [], [], [], []
    caches = {name: [] for name in ("reservoir_hu", "reservoir_hs", "reservoir_cell")}
    for _, path, _ in parts:
        local = np.load(path / "indptr.npy")
        indptr.extend((local[1:] + total).tolist())
        total += int(local[-1])
        indices.append(np.load(path / "indices.npy")); data.append(np.load(path / "data.npy"))
        pooled.append(np.load(path / "pooled_embedding.npy")); cells.append(pd.read_parquet(path / "cell_scores.parquet"))
        for name in caches:
            if (path / f"{name}.npy").exists(): caches[name].append(np.load(path / f"{name}.npy"))
    np.save(out / "indptr.npy", np.asarray(indptr, np.int64)); np.save(out / "indices.npy", np.concatenate(indices)); np.save(out / "data.npy", np.concatenate(data)); np.save(out / "pooled_embedding.npy", np.concatenate(pooled))
    pd.concat(cells, ignore_index=True).to_parquet(out / "cell_scores.parquet", index=False)
    for name, arrays in caches.items():
        if arrays: np.save(out / f"{name}.npy", np.concatenate(arrays))
    write_json(out / "status.json", {"status": "complete", "cells": sum(x[2]["cells"] for x in parts), "pairs": total, "range": [parts[0][0][0], parts[-1][0][1]]})
    return out

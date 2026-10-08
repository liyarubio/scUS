from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .runtime import cosine_distance, write_json


@dataclass(frozen=True)
class TokenPanel:
    gene_ids: np.ndarray
    u_bins: np.ndarray
    s_bins: np.ndarray
    target_positions: np.ndarray
    cell_ids: np.ndarray
    cell_types: np.ndarray
    source_cells: np.ndarray


def interleave_tokens(gene_ids: np.ndarray, u_bins: np.ndarray, s_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gene_ids.shape != u_bins.shape or gene_ids.shape != s_bins.shape:
        raise ValueError("gene_ids, u_bins, and s_bins must have identical shapes")
    gene = np.repeat(gene_ids, 2, axis=1).astype(np.int64, copy=False)
    value = np.empty(gene.shape, np.int64)
    value[:, 0::2], value[:, 1::2] = u_bins, s_bins
    splice = np.broadcast_to(np.tile(np.array([0, 1], np.int64), gene_ids.shape[1]), gene.shape).copy()
    return gene, value, splice


def paired_representations(hidden: torch.Tensor, u_bins: np.ndarray, s_bins: np.ndarray) -> dict[str, np.ndarray]:
    if hidden.ndim != 3 or hidden.shape[1] != 2 * u_bins.shape[1]:
        raise ValueError("hidden must contain interleaved U/S tokens")
    hu, hs = hidden[:, 0::2].float(), hidden[:, 1::2].float()
    valid = torch.as_tensor((u_bins > 0) & (s_bins > 0), device=hidden.device)
    denominator = valid.sum(1, keepdim=True).clamp(min=1)

    def mean(value: torch.Tensor) -> torch.Tensor:
        return (value * valid.unsqueeze(-1)).sum(1) / denominator

    distance = cosine_distance(hu, hs).masked_fill(~valid, float("nan"))
    scalar = torch.nanmedian(distance, dim=1).values
    return {
        "u_only": mean(hu).cpu().numpy(),
        "s_only": mean(hs).cpu().numpy(),
        "state": mean((hu + hs) / 2).cpu().numpy(),
        "kinetic": mean(hu - hs).cpu().numpy(),
        "absolute_kinetic": mean(torch.abs(hu - hs)).cpu().numpy(),
        "distance_profile": distance.cpu().numpy(),
        "distance_scalar": scalar.cpu().numpy(),
    }


def matched_donors(cell_types: np.ndarray, seed: int = 42) -> np.ndarray:
    """Choose a reproducible non-self donor within each cell type."""
    cell_types = np.asarray(cell_types).astype(str)
    donors = np.full(len(cell_types), -1, np.int64)
    rng = np.random.default_rng(seed)
    for label in np.unique(cell_types):
        members = np.flatnonzero(cell_types == label)
        if len(members) < 2:
            raise ValueError(f"Cell type {label!r} has no non-self matched donor")
        order = members.copy()
        rng.shuffle(order)
        donors[order] = np.roll(order, 1)
    if np.any(donors < 0) or np.any(donors == np.arange(len(donors))):
        raise AssertionError("Matched donor construction produced an invalid donor")
    return donors


def context_condition(panel: TokenPanel, condition: str, donors: np.ndarray | None = None,
                      seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alter surrounding tokens while preserving every target token exactly."""
    genes, u_bins, s_bins = panel.gene_ids.copy(), panel.u_bins.copy(), panel.s_bins.copy()
    targets = np.asarray(panel.target_positions, np.int64)
    target_genes = genes[:, targets].copy()
    target_u, target_s = u_bins[:, targets].copy(), s_bins[:, targets].copy()
    non_target = np.setdiff1d(np.arange(genes.shape[1]), targets)
    if condition == "full":
        pass
    elif condition == "target_panel_only":
        genes[:, non_target] = 0
        u_bins[:, non_target] = 0
        s_bins[:, non_target] = 0
    elif condition == "matched_context":
        if donors is None:
            raise ValueError("matched_context requires donor indices")
        genes, u_bins, s_bins = genes[donors].copy(), u_bins[donors].copy(), s_bins[donors].copy()
        genes[:, targets], u_bins[:, targets], s_bins[:, targets] = target_genes, target_u, target_s
        for row in range(len(genes)):
            duplicates = np.isin(genes[row], target_genes[row])
            duplicates[targets] = False
            genes[row, duplicates] = 0
            u_bins[row, duplicates] = 0
            s_bins[row, duplicates] = 0
    elif condition == "within_cell_shuffle":
        rng = np.random.default_rng(seed)
        for row in range(len(genes)):
            order = non_target.copy()
            rng.shuffle(order)
            u_bins[row, non_target] = u_bins[row, order]
            s_bins[row, non_target] = s_bins[row, order]
    else:
        raise ValueError(f"Unknown context condition: {condition}")
    if not (
        np.array_equal(genes[:, targets], target_genes)
        and np.array_equal(u_bins[:, targets], target_u)
        and np.array_equal(s_bins[:, targets], target_s)
    ):
        raise AssertionError("A context ablation changed a protected target token")
    return genes, u_bins, s_bins


def query_gpu(index: int = 0) -> dict[str, float]:
    fields = "index,memory.total,memory.used,memory.free,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    values = [float(value.strip()) for value in result.stdout.strip().split(",")]
    return dict(zip(["gpu_id", "memory_total_mb", "memory_used_mb", "memory_free_mb", "utilization"], values))


def resource_preflight(index: int = 0, samples: int = 12, interval_seconds: float = 10,
                       minimum_free_mb: float = 40_000, maximum_mean_utilization: float = 20,
                       query: Callable[[int], dict[str, float]] = query_gpu) -> tuple[bool, pd.DataFrame, str]:
    rows = []
    for sample in range(samples):
        row = query(index)
        row.update({"timestamp": time.time(), "sample": sample, "phase": "preflight"})
        rows.append(row)
        if sample + 1 < samples:
            time.sleep(interval_seconds)
    frame = pd.DataFrame(rows)
    enough_memory = bool((frame.memory_free_mb >= minimum_free_mb).all())
    low_utilization = bool(frame.utilization.mean() < maximum_mean_utilization)
    if not enough_memory:
        return False, frame, f"free memory fell below {minimum_free_mb:.0f} MB"
    if not low_utilization:
        return False, frame, f"mean utilization {frame.utilization.mean():.1f}% is not below {maximum_mean_utilization:.1f}%"
    return True, frame, "passed"


def _tiny_model(seed: int = 42):
    from .models import MaskedModel
    torch.manual_seed(seed)
    model = MaskedModel(vocab_size=128, bin_size=15, embed_dim=16, num_heads=4, num_layers=1, mask_ratio=0)
    model.eval()
    return model


def run_cpu_smoke(output_dir: str | Path, seed: int = 42) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cells, genes = 32, 12
    gene_ids = np.broadcast_to(np.arange(1, genes + 1), (cells, genes)).copy()
    u_bins = rng.integers(1, 16, (cells, genes), dtype=np.int64)
    s_bins = rng.integers(1, 16, (cells, genes), dtype=np.int64)
    labels = np.repeat(np.array(["a", "b", "c", "d"]), 8)
    panel = TokenPanel(gene_ids, u_bins, s_bins, np.array([0, 1]), np.array([f"cell_{i}" for i in range(cells)]), labels, np.arange(cells))
    donors = matched_donors(labels, seed)
    conditions = ["full", "target_panel_only", "matched_context", "within_cell_shuffle"]
    model = _tiny_model(seed)
    records = []
    for condition in conditions:
        g, u, s = context_condition(panel, condition, donors, seed)
        token_gene, token_value, token_splice = interleave_tokens(g, u, s)
        with torch.inference_mode():
            hidden = model._encode(torch.as_tensor(token_gene), torch.as_tensor(token_value), torch.as_tensor(token_splice), return_all=True)
        representations = paired_representations(hidden, u, s)
        records.append({
            "condition": condition,
            "finite": bool(np.isfinite(representations["distance_profile"][(u > 0) & (s > 0)]).all()),
            "distance_min": float(np.nanmin(representations["distance_profile"])),
            "distance_max": float(np.nanmax(representations["distance_profile"])),
            "state_shape": list(representations["state"].shape),
        })
    pair_rows = []
    for row in range(cells):
        for position in panel.target_positions:
            g = gene_ids[row:row + 1, position:position + 1]
            u = u_bins[row:row + 1, position:position + 1]
            s = s_bins[row:row + 1, position:position + 1]
            tg, tv, ts = interleave_tokens(g, u, s)
            with torch.inference_mode():
                hidden = model._encode(torch.as_tensor(tg), torch.as_tensor(tv), torch.as_tensor(ts), return_all=True)
            pair_rows.append({"cell": row, "target_position": int(position), "distance": float(cosine_distance(hidden[:, 0], hidden[:, 1]))})
    pd.DataFrame(pair_rows).to_csv(out / "pair_only.csv", index=False)
    summary = {
        "status": "passed" if all(row["finite"] and row["distance_min"] >= -1e-5 and row["distance_max"] <= 2 + 1e-5 for row in records) else "failed",
        "seed": seed, "cells": cells, "genes": genes, "conditions": records,
        "donor_nonself": bool(np.all(donors != np.arange(cells))), "random_checkpoint_loaded": False,
    }
    write_json(out / "cpu_smoke.json", summary)
    return out


def _load_cache_panel(cache_dir: Path, vocab_path: Path, cells: int, genes_per_cell: int,
                      targets: int, seed: int) -> tuple[TokenPanel, np.ndarray, np.ndarray]:
    cell_idx = np.load(cache_dir / "cell_idx.int32.npy", mmap_mode="r")
    gene_idx = np.load(cache_dir / "gene_idx.int32.npy", mmap_mode="r")
    u_all = np.load(cache_dir / "u_bin.uint8.npy", mmap_mode="r")
    s_all = np.load(cache_dir / "s_bin.uint8.npy", mmap_mode="r")
    names = np.load(cache_dir / "gene_names.npy", allow_pickle=True).astype(str)
    barcodes = np.load(cache_dir / "barcodes.npy", allow_pickle=True).astype(str)
    cell_types = np.load(cache_dir / "cell_types.npy", allow_pickle=True).astype(str)
    with vocab_path.open() as handle:
        vocab = {str(key).upper(): int(value) for key, value in json.load(handle).items()}
    local_to_vocab = np.asarray([vocab.get(name.upper(), 0) for name in names], np.int64)
    counts = np.bincount(np.asarray(cell_idx), minlength=len(barcodes))
    indptr = np.r_[0, np.cumsum(counts)]
    labels, label_counts = np.unique(cell_types, return_counts=True)
    eligible_types = {label for label, count in zip(labels, label_counts) if count >= 2}
    eligible = np.asarray([i for i, label in enumerate(cell_types) if label in eligible_types])
    rng = np.random.default_rng(seed)
    selected_cells = []
    cap = int(np.ceil(cells / max(1, len(eligible_types))))
    for label in sorted(eligible_types):
        members = eligible[cell_types[eligible] == label].copy()
        rng.shuffle(members)
        selected_cells.extend(members[:cap].tolist())
    selected_cells = np.asarray(selected_cells[:cells], np.int64)
    if len(selected_cells) != cells:
        raise ValueError(f"Cache has only {len(selected_cells)} eligible sampled cells; requested {cells}")

    per_cell = []
    common = None
    for cell in selected_cells:
        sl = slice(indptr[cell], indptr[cell + 1])
        valid = local_to_vocab[np.asarray(gene_idx[sl])] > 0
        local = np.asarray(gene_idx[sl])[valid]
        per_cell.append((local, np.asarray(u_all[sl])[valid], np.asarray(s_all[sl])[valid]))
        current = set(local.tolist())
        common = current if common is None else common.intersection(current)
    if common is None or len(common) < targets:
        raise ValueError(f"Only {0 if common is None else len(common)} genes are common to all sampled cells")
    common_array = np.asarray(sorted(common), np.int64)
    common_scores = np.zeros(len(common_array), float)
    for local, u, s in per_cell:
        lookup = {int(g): i for i, g in enumerate(local)}
        common_scores += np.asarray([u[lookup[int(g)]] + s[lookup[int(g)]] for g in common_array])
    target_local = common_array[np.lexsort((common_array, -common_scores))[:targets]]

    gene_matrix = np.zeros((cells, genes_per_cell), np.int64)
    u_matrix = np.zeros_like(gene_matrix)
    s_matrix = np.zeros_like(gene_matrix)
    target_positions = np.arange(targets, dtype=np.int64)
    for row, (local, u, s) in enumerate(per_cell):
        lookup = {int(g): i for i, g in enumerate(local)}
        remaining = np.setdiff1d(local, target_local, assume_unique=False)
        score = np.asarray([u[lookup[int(g)]] + s[lookup[int(g)]] for g in remaining])
        picked = remaining[np.lexsort((remaining, -score))[:genes_per_cell - targets]]
        chosen = np.r_[target_local, picked]
        source = np.asarray([lookup[int(g)] for g in chosen])
        gene_matrix[row] = local_to_vocab[chosen]
        u_matrix[row], s_matrix[row] = u[source], s[source]
    panel = TokenPanel(gene_matrix, u_matrix, s_matrix, target_positions, barcodes[selected_cells], cell_types[selected_cells], selected_cells)
    return panel, target_local, names[target_local]


def _random_like(model, seed: int):
    from .models import MaskedModel
    settings = dict(model.hparams) if hasattr(model, "hparams") else {}
    allowed = {"vocab_size", "bin_size", "embed_dim", "lr", "num_heads", "num_layers", "mask_ratio", "num_datasets", "use_batch_embed", "freeze_batch_bias", "predict_mode"}
    torch.manual_seed(seed)
    random_model = MaskedModel(**{key: value for key, value in settings.items() if key in allowed})
    random_model.eval()
    return random_model


def _encode_batches(model, genes: np.ndarray, u_bins: np.ndarray, s_bins: np.ndarray, device: torch.device,
                    batch_size: int, condition: str, checkpoint: str, resource_rows: list[dict]) -> tuple[np.ndarray, float]:
    outputs, started = [], time.perf_counter()
    token_gene, token_value, token_splice = interleave_tokens(genes, u_bins, s_bins)
    for start in range(0, len(genes), batch_size):
        stop = min(len(genes), start + batch_size)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model._encode(
                torch.as_tensor(token_gene[start:stop], device=device),
                torch.as_tensor(token_value[start:stop], device=device),
                torch.as_tensor(token_splice[start:stop], device=device), return_all=True,
            ).float()
        outputs.append(hidden.cpu().numpy())
        if device.type == "cuda":
            snapshot = query_gpu(device.index or 0)
            snapshot.update({
                "timestamp": time.time(), "phase": "runtime", "condition": condition,
                "checkpoint": checkpoint, "batch_size": stop - start, "cells_completed": stop,
                "allocated_memory_mb": torch.cuda.memory_allocated(device) / 2**20,
                "reserved_memory_mb": torch.cuda.memory_reserved(device) / 2**20,
            })
            resource_rows.append(snapshot)
    return np.concatenate(outputs), time.perf_counter() - started


def _plot_smoke(long: pd.DataFrame, pairing: pd.DataFrame, resources: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    summary = long.groupby(["checkpoint", "condition"]).distance.median().reset_index()
    for checkpoint, part in summary.groupby("checkpoint"):
        axes[0].plot(part.condition, part.distance, marker="o", label=checkpoint)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set(title="Context ablation", ylabel="median U/S distance")
    axes[0].legend(frameon=False)
    axes[1].bar(pairing.condition, pairing.mean_distance, color="#3B82F6")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set(title="Pairing controls", ylabel="mean distance")
    runtime = resources.loc[resources.phase.eq("runtime")]
    if not runtime.empty:
        axes[2].plot(runtime.timestamp - runtime.timestamp.min(), runtime.memory_used_mb / 1024, label="total GPU used")
        axes[2].plot(runtime.timestamp - runtime.timestamp.min(), runtime.allocated_memory_mb / 1024, label="scUS allocated")
        axes[2].legend(frameon=False)
    axes[2].set(title="GPU resource trace", xlabel="seconds", ylabel="GB")
    figure.suptitle("scUS frozen-pretraining ablation smoke")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_gpu_smoke(cache_dir: str | Path, checkpoint: str | Path, vocab: str | Path, output_dir: str | Path,
                  device_name: str = "cuda:0", cells: int = 256, genes_per_cell: int = 128,
                  target_genes: int = 8, batch_size: int = 4, seed: int = 42,
                  preflight_samples: int = 12, preflight_interval: float = 10,
                  minimum_free_mb: float = 40_000, maximum_mean_utilization: float = 20,
                  checkpoint_label: str = "epoch11", include_random: bool = True,
                  run_kind: str = "smoke") -> Path:
    from .zero_shot import load_encoder
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpu_index = int(device_name.split(":", 1)[1])
    passed, preflight, reason = resource_preflight(
        gpu_index, preflight_samples, preflight_interval,
        minimum_free_mb=minimum_free_mb,
        maximum_mean_utilization=maximum_mean_utilization,
    )
    preflight.to_csv(out / "resource_trace.csv", index=False)
    if not passed:
        write_json(out / "status.json", {"status": "resource_blocked", "reason": reason})
        (out / "SMOKE_REPORT.md").write_text(
            f"# scUS ablation GPU {run_kind}\n\n**Resource blocked:** {reason}. No model inference was started.\n"
        )
        return out
    device = torch.device(device_name)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 16 * 2**30 / total_memory), device)
    cfg = {"data": {"vocab": str(Path(vocab))}, "model": {"checkpoint": str(Path(checkpoint)), "bin_size": 15}}
    model = load_encoder(cfg, device)
    random_model = _random_like(model, seed).to(device) if include_random else None
    panel, target_local, target_names = _load_cache_panel(Path(cache_dir), Path(vocab), cells, genes_per_cell, target_genes, seed)
    donors = matched_donors(panel.cell_types, seed)
    pd.DataFrame({
        "cell_index": panel.source_cells, "cell_id": panel.cell_ids,
        "cell_type": panel.cell_types, "donor_row": donors, "donor_cell_id": panel.cell_ids[donors],
    }).to_csv(out / "cell_manifest.csv", index=False)
    pd.DataFrame({"target_position": panel.target_positions, "cache_gene_index": target_local, "gene_name": target_names}).to_csv(out / "target_gene_manifest.csv", index=False)

    resource_rows = preflight.to_dict("records")
    long_rows, elapsed = [], {}
    full_reference = None
    conditions = ["full", "target_panel_only", "matched_context", "within_cell_shuffle"]
    for condition in conditions:
        g, u, s = context_condition(panel, condition, donors, seed)
        hidden, seconds = _encode_batches(model, g, u, s, device, batch_size, condition, checkpoint_label, resource_rows)
        elapsed[f"{checkpoint_label}:{condition}"] = seconds
        rep = paired_representations(torch.as_tensor(hidden), u, s)
        if condition == "full":
            full_reference = rep["distance_profile"][:, panel.target_positions].copy()
        for row in range(cells):
            for position in panel.target_positions:
                long_rows.append({
                    "cell_id": panel.cell_ids[row], "cell_type": panel.cell_types[row],
                    "target_gene": target_names[position], "checkpoint": checkpoint_label, "condition": condition,
                    "distance": float(rep["distance_profile"][row, position]),
                })
    pair_genes = panel.gene_ids[:, panel.target_positions].reshape(-1, 1)
    pair_u = panel.u_bins[:, panel.target_positions].reshape(-1, 1)
    pair_s = panel.s_bins[:, panel.target_positions].reshape(-1, 1)
    pair_hidden, seconds = _encode_batches(
        model, pair_genes, pair_u, pair_s, device, max(batch_size, 32),
        "pair_only", checkpoint_label, resource_rows,
    )
    elapsed[f"{checkpoint_label}:pair_only"] = seconds
    pair_rep = paired_representations(torch.as_tensor(pair_hidden), pair_u, pair_s)["distance_profile"].reshape(cells, target_genes)
    for row in range(cells):
        for position in panel.target_positions:
            long_rows.append({
                "cell_id": panel.cell_ids[row], "cell_type": panel.cell_types[row],
                "target_gene": target_names[position], "checkpoint": checkpoint_label, "condition": "pair_only",
                "distance": float(pair_rep[row, position]),
            })
    if random_model is not None:
        full_hidden, seconds = _encode_batches(
            random_model, panel.gene_ids, panel.u_bins, panel.s_bins, device,
            batch_size, "full", "random", resource_rows,
        )
        elapsed["random:full"] = seconds
        random_rep = paired_representations(torch.as_tensor(full_hidden), panel.u_bins, panel.s_bins)
        for row in range(cells):
            for position in panel.target_positions:
                long_rows.append({
                    "cell_id": panel.cell_ids[row], "cell_type": panel.cell_types[row],
                    "target_gene": target_names[position], "checkpoint": "random", "condition": "full",
                    "distance": float(random_rep["distance_profile"][row, position]),
                })

    repeat_hidden, seconds = _encode_batches(
        model, panel.gene_ids[:32], panel.u_bins[:32], panel.s_bins[:32], device,
        batch_size, "full_repeat", checkpoint_label, resource_rows,
    )
    elapsed[f"{checkpoint_label}:full_repeat_32"] = seconds
    repeat_distance = paired_representations(
        torch.as_tensor(repeat_hidden), panel.u_bins[:32], panel.s_bins[:32]
    )["distance_profile"][:, panel.target_positions]
    deterministic_repeat = bool(np.array_equal(full_reference[:32], repeat_distance))

    cache = Path(cache_dir)
    h_u = np.load(cache / "h_u.float16.npy", mmap_mode="r")
    h_s = np.load(cache / "h_s.float16.npy", mmap_mode="r")
    cell_idx = np.load(cache / "cell_idx.int32.npy", mmap_mode="r")
    gene_idx = np.load(cache / "gene_idx.int32.npy", mmap_mode="r")
    counts = np.bincount(np.asarray(cell_idx), minlength=len(np.load(cache / "barcodes.npy", allow_pickle=True)))
    indptr = np.r_[0, np.cumsum(counts)]
    hu_target = np.zeros((cells, target_genes, h_u.shape[1]), np.float32)
    hs_target = np.zeros_like(hu_target)
    for row, cell in enumerate(panel.source_cells):
        sl = slice(indptr[cell], indptr[cell + 1])
        local = np.asarray(gene_idx[sl])
        lookup = {int(g): i for i, g in enumerate(local)}
        positions = np.asarray([lookup[int(g)] for g in target_local]) + indptr[cell]
        hu_target[row], hs_target[row] = h_u[positions], h_s[positions]

    def normalized(value):
        return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)

    true_distance = 1 - np.sum(normalized(hu_target) * normalized(hs_target), axis=-1)
    cell_shuffled = 1 - np.sum(normalized(hu_target) * normalized(hs_target[donors]), axis=-1)
    gene_shuffled_hs = np.roll(hs_target, 1, axis=1)
    gene_shuffled = 1 - np.sum(normalized(hu_target) * normalized(gene_shuffled_hs), axis=-1)
    pairing = pd.DataFrame([
        {"condition": "correct", "mean_distance": float(true_distance.mean()), "median_distance": float(np.median(true_distance))},
        {"condition": "matched_cell", "mean_distance": float(cell_shuffled.mean()), "median_distance": float(np.median(cell_shuffled))},
        {"condition": "within_cell_gene", "mean_distance": float(gene_shuffled.mean()), "median_distance": float(np.median(gene_shuffled))},
    ])
    long = pd.DataFrame(long_rows)
    long.to_parquet(out / "context_ablation_long.parquet", index=False)
    pairing.to_csv(out / "pairing_null_metrics.csv", index=False)
    resources = pd.DataFrame(resource_rows)
    resources.to_csv(out / "resource_trace.csv", index=False)
    _plot_smoke(long, pairing, resources, out / "smoke_summary.png")

    full = long.loc[(long.checkpoint == checkpoint_label) & (long.condition == "full")].sort_values(["cell_id", "target_gene"])
    random_full = long.loc[(long.checkpoint == "random") & (long.condition == "full")].sort_values(["cell_id", "target_gene"])
    runtime_rows = resources.loc[resources.phase.eq("runtime")]
    peak_allocated = float(runtime_rows.allocated_memory_mb.max()) if not runtime_rows.empty else 0.0
    checks = {
        "cell_ids_complete_unique": bool(len(np.unique(panel.cell_ids)) == cells),
        "target_tokens_preserved": True,
        "coverage": float(long.distance.notna().mean()),
        "finite": bool(np.isfinite(long.distance).all()),
        "distance_range": bool(long.distance.between(-1e-5, 2 + 1e-5).all()),
        "donor_nonself": bool(np.all(donors != np.arange(cells))),
        "same_seed_exact_repeat": deterministic_repeat,
        "random_not_identical": (
            bool(not np.allclose(full.distance.to_numpy(), random_full.distance.to_numpy()))
            if include_random else None
        ),
        "peak_allocated_mb": peak_allocated,
        "memory_under_16gb": peak_allocated <= 16 * 1024,
    }
    boolean_checks = [
        value for key, value in checks.items()
        if key not in {"coverage", "peak_allocated_mb"} and value is not None
    ]
    passed_all = all(boolean_checks) and checks["coverage"] >= .999
    total_seconds = float(sum(elapsed.values()))
    seconds_per_cell_condition = total_seconds / (cells * len(elapsed))
    eta_hours_three_10k = seconds_per_cell_condition * 30_000 * len(conditions) / 3600
    status = {
        "status": "passed" if passed_all else "failed", "run_kind": run_kind,
        "checkpoint_label": checkpoint_label, "checkpoint_path": str(Path(checkpoint).resolve()),
        "include_random": include_random, "gpu_id": gpu_index, "seed": seed,
        "cells": cells, "genes_per_cell": genes_per_cell, "target_genes": target_genes,
        "preflight": {
            "samples": preflight_samples, "interval_seconds": preflight_interval,
            "minimum_free_mb": minimum_free_mb,
            "maximum_mean_utilization": maximum_mean_utilization,
        },
        "checks": checks, "elapsed_seconds": elapsed,
        "estimated_three_dataset_hours": eta_hours_three_10k,
    }
    write_json(out / "status.json", status)
    report = f"""# scUS frozen-pretraining ablation GPU {run_kind}

- Status: **{status['status']}**
- Cells / genes / targets: {cells} / {genes_per_cell} / {target_genes}
- Checkpoint: {checkpoint_label}{' plus a same-architecture random initialization' if include_random else ''}
- Total measured inference time: {total_seconds:.1f} seconds
- Estimated three-dataset context runtime: {eta_hours_three_10k:.2f} GPU-hours
- Peak scUS allocated GPU memory: {peak_allocated / 1024:.2f} GB

## Acceptance checks

```json
{json.dumps(checks, indent=2)}
```

This run checks frozen-checkpoint context sensitivity and resource safety. It does not retrain or tune the Transformer.
"""
    (out / "SMOKE_REPORT.md").write_text(report)
    return out


def _representation_probe(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                          representation: str, seed: int, dataset: str = "Forebrain",
                          task: str = "cell_type") -> list[dict]:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

    encoded = LabelEncoder().fit_transform(labels)
    classes = np.unique(encoded)
    records = []
    high_dimensional = features.shape[1] > 512
    for fold, held_out in enumerate(sorted(np.unique(groups))):
        train, test = groups != held_out, groups == held_out
        train_features, test_features = features[train], features[test]
        observed_in_train = np.isfinite(train_features).any(axis=0)
        if not observed_in_train.all():
            train_features = train_features[:, observed_in_train]
            test_features = test_features[:, observed_in_train]
        steps = [SimpleImputer(strategy="median"), StandardScaler()]
        if high_dimensional:
            steps.append(PCA(n_components=min(128, int(train.sum()) - 1, train_features.shape[1]), svd_solver="randomized", random_state=seed))
        transformer = make_pipeline(*steps)
        train_x = transformer.fit_transform(train_features)
        test_x = transformer.transform(test_features)
        classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed,
                solver="liblinear",
            )
        )
        classifier.fit(train_x, encoded[train])
        prediction = classifier.predict(test_x)
        probability = classifier.predict_proba(test_x)
        knn = KNeighborsClassifier(n_neighbors=min(15, int(train.sum())), metric="cosine", weights="distance")
        knn.fit(train_x, encoded[train])
        knn_prediction = knn.predict(test_x)
        metrics = {
            "linear_balanced_accuracy": balanced_accuracy_score(encoded[test], prediction),
            "linear_macro_f1": f1_score(encoded[test], prediction, average="macro"),
            "knn_balanced_accuracy": balanced_accuracy_score(encoded[test], knn_prediction),
            "knn_macro_f1": f1_score(encoded[test], knn_prediction, average="macro"),
        }
        binary = label_binarize(encoded[test], classes=classes)
        if binary.shape[1] == probability.shape[1]:
            metrics["linear_macro_auprc"] = average_precision_score(binary, probability, average="macro")
        for metric, value in metrics.items():
            records.append({
                "dataset": dataset, "task": task, "representation": representation,
                "fold": fold, "held_out_group": held_out, "metric": metric, "value": float(value),
                "split_group": "sample_prefix", "seed": seed, "pretrained": True,
                "target_data_seen": False, "target_labels_seen": False,
                "input_features": int(features.shape[1]), "train_observed_features": int(observed_in_train.sum()),
            })
    return records


def _plot_formal_cache(representation: pd.DataFrame, pairing: pd.DataFrame, output: Path,
                       dataset: str = "Forebrain") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output.mkdir(parents=True, exist_ok=True)
    summary = representation.groupby(["representation", "metric"], as_index=False).value.mean()
    matrix = summary.pivot(index="representation", columns="metric", values="value")
    matrix = matrix.rename(columns={
        "knn_balanced_accuracy": "kNN balanced acc.", "knn_macro_f1": "kNN macro-F1",
        "linear_balanced_accuracy": "Linear balanced acc.", "linear_macro_auprc": "Linear macro-AUPRC",
        "linear_macro_f1": "Linear macro-F1",
    })
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1, ax=axis)
    axis.set_title(f"{dataset} sample-held-out representation ablation")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=20)
    figure.savefig(output / "02_representation_ablation_heatmap.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "02_representation_ablation_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    pair_summary = pairing.groupby("condition", as_index=False).mean_distance.mean().sort_values("mean_distance")
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.bar(pair_summary.condition, pair_summary.mean_distance, color=["#2563EB", "#F59E0B", "#EF4444"])
    axis.set(title=f"{dataset} U/S pairing nulls", ylabel="mean 1-cosine distance")
    axis.tick_params(axis="x", rotation=20)
    figure.savefig(output / "01_checkpoint_and_pairing_controls.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "01_checkpoint_and_pairing_controls.pdf", bbox_inches="tight")
    plt.close(figure)


def run_cache_formal(cache_dir: str | Path, output_dir: str | Path, pairs_per_cell: int = 512,
                     seed: int = 42, metadata_dir: str | Path | None = None,
                     dataset: str = "Forebrain", label_file: str = "cell_types.npy",
                     task: str = "cell_type", sample_side: str = "prefix",
                     drop_labels: tuple[str, ...] = ("nan", "unknown", "")) -> Path:
    """Run full cached epoch-11 representation and pairing ablations on CPU.

    This stage never re-encodes tokens and therefore does not consume a GPU.
    It is intentionally separate from checkpoint/context/top-k stages.
    """
    cache, out = Path(cache_dir), Path(output_dir)
    metadata = Path(metadata_dir) if metadata_dir is not None else cache
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    meta = json.loads((cache / "pair_cache_meta.json").read_text())
    cell_idx = np.load(cache / "cell_idx.int32.npy", mmap_mode="r")
    gene_idx = np.load(cache / "gene_idx.int32.npy", mmap_mode="r")
    h_u = np.load(cache / "h_u.float16.npy", mmap_mode="r")
    h_s = np.load(cache / "h_s.float16.npy", mmap_mode="r")
    has_bins = (cache / "u_bin.uint8.npy").exists() and (cache / "s_bin.uint8.npy").exists()
    u_bins = np.load(cache / "u_bin.uint8.npy", mmap_mode="r") if has_bins else None
    s_bins = np.load(cache / "s_bin.uint8.npy", mmap_mode="r") if has_bins else None
    cell_ids = np.load(metadata / "barcodes.npy", allow_pickle=True).astype(str)
    labels = np.load(metadata / label_file, allow_pickle=True).astype(str)
    gene_names = np.load(metadata / "gene_names.npy", allow_pickle=True).astype(str)
    n_cells, n_genes, embed_dim = len(cell_ids), len(gene_names), int(h_u.shape[1])
    counts = np.bincount(np.asarray(cell_idx), minlength=n_cells)
    indptr = np.r_[0, np.cumsum(counts)]
    if sample_side not in {"prefix", "suffix"}:
        raise ValueError("sample_side must be 'prefix' or 'suffix'")
    samples = np.asarray([
        value.split(":", 1)[0] if sample_side == "prefix" else value.rsplit(":", 1)[-1]
        for value in cell_ids
    ])
    if len(np.unique(samples)) < 2:
        raise ValueError("At least two sample prefixes are required for held-out evaluation")
    excluded = {value.lower() for value in drop_labels}
    evaluation_mask = np.asarray([value.strip().lower() not in excluded for value in labels])
    if not evaluation_mask.any():
        raise ValueError("All cells were removed by the invalid-label filter")

    u_only = np.zeros((n_cells, embed_dim), np.float32)
    s_only = np.zeros_like(u_only)
    state = np.zeros_like(u_only)
    kinetic = np.zeros_like(u_only)
    absolute_kinetic = np.zeros_like(u_only)
    concat = np.zeros((n_cells, 2 * embed_dim), np.float32)
    distance_scalar = np.zeros((n_cells, 1), np.float32)
    distance_profile = np.full((n_cells, n_genes), np.nan, np.float32)
    direct_u_s = np.full((n_cells, n_genes), np.nan, np.float32) if has_bins else None
    for cell in range(n_cells):
        sl = slice(indptr[cell], indptr[cell + 1])
        hu, hs = np.asarray(h_u[sl], np.float32), np.asarray(h_s[sl], np.float32)
        genes = np.asarray(gene_idx[sl])
        u_only[cell], s_only[cell] = hu.mean(0), hs.mean(0)
        state[cell] = ((hu + hs) / 2).mean(0)
        kinetic[cell] = (hu - hs).mean(0)
        absolute_kinetic[cell] = np.abs(hu - hs).mean(0)
        concat[cell] = np.r_[u_only[cell], s_only[cell]]
        distance = 1 - np.sum(
            hu / np.maximum(np.linalg.norm(hu, axis=1, keepdims=True), 1e-12)
            * hs / np.maximum(np.linalg.norm(hs, axis=1, keepdims=True), 1e-12), axis=1,
        )
        distance_profile[cell, genes] = distance
        if has_bins:
            direct_u_s[cell, genes] = np.asarray(u_bins[sl], np.float32) - np.asarray(s_bins[sl], np.float32)
        distance_scalar[cell, 0] = np.median(distance)

    representations = {
        "u_only": u_only, "s_only": s_only, "state": state, "signed_kinetic": kinetic,
        "absolute_kinetic": absolute_kinetic, "concatenated_u_s": concat,
        "distance_scalar": distance_scalar, "distance_profile": distance_profile,
    }
    if has_bins:
        representations["direct_u_s"] = direct_u_s
    metric_rows = []
    for name, features in representations.items():
        metric_rows.extend(_representation_probe(
            features[evaluation_mask], labels[evaluation_mask], samples[evaluation_mask],
            name, seed, dataset, task,
        ))
    representation_metrics = pd.DataFrame(metric_rows)
    representation_metrics.to_csv(out / "representation_metrics_long.csv", index=False)

    rng = np.random.default_rng(seed)
    donors = np.full(n_cells, -1, np.int64)
    valid_cells = np.flatnonzero(evaluation_mask)
    for cell in valid_cells:
        candidates = np.flatnonzero(evaluation_mask & (labels == labels[cell]) & (samples != samples[cell]))
        if not len(candidates):
            candidates = np.flatnonzero(evaluation_mask & (labels == labels[cell]) & (np.arange(n_cells) != cell))
        if not len(candidates):
            raise ValueError(f"No non-self donor for label {labels[cell]!r}")
        donors[cell] = candidates[int(rng.integers(len(candidates)))]
    pairing_rows, sample_rows = [], []
    for cell in valid_cells:
        own = slice(indptr[cell], indptr[cell + 1])
        donor = slice(indptr[donors[cell]], indptr[donors[cell] + 1])
        own_genes = np.asarray(gene_idx[own])
        donor_genes = np.asarray(gene_idx[donor])
        common, own_pos, donor_pos = np.intersect1d(own_genes, donor_genes, return_indices=True)
        if len(common) > pairs_per_cell:
            keep = rng.choice(len(common), pairs_per_cell, replace=False)
            common, own_pos, donor_pos = common[keep], own_pos[keep], donor_pos[keep]
        hu = np.asarray(h_u[own][own_pos], np.float32)
        hs = np.asarray(h_s[own][own_pos], np.float32)
        donor_hs = np.asarray(h_s[donor][donor_pos], np.float32)
        shuffled_hs = np.roll(hs, 1, axis=0)

        def distance(a, b):
            return 1 - np.sum(
                a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                * b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12), axis=1,
            )

        conditions = {
            "correct": distance(hu, hs),
            "matched_cell": distance(hu, donor_hs),
            "within_cell_gene": distance(hu, shuffled_hs),
        }
        for condition, values in conditions.items():
            pairing_rows.append({
                "dataset": dataset, "cell_id": cell_ids[cell], "label": labels[cell],
                "sample": samples[cell], "donor_cell_id": cell_ids[donors[cell]], "condition": condition,
                "n_pairs": len(values), "mean_distance": float(values.mean()),
                "median_distance": float(np.median(values)), "seed": seed,
            })
            take = min(16, len(values))
            for index in range(take):
                sample_rows.append({
                    "cell_id": cell_ids[cell], "gene_name": gene_names[common[index]],
                    "condition": condition, "distance": float(values[index]),
                })
    pairing = pd.DataFrame(pairing_rows)
    pairing.to_csv(out / "pairing_null_metrics_long.csv", index=False)
    pd.DataFrame(sample_rows).to_parquet(out / "pairing_pair_sample.parquet", index=False)

    wide = pairing.pivot(index=["cell_id", "sample"], columns="condition", values="mean_distance").reset_index()
    effect_rows = []
    bootstrap_rng = np.random.default_rng(seed)
    unique_samples = np.unique(wide["sample"])
    for null in ["matched_cell", "within_cell_gene"]:
        wide[f"delta_{null}"] = wide[null] - wide["correct"]
        group_values = wide.groupby("sample")[f"delta_{null}"].mean()
        bootstrap = np.asarray([
            group_values.loc[bootstrap_rng.choice(unique_samples, len(unique_samples), replace=True)].mean()
            for _ in range(1000)
        ])
        effect_rows.append({
            "dataset": dataset, "contrast": f"{null}-correct",
            "mean_delta": float(wide[f"delta_{null}"].mean()),
            "sample_bootstrap_ci_low": float(np.quantile(bootstrap, .025)),
            "sample_bootstrap_ci_high": float(np.quantile(bootstrap, .975)),
            "sample_groups": len(unique_samples), "cells": len(wide), "seed": seed,
        })
    pairing_effects = pd.DataFrame(effect_rows)
    pairing_effects.to_csv(out / "pairing_effect_summary.csv", index=False)
    _plot_formal_cache(representation_metrics, pairing, out, dataset)

    representation_rank = (
        representation_metrics.groupby(["representation", "metric"], as_index=False).value.mean()
        .assign(rank=lambda frame: frame.groupby("metric").value.rank(ascending=False, method="average"))
        .groupby("representation", as_index=False).agg(mean_rank=("rank", "mean"), mean_metric=("value", "mean"))
        .sort_values("mean_rank")
    )
    representation_rank.to_csv(out / "representation_rank_summary.csv", index=False)
    pair_summary = pairing.groupby("condition").mean_distance.mean()
    distance_rank = float(representation_rank.set_index("representation").loc["distance_profile", "mean_rank"])
    direct_rank = float(representation_rank.set_index("representation").loc["direct_u_s", "mean_rank"]) if has_bins else None
    scalar_rank = float(representation_rank.set_index("representation").loc["distance_scalar", "mean_rank"])
    status = {
        "status": "complete", "scope": "cached_epoch11_only", "dataset": dataset,
        "cells_input": n_cells, "cells": int(evaluation_mask.sum()), "genes": n_genes, "pairs": int(meta["n_pairs"]),
        "dropped_invalid_labels": int((~evaluation_mask).sum()),
        "sample_groups": sorted(np.unique(samples).tolist()), "cell_types": sorted(np.unique(labels).tolist()),
        "runtime_seconds": time.perf_counter() - started,
        "correct_distance": float(pair_summary["correct"]),
        "matched_cell_distance": float(pair_summary["matched_cell"]),
        "within_cell_gene_distance": float(pair_summary["within_cell_gene"]),
        "checkpoint": "epoch11_cached", "transformer_reencoded": False,
        "claim_checks": {
            "correct_lt_matched_cell": bool(pair_summary["correct"] < pair_summary["matched_cell"]),
            "correct_lt_gene_shuffle": bool(pair_summary["correct"] < pair_summary["within_cell_gene"]),
            "distance_profile_beats_scalar": bool(distance_rank < scalar_rank),
            "distance_profile_beats_direct_u_s": bool(distance_rank < direct_rank) if direct_rank is not None else None,
        },
    }
    write_json(out / "status.json", status)
    best = representation_rank.iloc[0]
    direct_text = f"{direct_rank:.2f}" if direct_rank is not None else "not available (bins absent)"
    report = f"""# {dataset} frozen scUS cached ablation

- Scope: epoch-11 cached embeddings only; no GPU re-encoding or checkpoint selection.
- Evaluated cells / input cells / genes / cached pairs: {int(evaluation_mask.sum()):,} / {n_cells:,} / {n_genes:,} / {int(meta['n_pairs']):,}
- Held-out groups: {', '.join(sorted(np.unique(samples)))}
- Best cross-task mean rank: `{best.representation}` ({best.mean_rank:.2f})
- Distance profile / direct U-S / scalar mean ranks: {distance_rank:.2f} / {direct_text} / {scalar_rank:.2f}
- Correct / matched-cell / gene-shuffle mean distance: {pair_summary['correct']:.4f} / {pair_summary['matched_cell']:.4f} / {pair_summary['within_cell_gene']:.4f}
- Runtime: {status['runtime_seconds']:.1f} seconds on CPU.

This is a formal cached representation/pairing ablation. Random, epoch 5/8,
context, top-k, Traxler, and Norman remain gated GPU stages and are not inferred
from the epoch-11 cache.{" Direct U-S ranks ahead of the scUS distance profile in this task, so this dataset does not support a claim that learned distance universally outperforms direct U/S information." if direct_rank is not None and direct_rank < distance_rank else ""}
"""
    (out / "PRETRAINING_ABLATION_REPORT.md").write_text(report)
    return out


def plot_checkpoint_context_controls(combined: pd.DataFrame, out: Path) -> None:
    """Categorical medians and paired changes; intervals describe pairs, not uncertainty."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(out)
    conditions = ["full", "matched_context", "within_cell_shuffle", "target_panel_only", "pair_only"]
    labels = ["Full", "Matched context", "Within-cell shuffle", "Target panel only", "Pair only"]
    checkpoints = [name for name in ("epoch00", "epoch03", "epoch06", "epoch09", "epoch11")
                   if name in combined.run_id.unique()]
    fixed_colors = {"epoch00": "#7856A4", "epoch03": "#3976B8", "epoch06": "#D38B32",
                    "epoch09": "#BC5264", "epoch11": "#258777"}
    colors = [fixed_colors.get(name, plt.get_cmap("tab20")(i % 20)) for i, name in enumerate(checkpoints)]
    rows = combined.loc[combined.run_id.isin(checkpoints)].copy()
    reference = rows.loc[rows.condition.eq("full"), ["run_id", "checkpoint", "cell_id", "target_gene", "distance"]]
    paired = rows.merge(reference, on=["run_id", "checkpoint", "cell_id", "target_gene"],
                        suffixes=("", "_full"), validate="many_to_one")
    paired["delta"] = paired.distance - paired.distance_full
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6.5, 5 + .6 * len(checkpoints))))
    fig.subplots_adjust(left=.14, right=.98, top=.79, bottom=.23, wspace=.48)
    statistics = []
    for index, (checkpoint, color) in enumerate(zip(checkpoints, colors)):
        spacing = .76 / max(1, len(checkpoints))
        offset = (index - (len(checkpoints) - 1) / 2) * spacing
        for position, condition in enumerate(conditions):
            part = paired.loc[paired.checkpoint.eq(checkpoint) & paired.condition.eq(condition)]
            if part.empty:
                continue
            median = float(part.distance.median())
            axes[0].scatter(median, position + offset, s=45, color=color, zorder=3)
            q10, q25, q50, q75, q90 = np.quantile(part.delta, [.1, .25, .5, .75, .9])
            statistics.append(dict(checkpoint=checkpoint, condition=condition, pairs=len(part),
                                   distance_median=median, delta_q10=q10, delta_q25=q25,
                                   delta_median=q50, delta_q75=q75, delta_q90=q90))
            if condition != "full":
                axes[1].bxp([dict(med=q50, q1=q25, q3=q75, whislo=q10, whishi=q90, fliers=[])],
                            positions=[position - 1 + offset], widths=spacing * .78, vert=False,
                            patch_artist=True, showfliers=False, manage_ticks=False,
                            boxprops=dict(facecolor=color, edgecolor=color, alpha=.45),
                            medianprops=dict(color=color, linewidth=2),
                            whiskerprops=dict(color=color), capprops=dict(color=color))
    random_full = rows.loc[rows.checkpoint.eq("random") & rows.condition.eq("full")]
    if not random_full.empty:
        value = float(random_full.distance.median())
        axes[0].scatter(value, 0, marker="X", s=75, color="#303440", zorder=4)
        statistics.append(dict(checkpoint="random", condition="full", pairs=len(random_full), distance_median=value))
    axes[0].set(yticks=np.arange(5), yticklabels=labels, ylim=(4.5, -.5),
                xlabel="Median U–S cosine distance", title="A   Distance by experimental condition")
    axes[1].set(yticks=np.arange(4), yticklabels=labels[1:], ylim=(3.5, -.5),
                xlabel="Paired Δdistance = condition − Full", title="B   Changes for the same cell–gene pairs")
    axes[1].axvline(0, color="#64748B", linestyle="--", linewidth=1, zorder=0)
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.grid(axis="x", color="#E2E8F0", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.set_title(ax.get_title(), fontsize=13, pad=16)
    handles = [Line2D([], [], marker="o", linestyle="none", color=c, label=f"Epoch {int(e[-2:])}")
               for e, c in zip(checkpoints, colors)]
    if not random_full.empty:
        handles.append(Line2D([], [], marker="X", linestyle="none", color="#303440", label="Random init · Full only"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .10), ncol=min(6, len(handles)), frameon=False, fontsize=11)
    fig.suptitle("Frozen scUS: checkpoint and context ablations", fontsize=19, fontweight="semibold", y=.96)
    fig.text(.5, .88, "Forebrain · 1,000 cells × 24 target genes · top-k 256 · sampling seed 42",
             ha="center", color="#64748B", fontsize=11)
    fig.text(.5, .045, "A: pooled medians.  B: boxes = 25–75%; whiskers = 10–90% of paired changes (not confidence intervals).",
             ha="center", color="#475569", fontsize=10)
    fig.text(.5, .012, "Matched-context donors share cell type only; input removal also changes context size.",
             ha="center", color="#64748B", fontsize=10)
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"01_checkpoint_and_pairing_controls.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(statistics).to_csv(out / "01_checkpoint_and_pairing_controls_plot_data.csv", index=False)


def summarize_gpu_formal(input_dir: str | Path, output_dir: str | Path) -> Path:
    """Combine completed frozen-checkpoint context/top-k runs without selecting a checkpoint."""
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    root, out = Path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, run_rows = [], []
    for status_path in sorted(root.glob("*/status.json")):
        status = json.loads(status_path.read_text())
        parquet = status_path.parent / "context_ablation_long.parquet"
        if status.get("status") != "passed" or not parquet.exists():
            continue
        name = status_path.parent.name
        match = re.search(r"topk(\d+)_seed(\d+)", name)
        top_k = int(status.get("genes_per_cell") or (match.group(1) if match else 256))
        seed = int(status.get("seed") or (match.group(2) if match else 42))
        checkpoint = status.get("checkpoint_label") or (name if name.startswith("epoch") else "epoch11")
        frame = pd.read_parquet(parquet)
        frame["run_id"], frame["top_k"], frame["seed"] = name, top_k, seed
        frames.append(frame)
        elapsed = status.get("elapsed_seconds", {})
        primary_seconds = sum(value for key, value in elapsed.items() if key.startswith(f"{checkpoint}:"))
        run_rows.append({
            "run_id": name, "checkpoint": checkpoint, "top_k": top_k, "seed": seed,
            "cells": status.get("cells", frame.cell_id.nunique()),
            "targets": status.get("target_genes", frame.target_gene.nunique()),
            "runtime_seconds": primary_seconds,
            "peak_allocated_mb": status.get("checks", {}).get("peak_allocated_mb"),
            "coverage": status.get("checks", {}).get("coverage"),
        })
    if not frames:
        raise ValueError(f"No passed GPU formal runs found below {root}")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out / "context_ablation_long.parquet", index=False)
    runs = pd.DataFrame(run_rows).sort_values(["checkpoint", "top_k", "seed"])
    runs.to_csv(out / "resource_metrics.csv", index=False)

    primary = combined.loc[combined.checkpoint.ne("random")].copy()
    reference = primary.loc[primary.condition.eq("full"), ["run_id", "cell_id", "target_gene", "distance"]]
    effects = primary.merge(reference, on=["run_id", "cell_id", "target_gene"], suffixes=("", "_full"))
    effects["delta_from_full"] = effects.distance - effects.distance_full
    context_summary = effects.groupby(
        ["run_id", "checkpoint", "top_k", "seed", "condition"], as_index=False
    ).agg(
        median_distance=("distance", "median"), mean_distance=("distance", "mean"),
        median_delta_from_full=("delta_from_full", "median"),
        mean_absolute_delta_from_full=("delta_from_full", lambda value: float(np.mean(np.abs(value)))),
    )
    context_summary.to_csv(out / "context_effect_summary.csv", index=False)

    baseline_runs = {name: name for name in sorted(combined.run_id.unique())
                     if name.startswith("epoch") and name[5:].isdigit()}
    checkpoint_rows = []
    base = combined.loc[
        combined.run_id.eq("epoch11") & combined.checkpoint.eq("epoch11") & combined.condition.eq("full"),
        ["cell_id", "target_gene", "distance"],
    ].rename(columns={"distance": "epoch11_distance"})
    for checkpoint, run_id in baseline_runs.items():
        current = combined.loc[
            combined.run_id.eq(run_id) & combined.checkpoint.eq(checkpoint) & combined.condition.eq("full"),
            ["cell_id", "target_gene", "distance"],
        ]
        merged = current.merge(base, on=["cell_id", "target_gene"])
        checkpoint_rows.append({
            "checkpoint": checkpoint, "run_id": run_id, "pairs": len(merged),
            "median_distance": float(merged.distance.median()),
            "spearman_vs_epoch11": float(merged.distance.corr(merged.epoch11_distance, method="spearman")),
            "mean_absolute_difference_vs_epoch11": float(np.mean(np.abs(merged.distance - merged.epoch11_distance))),
        })
    random_current = combined.loc[
        combined.run_id.eq("epoch11") & combined.checkpoint.eq("random") & combined.condition.eq("full"),
        ["cell_id", "target_gene", "distance"],
    ]
    if not random_current.empty:
        merged = random_current.merge(base, on=["cell_id", "target_gene"])
        checkpoint_rows.append({
            "checkpoint": "random", "run_id": "epoch11", "pairs": len(merged),
            "median_distance": float(merged.distance.median()),
            "spearman_vs_epoch11": float(merged.distance.corr(merged.epoch11_distance, method="spearman")),
            "mean_absolute_difference_vs_epoch11": float(np.mean(np.abs(merged.distance - merged.epoch11_distance))),
        })
    checkpoint_sensitivity = pd.DataFrame(checkpoint_rows)
    checkpoint_sensitivity.to_csv(out / "checkpoint_sensitivity.csv", index=False)

    topk_rows = []
    topk_base = base
    for run in runs.loc[(runs.checkpoint == "epoch11") & (runs.seed == 42)].itertuples():
        current = combined.loc[
            combined.run_id.eq(run.run_id) & combined.checkpoint.eq("epoch11") & combined.condition.eq("full"),
            ["cell_id", "target_gene", "distance"],
        ]
        merged = current.merge(topk_base, on=["cell_id", "target_gene"])
        topk_rows.append({
            "run_id": run.run_id, "top_k": run.top_k, "pairs": len(merged),
            "spearman_vs_topk256": float(merged.distance.corr(merged.epoch11_distance, method="spearman")),
            "mean_absolute_difference_vs_topk256": float(np.mean(np.abs(merged.distance - merged.epoch11_distance))),
            "runtime_seconds": run.runtime_seconds, "peak_allocated_mb": run.peak_allocated_mb,
        })
    topk_sensitivity = pd.DataFrame(topk_rows).sort_values("top_k")
    topk_sensitivity.to_csv(out / "topk_sensitivity.csv", index=False)

    seed_summary = context_summary.loc[
        context_summary.run_id.isin(["epoch11", "topk256_seed43", "topk256_seed44"])
        & context_summary.checkpoint.eq("epoch11")
        & context_summary.top_k.eq(256)
    ].copy()
    seed_summary.to_csv(out / "seed_sensitivity.csv", index=False)

    canonical = context_summary.loc[
        context_summary.run_id.isin(baseline_runs)
    ]
    plot_checkpoint_context_controls(combined, out)

    matrix = canonical.pivot(index="checkpoint", columns="condition", values="mean_absolute_delta_from_full")
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="mako", ax=ax)
    ax.set(title="Context dependence relative to Full", xlabel="condition", ylabel="checkpoint")
    fig.savefig(out / "03_context_dependence_effects.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "03_context_dependence_effects.pdf", bbox_inches="tight")
    plt.close(fig)

    if not topk_sensitivity.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
        axes[0].plot(topk_sensitivity.top_k, topk_sensitivity.spearman_vs_topk256, marker="o")
        axes[0].set(xlabel="top-k genes", ylabel="Spearman vs top-k 256", ylim=(0, 1.02))
        axes[1].plot(topk_sensitivity.top_k, topk_sensitivity.runtime_seconds, marker="o", color="#D97706")
        axes[1].set(xlabel="top-k genes", ylabel="inference seconds")
        fig.suptitle("Input-scale robustness and cost")
        fig.savefig(out / "05_topk_robustness_and_cost.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "05_topk_robustness_and_cost.pdf", bbox_inches="tight")
        plt.close(fig)

    checkpoint_lookup = checkpoint_sensitivity.set_index("checkpoint")
    topk_lookup = topk_sensitivity.set_index("top_k") if not topk_sensitivity.empty else pd.DataFrame()
    matched_epoch11 = canonical.loc[
        canonical.checkpoint.eq("epoch11") & canonical.condition.eq("matched_context"),
        "median_delta_from_full",
    ]
    random_text = (
        f"{checkpoint_lookup.loc['random', 'median_distance']:.3f} / "
        f"{checkpoint_lookup.loc['epoch11', 'median_distance']:.3f}"
        if "random" in checkpoint_lookup.index else "not available"
    )
    topk_text = (
        f"{topk_lookup.loc[512, 'spearman_vs_topk256']:.3f} / "
        f"{topk_lookup.loc[1000, 'spearman_vs_topk256']:.3f}"
        if {512, 1000}.issubset(set(topk_lookup.index)) else "pending"
    )
    report = f"""# Frozen scUS GPU ablation summary

- Passed runs combined: {len(runs)}
- Context observations: {len(combined):,}
- Checkpoint comparison: the main figure uses epoch 0, 3, 6, 9, and 11 on identical seed-42 inputs; all completed checkpoints remain available in the tables, and epoch 11 remains the fixed checkpoint.
- Top-k comparison: {', '.join(map(str, topk_sensitivity.top_k.tolist())) if not topk_sensitivity.empty else 'pending'}.
- Epoch 5 / epoch 8 distance Spearman versus epoch 11: {checkpoint_lookup.loc['epoch05', 'spearman_vs_epoch11']:.3f} / {checkpoint_lookup.loc['epoch08', 'spearman_vs_epoch11']:.3f}.
- Random-init / epoch-11 Full median distance: {random_text}.
- Epoch-11 Matched-context median delta from Full: {float(matched_epoch11.iloc[0]):.6f}.
- Top-k 512 / 1,000 Spearman versus top-k 256: {topk_text}.

The tables report all completed runs; no checkpoint is re-selected from these results.
Context sensitivity is measured relative to the same target U/S tokens under `Full`.
Near-zero `Matched-context − Full` effects are retained as negative evidence rather
than interpreted as proof of context dependence. The marked top-k sensitivity means
that gene-selection scale must be fixed in benchmark comparisons; it is not currently
valid to claim scale-invariant distance profiles. A random model's larger distance
magnitude alone is not a biological performance comparison.

Important protocol boundary: this Forebrain pass uses a non-self donor matched by
cell type. It does not yet enforce the stricter sample, library-size, detected-gene,
and target-bin matching registered in PLAN.md. Therefore it is a formal-size
checkpoint/input-scale result but only a preliminary context-null result; it cannot
by itself pass or fail the final context-dependence acceptance criterion.
"""
    (out / "PRETRAINING_ABLATION_REPORT.md").write_text(report)
    return out

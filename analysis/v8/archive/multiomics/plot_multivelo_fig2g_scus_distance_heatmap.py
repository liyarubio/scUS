#!/usr/bin/env python3
"""Fig2g-style heatmaps for scUS embedding distances.

Cells follow the same latent-time order used by scVelo heatmap. Genes are
split by MultiVelo fit_model and sorted by the peak order of smoothed Ms,
matching Fig2g's gene ordering rule. Heatmap values are scUS pairwise
embedding cosine distances.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.colors import ListedColormap


ROOT = Path("/data1/liyaru/proj_us/scUS_6.0_moments_uors_alldata/cell_emb3/multivelo_mouse_brain_trimodal_scus_100ep")
ADATA_PATH = ROOT / "multivelo_fig2g_latenttime" / "multivelo_result_with_latent_time.h5ad"
META_PATH = ROOT / "trimodal_distance_meta.npz"
OUT_DIR = ROOT / "multivelo_fig2g_latenttime" / "fig2g_scus_distance_heatmaps"

PAIR_NAMES = ("us", "ua", "sa")
PAIR_LABEL = {"us": "U-S", "ua": "U-ATAC", "sa": "S-ATAC"}
HEATMAP_CMAP = "viridis"
N_CONVOLVE = 30

CELLTYPE_PALETTE = {
    "Upper Layer": "#1f77b4",
    "Deeper Layer": "#ff7f0e",
    "V-SVZ": "#2ca02c",
    "RG, Astro, OPC": "#d62728",
    "Ependymal cells": "#9467bd",
    "IPC": "#8c564b",
    "Subplate": "#e377c2",
}


def smooth_rows(X: np.ndarray, n: int = N_CONVOLVE) -> np.ndarray:
    if n is None or n <= 1:
        return X
    weights = np.ones(n, dtype=np.float32) / float(n)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = np.convolve(X[i], weights, mode="same")
    return out


def row_minmax(X: np.ndarray) -> np.ndarray:
    out = X.astype(np.float32, copy=True)
    mn = np.nanmin(out, axis=1, keepdims=True)
    mx = np.nanmax(out, axis=1, keepdims=True)
    denom = mx - mn
    denom[~np.isfinite(denom) | (denom <= 1e-12)] = 1.0
    out = (out - mn) / denom
    out[~np.isfinite(out)] = 0.0
    return out


def row_robust_minmax(X: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> np.ndarray:
    out = X.astype(np.float32, copy=True)
    lo = np.nanpercentile(out, q_low, axis=1, keepdims=True)
    hi = np.nanpercentile(out, q_high, axis=1, keepdims=True)
    denom = hi - lo
    denom[~np.isfinite(denom) | (denom <= 1e-12)] = 1.0
    out = np.clip((out - lo) / denom, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def row_zscore_clip(X: np.ndarray, clip: float = 2.5) -> np.ndarray:
    out = X.astype(np.float32, copy=True)
    mu = np.nanmean(out, axis=1, keepdims=True)
    sd = np.nanstd(out, axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    out = np.clip((out - mu) / sd, -clip, clip)
    out[~np.isfinite(out)] = 0.0
    return out


def global_robust_minmax(X: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    out = X.astype(np.float32, copy=True)
    lo, hi = np.nanpercentile(out, [q_low, q_high])
    denom = hi - lo
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = 1.0
    out = np.clip((out - lo) / denom, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def median_impute_gene_matrix(X_cells_by_genes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = X_cells_by_genes.astype(np.float32, copy=True)
    med = np.nanmedian(out, axis=0).astype(np.float32)
    med[~np.isfinite(med)] = 0.0
    rows, cols = np.where(~np.isfinite(out))
    out[rows, cols] = med[cols]
    return out, med


def fig2g_gene_order(adata: sc.AnnData, genes: list[str], cell_order: np.ndarray) -> list[str]:
    X = adata[cell_order, genes].layers["Ms"]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32).T
    X = smooth_rows(X)
    order = np.argsort(np.argmax(X, axis=1))
    return [genes[i] for i in order]


def celltype_color_strip(cell_types: np.ndarray) -> tuple[np.ndarray, ListedColormap]:
    order = [ct for ct in CELLTYPE_PALETTE if ct in set(cell_types)]
    order += [ct for ct in sorted(set(cell_types)) if ct not in CELLTYPE_PALETTE]
    palette = [CELLTYPE_PALETTE.get(ct, "#777777") for ct in order]
    index = {ct: i for i, ct in enumerate(order)}
    values = np.array([index[ct] for ct in cell_types], dtype=np.int32)[None, :]
    return values, ListedColormap(palette)


def plot_heatmap(
    data: np.ndarray,
    cell_types: np.ndarray,
    title: str,
    out_prefix: Path,
    cmap: str,
    gene_labels: list[str] | None = None,
) -> None:
    strip, strip_cmap = celltype_color_strip(cell_types)
    fig = plt.figure(figsize=(9.2, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.12, 5.0], hspace=0.02)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_strip.imshow(strip, aspect="auto", cmap=strip_cmap, interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Cells ordered by latent time")
    ax.set_ylabel("Genes ordered by Fig2g Ms peak")
    ax.set_xticks([])
    if gene_labels and len(gene_labels) <= 80:
        ax.set_yticks(np.arange(len(gene_labels)))
        ax.set_yticklabels(gene_labels, fontsize=5)
    else:
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, label="row-scaled scUS distance")

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=ct, markersize=6)
        for ct, color in CELLTYPE_PALETTE.items()
        if ct in set(cell_types)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False, fontsize=7)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_grid(
    heatmaps: dict[tuple[int, str], np.ndarray],
    out_prefix: Path,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    colorbar_label: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    for row, model in enumerate((1, 2)):
        for col, pair in enumerate(PAIR_NAMES):
            ax = axes[row, col]
            im = ax.imshow(
                heatmaps[(model, pair)],
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"Model {model}: {PAIR_LABEL[pair]}")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("Genes")
            if row == 1:
                ax.set_xlabel("Cells by latent time")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=colorbar_label)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(ADATA_PATH)
    meta = np.load(META_PATH, allow_pickle=True)
    gene_vocab_ids = np.asarray(meta["gene_vocab_ids_present"], dtype=np.int64)
    mouse_genes = np.asarray(meta["gene_names_present"], dtype=str)
    vocab_by_mouse = {gene: int(vocab) for gene, vocab in zip(mouse_genes, gene_vocab_ids)}
    col_by_vocab = {int(v): i for i, v in enumerate(gene_vocab_ids)}

    barcodes = np.asarray(meta["barcodes"], dtype=str)
    if not np.array_equal(barcodes, adata.obs_names.astype(str).to_numpy()):
        barcode_to_row = {bc: i for i, bc in enumerate(barcodes)}
        dist_row_order = np.array([barcode_to_row[bc] for bc in adata.obs_names.astype(str)], dtype=np.int64)
    else:
        dist_row_order = np.arange(adata.n_obs, dtype=np.int64)

    latent = np.asarray(adata.obs["latent_time"], dtype=float)
    cell_order = np.argsort(latent)
    ordered_cell_types = adata.obs["celltype"].astype(str).to_numpy()[cell_order]

    model_gene_orders: dict[int, list[str]] = {}
    for model in (1, 2):
        model_genes = adata.var_names[np.asarray(adata.var["fit_model"].values == model)].astype(str).tolist()
        matched = [g for g in model_genes if g in vocab_by_mouse]
        model_gene_orders[model] = fig2g_gene_order(adata, matched, cell_order)
        print(f"[genes] model={model} total={len(model_genes)} matched_scus={len(matched)}", flush=True)

    pair_distance: dict[str, np.ndarray] = {}
    pair_observed: dict[str, np.ndarray] = {}
    for pair in PAIR_NAMES:
        mat = sp.load_npz(ROOT / f"{pair}_cosine_dist_sparse.npz")[:, gene_vocab_ids].toarray().astype(np.float32)
        mat = mat[dist_row_order]
        obs = mat != 0
        mat[~obs] = np.nan
        pair_distance[pair] = mat
        pair_observed[pair] = obs

    summary_rows = []
    raw_heatmaps: dict[tuple[int, str], np.ndarray] = {}
    heatmaps: dict[tuple[int, str], np.ndarray] = {}
    for model in (1, 2):
        genes = model_gene_orders[model]
        cols = np.array([col_by_vocab[vocab_by_mouse[g]] for g in genes], dtype=np.int64)
        for pair in PAIR_NAMES:
            X = pair_distance[pair][:, cols]
            obs = pair_observed[pair][:, cols]
            X_imp, _ = median_impute_gene_matrix(X)
            H = X_imp[cell_order].T
            H = smooth_rows(H)
            raw_heatmaps[(model, pair)] = H
            heatmaps[(model, pair)] = row_minmax(H)
            summary_rows.append(
                {
                    "fit_model": model,
                    "pair": pair,
                    "pair_label": PAIR_LABEL[pair],
                    "n_genes_fig2g_model": int((adata.var["fit_model"].values == model).sum()),
                    "n_genes_matched_scus": len(genes),
                    "n_cells": adata.n_obs,
                    "observed_fraction": float(obs.mean()),
                    "mean_distance_observed": float(np.nanmean(X)),
                    "median_distance_observed": float(np.nanmedian(X)),
                }
            )
            plot_heatmap(
                H,
                ordered_cell_types,
                f"Fig2g order, scUS distance: Model {model} genes, {PAIR_LABEL[pair]}",
                OUT_DIR / f"fig2g_model{model}_{pair}_scus_distance_heatmap",
                HEATMAP_CMAP,
            )

    plot_grid(
        heatmaps,
        OUT_DIR / "fig2g_scus_distance_heatmap_2x3",
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=1,
        title="Fig2g order, scUS distance: row min-max, viridis",
        colorbar_label="row-scaled scUS distance",
    )

    variant_dir = OUT_DIR / "normalization_variants"
    variant_dir.mkdir(exist_ok=True)
    variant_specs = [
        ("row_minmax_viridis", heatmaps, "viridis", 0.0, 1.0, "Row min-max, viridis", "row min-max"),
        (
            "row_robust_magma",
            {k: row_robust_minmax(v) for k, v in raw_heatmaps.items()},
            "magma",
            0.0,
            1.0,
            "Row robust 2-98% scale, magma",
            "row robust scale",
        ),
        (
            "row_robust_turbo",
            {k: row_robust_minmax(v) for k, v in raw_heatmaps.items()},
            "turbo",
            0.0,
            1.0,
            "Row robust 2-98% scale, turbo",
            "row robust scale",
        ),
        (
            "row_zscore_RdBu_r",
            {k: row_zscore_clip(v) for k, v in raw_heatmaps.items()},
            "RdBu_r",
            -2.5,
            2.5,
            "Row z-score clipped at +/-2.5, RdBu_r",
            "row z-score",
        ),
        (
            "global_robust_magma",
            {k: global_robust_minmax(v) for k, v in raw_heatmaps.items()},
            "magma",
            0.0,
            1.0,
            "Global robust 1-99% scale, magma",
            "global robust scale",
        ),
    ]
    variant_rows = []
    for name, mats, cmap, vmin, vmax, title, cbar in variant_specs:
        plot_grid(
            mats,
            variant_dir / f"fig2g_scus_distance_heatmap_2x3_{name}",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            colorbar_label=cbar,
        )
        variant_rows.append(
            {
                "variant": name,
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
                "description": title,
            }
        )
    pd.DataFrame(variant_rows).to_csv(variant_dir / "normalization_variant_manifest.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "fig2g_scus_distance_heatmap_summary.csv", index=False)
    pd.DataFrame(
        [
            {"fit_model": model, "rank": i, "gene": gene, "vocab_id": vocab_by_mouse[gene]}
            for model, genes in model_gene_orders.items()
            for i, gene in enumerate(genes)
        ]
    ).to_csv(OUT_DIR / "fig2g_scus_distance_gene_order.csv", index=False)
    np.savez_compressed(
        OUT_DIR / "fig2g_scus_distance_heatmap_matrices.npz",
        model1_us=heatmaps[(1, "us")],
        model1_ua=heatmaps[(1, "ua")],
        model1_sa=heatmaps[(1, "sa")],
        model2_us=heatmaps[(2, "us")],
        model2_ua=heatmaps[(2, "ua")],
        model2_sa=heatmaps[(2, "sa")],
        raw_model1_us=raw_heatmaps[(1, "us")],
        raw_model1_ua=raw_heatmaps[(1, "ua")],
        raw_model1_sa=raw_heatmaps[(1, "sa")],
        raw_model2_us=raw_heatmaps[(2, "us")],
        raw_model2_ua=raw_heatmaps[(2, "ua")],
        raw_model2_sa=raw_heatmaps[(2, "sa")],
        cell_order=cell_order,
        latent_time=latent[cell_order],
        ordered_cell_types=ordered_cell_types,
        model1_genes=np.asarray(model_gene_orders[1], dtype=object),
        model2_genes=np.asarray(model_gene_orders[2], dtype=object),
    )
    print(f"[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

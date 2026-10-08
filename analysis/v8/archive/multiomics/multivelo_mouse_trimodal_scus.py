#!/usr/bin/env python3
"""Fine-tune scUS with U/S/ATAC tokens on MultiVelo mouse multiome data.

This follows the all-gene distance style of
``redeem_us_distance_genome_wide_allgenes.py`` and extends the two RNA states to
three token modalities:

    0 = unspliced, 1 = spliced, 2 = ATAC gene activity

The requested base checkpoint uses the scBaseCount 36,602-gene vocabulary, while
the MultiVelo test data uses mouse-style gene symbols. Genes are therefore mapped
as:

    original mouse gene symbol -> uppercase symbol -> scBaseCount vocab id
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = Path("/data1/liyaru/proj_us/scUS_6.0_moments_uors_alldata")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.flash_mha_fallback as flash_mha_fallback
from models.flash_mha_fallback import SpatialTransformer
from utils.binning import rank_bin

RNA_PATH = Path(
    os.environ.get(
        "SCUS_TRIMODAL_RNA_PATH",
        "/data1/liyaru/software/14_multiVelo/MultiVelo-main/test_files/adata_postpro.h5ad",
    )
)
ATAC_PATH = Path(
    os.environ.get(
        "SCUS_TRIMODAL_ATAC_PATH",
        "/data1/liyaru/software/14_multiVelo/MultiVelo-main/test_files/adata_atac_postpro.h5ad",
    )
)
RNA_U_LAYER = os.environ.get("SCUS_TRIMODAL_U_LAYER", "Mu")
RNA_S_LAYER = os.environ.get("SCUS_TRIMODAL_S_LAYER", "Ms")
ATAC_LAYER = os.environ.get("SCUS_TRIMODAL_ATAC_LAYER", "Mc")
VOCAB_PATH = Path("/data1/liyaru/proj_us/data/scBaseCount/gene_vocab.json")
BASE_CKPT = ROOT / "checkpoints/scUS/scUS-epoch=11-val_loss_epoch=2.0079.ckpt"

OUT_DIR = Path(
    os.environ.get(
        "SCUS_MULTIVELO_TRIMODAL_OUT_DIR",
        str(ROOT / "cell_emb3" / "multivelo_mouse_trimodal_scus_100ep"),
    )
)
PT_PATH = OUT_DIR / "multivelo_mouse_trimodal.pt"
CKPT_DIR = OUT_DIR / "checkpoints"

PAIR_NAMES = ("us", "ua", "sa")
PAIR_LABEL = {"us": "U-S", "ua": "U-ATAC", "sa": "S-ATAC"}
PAIR_COLOR = {"us": "#1f77b4", "ua": "#ff7f0e", "sa": "#2ca02c"}
PAIR_CMAP = {"us": "Blues", "ua": "Oranges", "sa": "Greens"}
MODALITY_NAMES = {0: "u", 1: "s", 2: "atac"}


class TriModalGeneValueEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_value_bins: int,
        embed_dim: int,
        dropout: float = 0.1,
        num_modalities: int = 3,
    ) -> None:
        super().__init__()
        self.gene_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_embedding = nn.Embedding(n_value_bins + 1, embed_dim, padding_idx=0)
        self.modality_embedding = nn.Embedding(num_modalities, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gene_ids, value_bins, modality_flags):  # type: ignore[override]
        return self.dropout(
            self.gene_embedding(gene_ids)
            + self.value_embedding(value_bins)
            + self.modality_embedding(modality_flags)
        )


class TriModalMaskedModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        bin_size: int,
        *,
        embed_dim: int = 128,
        lr: float = 1e-4,
        num_heads: int = 8,
        num_layers: int = 6,
        mask_ratio: float = 0.3,
        num_modalities: int = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.bin_size = bin_size
        self.mask_ratio = mask_ratio
        self.mask_token = bin_size + 1
        self.embed = TriModalGeneValueEmbedding(
            vocab_size=vocab_size,
            n_value_bins=bin_size + 1,
            embed_dim=embed_dim,
            num_modalities=num_modalities,
        )
        self.encoder = SpatialTransformer(embed_dim, num_heads, num_layers)
        self.head_gene = nn.Linear(embed_dim, bin_size + 1)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)

    def _rand_mask(self, value_bins: torch.Tensor) -> torch.Tensor:
        candidates = value_bins > 0
        mask = torch.rand_like(value_bins, dtype=torch.float).lt(self.mask_ratio) & candidates
        need_fix = (mask.sum(1) == 0) & (candidates.sum(1) > 0)
        if need_fix.any():
            col = torch.multinomial(candidates[need_fix].float(), 1).squeeze(1)
            mask[need_fix, col] = True
        return mask

    def _encode(
        self,
        gene_ids: torch.Tensor,
        value_bins: torch.Tensor,
        modality_flags: torch.Tensor,
        *,
        return_all: bool = False,
    ) -> torch.Tensor:
        tok = self.embed(gene_ids, value_bins, modality_flags)
        if tok.device.type != "cuda":
            flash_mha_fallback._FLASH_ATTN_AVAILABLE = False
        pad_mask = value_bins == 0
        out = self.encoder(tok, mask=pad_mask)
        return out if return_all else out

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self._encode(batch["gene_ids"], batch["value_bins"], batch["modality_flags"])
        return self.head_gene(h)

    def _shared_step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        gene_ids = batch["gene_ids"]
        orig_bins = batch["value_bins"]
        modality_flags = batch["modality_flags"]
        vb = orig_bins.clone()
        mask = self._rand_mask(orig_bins)
        vb[mask] = self.mask_token

        out = self._encode(gene_ids, vb, modality_flags)
        logits = self.head_gene(out)
        loss_all = self.loss_ce(logits[mask], orig_bins[mask].long()) if mask.any() else 0.0 * logits.sum()
        logs: dict[str, Any] = {f"{prefix}_loss": loss_all}
        for flag, name in MODALITY_NAMES.items():
            m = mask & (modality_flags == flag)
            loss_m = self.loss_ce(logits[m], orig_bins[m].long()) if m.any() else 0.0 * logits.sum()
            logs[f"{prefix}_loss_{name}"] = loss_m
            logs[f"{prefix}_n_{name}"] = m.sum()
        self.log_dict(
            logs,
            prog_bar=True,
            on_step=prefix == "train",
            on_epoch=True,
            batch_size=gene_ids.size(0),
            sync_dist=False,
        )
        return loss_all

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss", "interval": "epoch", "frequency": 1},
        }


class RaggedTriModalDataset(Dataset):
    def __init__(self, shard: dict[str, Any]) -> None:
        self.shard = shard

    def __len__(self) -> int:
        return len(self.shard["gene_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "gene_ids": self.shard["gene_ids"][idx].long(),
            "value_bins": self.shard["value_bins"][idx].long(),
            "modality_flags": self.shard["modality_flags"][idx].long(),
        }


def collate_records(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(r["gene_ids"].numel()) for r in records)
    bs = len(records)
    out = {
        "gene_ids": torch.zeros((bs, max_len), dtype=torch.long),
        "value_bins": torch.zeros((bs, max_len), dtype=torch.long),
        "modality_flags": torch.zeros((bs, max_len), dtype=torch.long),
    }
    for i, rec in enumerate(records):
        n = rec["gene_ids"].numel()
        for key in out:
            out[key][i, :n] = rec[key]
    return out


def sparse_row_to_dense(mat: Any, i: int) -> np.ndarray:
    row = mat[i]
    if hasattr(row, "toarray"):
        return np.asarray(row.toarray()).ravel()
    return np.asarray(row).ravel()


def load_vocab() -> tuple[dict[str, int], list[str]]:
    with open(VOCAB_PATH) as f:
        gene2idx = json.load(f)
    idx2gene = [""] * len(gene2idx)
    for gene, idx in gene2idx.items():
        idx2gene[int(idx)] = gene
    return gene2idx, idx2gene


def build_gene_mapping(rna: sc.AnnData, gene2idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    src_idx: list[int] = []
    vocab_idx: list[int] = []
    original_genes: list[str] = []
    mapped_genes: list[str] = []
    seen_vocab: set[int] = set()
    exact = 0
    uppercase = 0
    duplicate_targets = 0

    for j, gene_raw in enumerate(rna.var_names.astype(str)):
        if gene_raw in gene2idx:
            mapped = gene_raw
            exact += 1
        elif gene_raw.upper() in gene2idx:
            mapped = gene_raw.upper()
            uppercase += 1
        else:
            continue
        vid = int(gene2idx[mapped])
        if vid in seen_vocab:
            duplicate_targets += 1
            continue
        seen_vocab.add(vid)
        src_idx.append(j)
        vocab_idx.append(vid)
        original_genes.append(gene_raw)
        mapped_genes.append(mapped)

    print(
        f"[prepare] gene mapping: matched={len(src_idx)} / {rna.n_vars}; "
        f"exact={exact}, uppercase={uppercase}, duplicate_targets_skipped={duplicate_targets}",
        flush=True,
    )
    if len(src_idx) < 100:
        raise RuntimeError("Too few MultiVelo genes mapped to scBaseCount vocab.")
    return (
        np.asarray(src_idx, dtype=np.int64),
        np.asarray(vocab_idx, dtype=np.int64),
        original_genes,
        mapped_genes,
    )


def build_trimodal_pt(bin_size: int, force: bool = False, binning: str = "hybrid") -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PT_PATH.exists() and not force:
        return torch.load(PT_PATH, map_location="cpu", weights_only=False)

    gene2idx, _ = load_vocab()
    rna = sc.read_h5ad(RNA_PATH)
    atac = sc.read_h5ad(ATAC_PATH)
    rna.var_names_make_unique()
    atac.var_names_make_unique()

    print(f"[prepare] RNA={RNA_PATH} shape={rna.shape}", flush=True)
    print(f"[prepare] ATAC={ATAC_PATH} shape={atac.shape}", flush=True)
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC obs_names are not aligned.")
    if not rna.var_names.equals(atac.var_names):
        raise ValueError("RNA and ATAC var_names are not aligned.")
    for layer in (RNA_U_LAYER, RNA_S_LAYER):
        if layer not in rna.layers:
            raise KeyError(f"Missing RNA layer: {layer}")
    if ATAC_LAYER not in atac.layers:
        raise KeyError(f"Missing ATAC layer: {ATAC_LAYER}")

    src_idx, vocab_idx, original_genes, mapped_genes = build_gene_mapping(rna, gene2idx)
    Xu = rna.layers[RNA_U_LAYER]
    Xs = rna.layers[RNA_S_LAYER]
    Xa = atac.layers[ATAC_LAYER]

    gene_ids_list: list[torch.Tensor] = []
    value_bins_list: list[torch.Tensor] = []
    modality_flags_list: list[torch.Tensor] = []
    tok_counts: list[int] = []

    for i in tqdm(range(rna.n_obs), desc="tokenize"):
        u = sparse_row_to_dense(Xu, i)[src_idx].astype(np.float32, copy=False)
        s = sparse_row_to_dense(Xs, i)[src_idx].astype(np.float32, copy=False)
        a = sparse_row_to_dense(Xa, i)[src_idx].astype(np.float32, copy=False)
        keep = (u > 0) | (s > 0) | (a > 0)
        if not keep.any():
            gene_ids = torch.zeros(1, dtype=torch.int32)
            value_bins = torch.zeros(1, dtype=torch.uint8)
            modality_flags = torch.zeros(1, dtype=torch.int8)
        else:
            gids = vocab_idx[keep]
            u_sel, s_sel, a_sel = u[keep], s[keep], a[keep]
            gene_ids_np = np.repeat(gids, 3).astype(np.int64)
            values = np.empty(3 * len(gids), dtype=np.float32)
            values[0::3] = u_sel
            values[1::3] = s_sel
            values[2::3] = a_sel
            if binning == "joint":
                vb = rank_bin(values, bins=bin_size).astype(np.int64)
            elif binning == "per_modality":
                vb = np.zeros_like(values, dtype=np.int64)
                vb[0::3] = rank_bin(u_sel, bins=bin_size)
                vb[1::3] = rank_bin(s_sel, bins=bin_size)
                vb[2::3] = rank_bin(a_sel, bins=bin_size)
            elif binning == "hybrid":
                vb = np.zeros_like(values, dtype=np.int64)
                us = np.empty(2 * len(gids), dtype=np.float32)
                us[0::2] = u_sel
                us[1::2] = s_sel
                us_bins = rank_bin(us, bins=bin_size)
                vb[0::3] = us_bins[0::2]
                vb[1::3] = us_bins[1::2]
                vb[2::3] = rank_bin(a_sel, bins=bin_size)
            else:
                raise ValueError(f"Unknown binning mode: {binning}")
            flags = np.tile(np.array([0, 1, 2], dtype=np.int8), len(gids))
            gene_ids = torch.from_numpy(gene_ids_np).to(torch.int32)
            value_bins = torch.from_numpy(vb).to(torch.uint8)
            modality_flags = torch.from_numpy(flags).to(torch.int8)
        gene_ids_list.append(gene_ids)
        value_bins_list.append(value_bins)
        modality_flags_list.append(modality_flags)
        tok_counts.append(int(gene_ids.numel()))

    obs_dict: dict[str, list[str]] = {}
    for col in ["celltype", "CellType", "STD.CellType", "Sample"]:
        if col in rna.obs:
            obs_dict[f"obs_{col}"] = rna.obs[col].astype(str).tolist()

    shard = {
        "gene_ids": gene_ids_list,
        "value_bins": value_bins_list,
        "modality_flags": modality_flags_list,
        "obs_barcode": rna.obs_names.astype(str).tolist(),
        "gene_names_present": original_genes,
        "gene_names_mapped": mapped_genes,
        "gene_vocab_ids_present": vocab_idx,
        "mapping_rule": "exact if possible else uppercase",
        "binning": binning,
        **obs_dict,
    }
    torch.save(shard, PT_PATH, _use_new_zipfile_serialization=True)
    pd.DataFrame(
        {
            "source_index": src_idx,
            "mouse_gene": original_genes,
            "mapped_gene": mapped_genes,
            "vocab_id": vocab_idx,
        }
    ).to_csv(OUT_DIR / "gene_mapping_uppercase_to_scbasecount.csv", index=False)
    print(
        f"[prepare] saved {PT_PATH}; cells={rna.n_obs}, mapped_genes={len(vocab_idx)}, "
        f"tokens/cell min={min(tok_counts)}, median={int(np.median(tok_counts))}, max={max(tok_counts)}",
        flush=True,
    )
    return shard


def init_model_from_scus(vocab_size: int, bin_size: int, base_ckpt: Path, lr: float, mask_ratio: float) -> TriModalMaskedModel:
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    ckpt_vocab_size = int(hp.get("vocab_size", ckpt["state_dict"]["embed.gene_embedding.weight"].shape[0]))
    if ckpt_vocab_size != vocab_size:
        raise ValueError(f"Base checkpoint vocab_size={ckpt_vocab_size}, but requested vocab_size={vocab_size}")

    model = TriModalMaskedModel(
        vocab_size=vocab_size,
        bin_size=bin_size,
        embed_dim=int(hp.get("embed_dim", 128)),
        lr=lr,
        num_heads=int(hp.get("num_heads", 8)),
        num_layers=int(hp.get("num_layers", 6)),
        mask_ratio=mask_ratio,
    )
    new_state = model.state_dict()
    old_state = ckpt["state_dict"]
    copied = 0
    skipped: list[str] = []
    for key, val in old_state.items():
        if key == "embed.splice_embedding.weight":
            w = new_state["embed.modality_embedding.weight"].clone()
            w[:2] = val
            w[2] = val.mean(dim=0)
            new_state["embed.modality_embedding.weight"] = w
            copied += 1
        elif key in new_state and tuple(new_state[key].shape) == tuple(val.shape):
            new_state[key] = val
            copied += 1
        else:
            skipped.append(key)
    model.load_state_dict(new_state, strict=True)
    print(
        f"[model] initialized from {base_ckpt}; copied={copied}, skipped={len(skipped)}, "
        f"gene_embedding={tuple(new_state['embed.gene_embedding.weight'].shape)}",
        flush=True,
    )
    return model


def train(args: argparse.Namespace) -> Path:
    shard = build_trimodal_pt(args.bin_size, force=args.force_prepare, binning=args.binning)
    gene2idx, _ = load_vocab()
    model = init_model_from_scus(
        vocab_size=len(gene2idx),
        bin_size=args.bin_size,
        base_ckpt=Path(args.base_ckpt),
        lr=args.lr,
        mask_ratio=args.mask_ratio,
    )
    ds = RaggedTriModalDataset(shard)
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_records,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_records,
        pin_memory=True,
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="trimodal-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        logger=[
            CSVLogger(save_dir=str(OUT_DIR / "logs"), name="csv"),
            TensorBoardLogger(save_dir=str(OUT_DIR / "logs"), name="tb"),
        ],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = checkpoint_cb.best_model_path or str(CKPT_DIR / "last.ckpt")
    print(f"[train] best checkpoint: {best}", flush=True)
    return Path(best)


def latest_checkpoint() -> Path:
    last = CKPT_DIR / "last.ckpt"
    if last.exists():
        return last
    ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {CKPT_DIR}")
    return ckpts[0]


def normalize_pt_entry(entry: Any) -> np.ndarray:
    if torch.is_tensor(entry):
        return entry.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(entry, dtype=np.int64)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    dist = np.zeros(a.shape[0], dtype=np.float32)
    if valid.any():
        aa = a[valid] / na[valid, None]
        bb = b[valid] / nb[valid, None]
        dist[valid] = 1.0 - np.sum(aa * bb, axis=1)
    return dist, valid


def compute_pairwise_distances(args: argparse.Namespace, ckpt_path: Path | None = None) -> None:
    shard = build_trimodal_pt(args.bin_size, force=False, binning=args.binning)
    gene2idx, idx2gene = load_vocab()
    ckpt = ckpt_path or (Path(args.trimodal_ckpt) if args.trimodal_ckpt else latest_checkpoint())
    model = TriModalMaskedModel.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embed_dim = int(model.hparams.embed_dim)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_amp = device.type == "cuda"

    rows = {p: [] for p in PAIR_NAMES}
    cols = {p: [] for p in PAIR_NAMES}
    vals = {p: [] for p in PAIR_NAMES}
    barcodes = [str(x) for x in shard["obs_barcode"]]
    cell_types = [str(x) for x in shard.get("obs_celltype", ["Unknown"] * len(barcodes))]

    records: list[dict[str, torch.Tensor]] = []
    cell_indices: list[int] = []
    for i in range(len(barcodes)):
        gids = torch.from_numpy(normalize_pt_entry(shard["gene_ids"][i]))
        vbins = torch.from_numpy(normalize_pt_entry(shard["value_bins"][i]))
        flags = torch.from_numpy(normalize_pt_entry(shard["modality_flags"][i]))
        valid = vbins > 0
        if valid.sum() == 0:
            continue
        records.append({"gene_ids": gids[valid], "value_bins": vbins[valid], "modality_flags": flags[valid]})
        cell_indices.append(i)

    for start in tqdm(range(0, len(records), args.infer_batch_size), desc="distance"):
        chunk = records[start : start + args.infer_batch_size]
        batch = collate_records(chunk)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model._encode(batch["gene_ids"], batch["value_bins"], batch["modality_flags"], return_all=True)
            else:
                out = model._encode(batch["gene_ids"], batch["value_bins"], batch["modality_flags"], return_all=True)
        out_np = out.detach().cpu().float().numpy()
        for bi, rec in enumerate(chunk):
            cell_row = cell_indices[start + bi]
            L = int(rec["gene_ids"].numel())
            cell_out = out_np[bi, :L]
            gids = rec["gene_ids"].numpy()
            flags = rec["modality_flags"].numpy()
            unique_gids, inverse = np.unique(gids, return_inverse=True)
            emb = np.zeros((3, unique_gids.size, embed_dim), dtype=np.float32)
            has = np.zeros((3, unique_gids.size), dtype=bool)
            for pos, gpos in enumerate(inverse):
                flag = int(flags[pos])
                if 0 <= flag <= 2:
                    emb[flag, gpos] = cell_out[pos]
                    has[flag, gpos] = True
            for pair, a, b in (("us", 0, 1), ("ua", 0, 2), ("sa", 1, 2)):
                both = has[a] & has[b]
                if not both.any():
                    continue
                d, valid = cosine_distance(emb[a, both], emb[b, both])
                gids_pair = unique_gids[both][valid]
                d = d[valid]
                rows[pair].extend([cell_row] * len(gids_pair))
                cols[pair].extend(gids_pair.tolist())
                vals[pair].extend(d.tolist())
        del out, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta = {
        "barcodes": np.array(barcodes, dtype=object),
        "cell_types": np.array(cell_types, dtype=object),
        "gene_names": np.array(idx2gene, dtype=object),
        "gene_vocab_ids_present": np.asarray(shard["gene_vocab_ids_present"], dtype=np.int64),
        "gene_names_present": np.asarray(shard["gene_names_present"], dtype=object),
        "gene_names_mapped": np.asarray(shard["gene_names_mapped"], dtype=object),
    }
    for pair in PAIR_NAMES:
        mat = sp.csr_matrix(
            (
                np.asarray(vals[pair], dtype=np.float32),
                (np.asarray(rows[pair], dtype=np.int64), np.asarray(cols[pair], dtype=np.int64)),
            ),
            shape=(len(barcodes), len(gene2idx)),
        )
        sp.save_npz(OUT_DIR / f"{pair}_cosine_dist_sparse.npz", mat)
        meta[f"{pair}_gene_counts"] = np.diff(mat.tocsc().indptr)
        print(f"[distance] {pair}: shape={mat.shape}, nnz={mat.nnz}", flush=True)
    np.savez(OUT_DIR / "trimodal_distance_meta.npz", **meta)
    summarize_distances()


def dense_imputed_for_genes(pair: str, gene_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = sp.load_npz(OUT_DIR / f"{pair}_cosine_dist_sparse.npz")[:, gene_ids].toarray().astype(np.float32)
    observed = mat != 0
    tmp = mat.copy()
    tmp[~observed] = np.nan
    med = np.nanmedian(tmp, axis=0)
    med[~np.isfinite(med)] = 0.5
    rows, cols = np.where(~observed)
    mat[rows, cols] = med[cols]
    return mat, observed.sum(axis=0)


def run_umap_matrix(X: np.ndarray, seed: int, n_pcs: int, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(n_pcs, X_scaled.shape[0] - 1, X_scaled.shape[1] - 1)
    pcs = PCA(n_components=n_comp, random_state=seed).fit_transform(X_scaled).astype(np.float32)
    ad = sc.AnnData(X=sp.csr_matrix((X.shape[0], 1)))
    ad.obsm["X_pca"] = pcs
    sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=min(n_neighbors, X.shape[0] - 1))
    sc.tl.umap(ad, random_state=seed)
    return ad.obsm["X_umap"].astype(np.float32), pcs


def plot_3panel(coords: np.ndarray, values_by_pair: dict[str, np.ndarray], out_dir: Path, prefix: str, title_prefix: str, size: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    for ax, pair in zip(axes, PAIR_NAMES):
        values = values_by_pair[pair]
        vmin, vmax = np.nanpercentile(values, [1, 99])
        sca = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=size,
            cmap=PAIR_CMAP[pair],
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(f"{title_prefix}: {PAIR_LABEL[pair]}")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04, label="mean cosine distance")
    fig.savefig(out_dir / f"{prefix}_3panel.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{prefix}_3panel.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_celltype(coords: np.ndarray, cell_types: np.ndarray, out_dir: Path, name: str, title: str, size: int = 5) -> None:
    order = sorted(set(cell_types.astype(str)))
    cmap = plt.get_cmap("tab20")
    palette = {ct: matplotlib.colors.rgb2hex(cmap(i % cmap.N)) for i, ct in enumerate(order)}
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for ct in order:
        mask = cell_types == ct
        ax.scatter(coords[mask, 0], coords[mask, 1], s=size, c=palette[ct], label=f"{ct} ({mask.sum()})", linewidths=0, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2, fontsize=8)
    fig.savefig(out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_allgene_umaps(args: argparse.Namespace) -> None:
    meta = np.load(OUT_DIR / "trimodal_distance_meta.npz", allow_pickle=True)
    gene_ids = np.asarray(meta["gene_vocab_ids_present"], dtype=np.int64)
    barcodes = meta["barcodes"].astype(str)
    cell_types = meta["cell_types"].astype(str)
    mapped_genes = meta["gene_names_mapped"].astype(str)
    source_genes = meta["gene_names_present"].astype(str)

    dist: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for pair in PAIR_NAMES:
        dist[pair], counts[pair] = dense_imputed_for_genes(pair, gene_ids)
        print(pair, dist[pair].shape, "observed genes", int((counts[pair] > 0).sum()), flush=True)

    shared_dir = OUT_DIR / "shared_umap_allgenes"
    shared_dir.mkdir(exist_ok=True)
    X_cell = np.hstack([dist[pair] for pair in PAIR_NAMES]).astype(np.float32)
    cell_coords, cell_pcs = run_umap_matrix(X_cell, args.seed, args.n_pcs, args.n_neighbors)
    cell_values = {pair: dist[pair].mean(axis=1) for pair in PAIR_NAMES}
    plot_3panel(cell_coords, cell_values, shared_dir, "cell_allgenes_umap_distances", "All-gene shared cell UMAP", size=5)
    plot_celltype(cell_coords, cell_types, shared_dir, "cell_allgenes_umap_celltype", "All-gene shared cell UMAP: cell type")
    pd.DataFrame(
        {
            "barcode": barcodes,
            "cell_type": cell_types,
            "umap_x": cell_coords[:, 0],
            "umap_y": cell_coords[:, 1],
            "mean_us": cell_values["us"],
            "mean_ua": cell_values["ua"],
            "mean_sa": cell_values["sa"],
        }
    ).to_csv(shared_dir / "cell_allgenes_umap_metadata.csv", index=False)

    X_gene = np.hstack([dist[pair].T for pair in PAIR_NAMES]).astype(np.float32)
    gene_coords, _ = run_umap_matrix(X_gene, args.seed, args.n_pcs, args.n_neighbors)
    gene_values = {pair: dist[pair].mean(axis=0) for pair in PAIR_NAMES}
    plot_3panel(gene_coords, gene_values, shared_dir, "gene_allgenes_umap_distances", "All-gene shared gene UMAP", size=20)
    pd.DataFrame(
        {
            "mouse_gene": source_genes,
            "mapped_gene": mapped_genes,
            "vocab_id": gene_ids,
            "umap_x": gene_coords[:, 0],
            "umap_y": gene_coords[:, 1],
            "mean_us": gene_values["us"],
            "mean_ua": gene_values["ua"],
            "mean_sa": gene_values["sa"],
            "count_us": counts["us"],
            "count_ua": counts["ua"],
            "count_sa": counts["sa"],
        }
    ).to_csv(shared_dir / "gene_allgenes_umap_metadata.csv", index=False)
    np.savez_compressed(
        shared_dir / "allgenes_shared_umap_coords.npz",
        cell_umap=cell_coords,
        cell_pca=cell_pcs,
        gene_umap=gene_coords,
        barcodes=barcodes,
        cell_types=cell_types,
        mapped_genes=mapped_genes,
        mouse_genes=source_genes,
        gene_vocab_ids=gene_ids,
    )

    pair_dir = OUT_DIR / "per_pair_cell_umap_allgenes"
    pair_dir.mkdir(exist_ok=True)
    X_pair = np.vstack([dist[pair] for pair in PAIR_NAMES]).astype(np.float32)
    pair_coords, pair_pcs = run_umap_matrix(X_pair, args.seed, args.n_pcs, args.n_neighbors)
    n_cells = len(barcodes)
    pair_ids = np.concatenate([np.full(n_cells, pair, dtype=object) for pair in PAIR_NAMES])
    pair_cell_types = np.tile(cell_types, len(PAIR_NAMES))
    pair_barcodes = np.tile(barcodes, len(PAIR_NAMES))
    mean_distance = np.concatenate([dist[pair].mean(axis=1) for pair in PAIR_NAMES]).astype(np.float32)

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    for pair in PAIR_NAMES:
        mask = pair_ids == pair
        ax.scatter(
            pair_coords[mask, 0],
            pair_coords[mask, 1],
            s=4,
            c=PAIR_COLOR[pair],
            label=f"{PAIR_LABEL[pair]} ({mask.sum()})",
            linewidths=0,
            alpha=0.65,
        )
    ax.set_title("All-gene cell-pair UMAP: pair")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=3)
    fig.savefig(pair_dir / "cell_pair_allgenes_umap_by_pair.png", dpi=180, bbox_inches="tight")
    fig.savefig(pair_dir / "cell_pair_allgenes_umap_by_pair.pdf", bbox_inches="tight")
    plt.close(fig)

    pair_values = {pair: mean_distance[pair_ids == pair] for pair in PAIR_NAMES}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    for ax, pair in zip(axes, PAIR_NAMES):
        mask = pair_ids == pair
        values = mean_distance[mask]
        vmin, vmax = np.nanpercentile(values, [1, 99])
        sca = ax.scatter(pair_coords[mask, 0], pair_coords[mask, 1], c=values, s=5, cmap=PAIR_CMAP[pair], vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(f"All-gene cell-pair UMAP: {PAIR_LABEL[pair]}")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04, label="mean cosine distance")
    fig.savefig(pair_dir / "cell_pair_allgenes_umap_3panel.png", dpi=180, bbox_inches="tight")
    fig.savefig(pair_dir / "cell_pair_allgenes_umap_3panel.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_celltype(pair_coords, pair_cell_types, pair_dir, "cell_pair_allgenes_umap_by_celltype", "All-gene cell-pair UMAP: cell type", size=4)
    labels = np.array([PAIR_LABEL[p] for p in pair_ids])
    sample_n = min(10000, pair_coords.shape[0])
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(pair_coords.shape[0], size=sample_n, replace=False)
    sil_umap = silhouette_score(pair_coords[sample_idx], labels[sample_idx])
    sil_pca = silhouette_score(pair_pcs[sample_idx], labels[sample_idx])
    centroids = {pair: pair_coords[pair_ids == pair].mean(axis=0) for pair in PAIR_NAMES}
    summary = []
    for pair in PAIR_NAMES:
        row = {
            "pair": pair,
            "pair_label": PAIR_LABEL[pair],
            "n_points": int((pair_ids == pair).sum()),
            "mean_distance": float(pair_values[pair].mean()),
            "std_distance": float(pair_values[pair].std()),
            "umap_centroid_x": float(centroids[pair][0]),
            "umap_centroid_y": float(centroids[pair][1]),
            "silhouette_umap_by_pair": float(sil_umap),
            "silhouette_pca_by_pair": float(sil_pca),
            "silhouette_sample_n": int(sample_n),
        }
        for q in PAIR_NAMES:
            row[f"centroid_distance_to_{q}"] = float(np.linalg.norm(centroids[pair] - centroids[q]))
        summary.append(row)
    pd.DataFrame(summary).to_csv(pair_dir / "cell_pair_allgenes_umap_summary.csv", index=False)
    pd.DataFrame(
        {
            "barcode": pair_barcodes,
            "cell_type": pair_cell_types,
            "pair": pair_ids,
            "pair_label": labels,
            "mean_distance": mean_distance,
            "umap_x": pair_coords[:, 0],
            "umap_y": pair_coords[:, 1],
        }
    ).to_csv(pair_dir / "cell_pair_allgenes_umap_metadata.csv", index=False)
    np.savez_compressed(
        pair_dir / "cell_pair_allgenes_umap_coords.npz",
        X_umap=pair_coords,
        X_pca=pair_pcs,
        barcodes=pair_barcodes,
        cell_types=pair_cell_types,
        pairs=pair_ids,
        mean_distance=mean_distance,
        mapped_genes=mapped_genes,
        mouse_genes=source_genes,
        gene_vocab_ids=gene_ids,
    )
    print(f"[umap] saved {shared_dir} and {pair_dir}", flush=True)


def summarize_distances() -> None:
    meta = np.load(OUT_DIR / "trimodal_distance_meta.npz", allow_pickle=True)
    cell_types = meta["cell_types"].astype(str)
    gene_names = meta["gene_names"].astype(str)
    records = []
    gene_records = []
    for pair in PAIR_NAMES:
        mat = sp.load_npz(OUT_DIR / f"{pair}_cosine_dist_sparse.npz").tocsr()
        nonzero_per_cell = np.diff(mat.indptr)
        sums = np.asarray(mat.sum(axis=1)).ravel()
        means = np.divide(sums, nonzero_per_cell, out=np.full_like(sums, np.nan, dtype=np.float64), where=nonzero_per_cell > 0)
        for ct in sorted(set(cell_types)):
            mask = cell_types == ct
            records.append(
                {
                    "pair": pair,
                    "cell_type": ct,
                    "n_cells": int(mask.sum()),
                    "mean_cell_distance": float(np.nanmean(means[mask])),
                    "median_cell_distance": float(np.nanmedian(means[mask])),
                }
            )
        csc = mat.tocsc()
        counts = np.diff(csc.indptr)
        gene_sums = np.asarray(csc.sum(axis=0)).ravel()
        gene_means = np.divide(gene_sums, counts, out=np.full_like(gene_sums, np.nan, dtype=np.float64), where=counts > 0)
        top_idx = np.argsort(np.nan_to_num(gene_means, nan=-np.inf))[::-1][:100]
        for idx in top_idx:
            if np.isfinite(gene_means[idx]):
                gene_records.append({"pair": pair, "gene": gene_names[idx], "n_cells": int(counts[idx]), "mean_distance": float(gene_means[idx])})
    pd.DataFrame(records).to_csv(OUT_DIR / "pairwise_distance_by_celltype.csv", index=False)
    pd.DataFrame(gene_records).to_csv(OUT_DIR / "top_pairwise_distance_genes.csv", index=False)
    print("[summary] saved pairwise summary tables", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prepare", "train", "distance", "umap", "all"], default="all")
    p.add_argument("--base_ckpt", default=str(BASE_CKPT))
    p.add_argument("--trimodal_ckpt", default="")
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--binning", choices=["hybrid", "joint", "per_modality"], default="hybrid")
    p.add_argument("--bin_size", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--infer_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    ckpt_path: Path | None = None
    if args.stage in ("prepare", "all"):
        build_trimodal_pt(args.bin_size, force=args.force_prepare, binning=args.binning)
    if args.stage in ("train", "all"):
        ckpt_path = train(args)
    if args.stage in ("distance", "all"):
        compute_pairwise_distances(args, ckpt_path=ckpt_path)
    if args.stage in ("umap", "all"):
        run_allgene_umaps(args)
    print(f"Done. Outputs: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

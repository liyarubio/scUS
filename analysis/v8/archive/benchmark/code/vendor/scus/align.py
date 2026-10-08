from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from .runtime import cosine_distance, provenance, stage_directory, write_json
from .zero_shot import encode


class SharedResidualProjector(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 256, alpha_init: float = 0.05):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))
        nn.init.normal_(self.mlp[2].weight, std=1e-3)
        nn.init.zeros_(self.mlp[2].bias)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, value):
        return F.normalize(self.norm(value + self.alpha * self.mlp(value)), dim=-1)


class ModalityDiscriminator(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, value):
        return self.network(value)


def _confusion(discriminator, value):
    log_probability = F.log_softmax(discriminator(value), dim=1)
    return -(0.5 * log_probability[:, 0] + 0.5 * log_probability[:, 1]).mean()


def _effective_rank(value: np.ndarray) -> float:
    value = np.asarray(value, np.float64)
    if len(value) < 2:
        return 0.0
    singular = np.linalg.svd(value - value.mean(0, keepdims=True), compute_uv=False)
    energy = singular ** 2
    if not np.isfinite(energy).all() or energy.sum() <= 0:
        return 0.0
    probability = energy / energy.sum()
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _pairwise_cosine(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    normalized = value / np.maximum(norms, 1e-12)
    rows, columns = np.triu_indices(len(value), 1)
    return 1 - np.sum(normalized[rows] * normalized[columns], axis=1)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _knn_overlap(raw: np.ndarray, projected: np.ndarray, neighbours: int = 15) -> float:
    n = min(len(raw), len(projected))
    if n < 3:
        return float("nan")
    k = min(neighbours, n - 1)
    def graph(value):
        value = value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
        similarity = value @ value.T
        np.fill_diagonal(similarity, -np.inf)
        return np.argpartition(similarity, -k, axis=1)[:, -k:]
    before, after = graph(np.asarray(raw[:n], np.float64)), graph(np.asarray(projected[:n], np.float64))
    return float(np.mean([len(np.intersect1d(before[row], after[row], assume_unique=False)) / k for row in range(n)]))


def _alignment_qc(u: torch.Tensor, s: torch.Tensor, zu: torch.Tensor, zs: torch.Tensor,
                  discriminator: nn.Module) -> dict[str, float | bool]:
    raw_u, raw_s = u.detach().cpu().numpy(), s.detach().cpu().numpy()
    projected_u, projected_s = zu.detach().cpu().numpy(), zs.detach().cpu().numpy()
    raw_distance = cosine_distance(u, s).detach().cpu().numpy()
    projected_distance = cosine_distance(zu, zs).detach().cpu().numpy()
    shuffled_distance = cosine_distance(zu, torch.roll(zs, 1, 0)).detach().cpu().numpy()
    logits = discriminator(torch.cat([zu, zs]))
    labels = torch.cat([torch.zeros(len(zu), dtype=torch.long, device=zu.device), torch.ones(len(zs), dtype=torch.long, device=zs.device)])
    discriminator_loss = float(F.cross_entropy(logits, labels))
    discriminator_accuracy = float((logits.argmax(1) == labels).float().mean())

    geometry_n = min(len(raw_u), 512)
    raw_u_geometry, projected_u_geometry = _pairwise_cosine(raw_u[:geometry_n]), _pairwise_cosine(projected_u[:geometry_n])
    raw_s_geometry, projected_s_geometry = _pairwise_cosine(raw_s[:geometry_n]), _pairwise_cosine(projected_s[:geometry_n])
    rank_n = min(len(raw_u), 8192)
    raw_rank = _effective_rank(np.concatenate([raw_u[:rank_n], raw_s[:rank_n]]))
    projected_rank = _effective_rank(np.concatenate([projected_u[:rank_n], projected_s[:rank_n]]))
    raw_iqr = float(np.quantile(raw_distance, .75) - np.quantile(raw_distance, .25))
    projected_iqr = float(np.quantile(projected_distance, .75) - np.quantile(projected_distance, .25))
    finite = bool(np.isfinite(projected_u).all() and np.isfinite(projected_s).all() and np.isfinite(projected_distance).all())
    norms = np.concatenate([np.linalg.norm(projected_u, axis=1), np.linalg.norm(projected_s, axis=1)])
    coverage = float(np.mean(np.isfinite(norms) & (norms > 0)))
    knn_n = min(len(raw_u), 512)
    overlap = float(np.nanmean([
        _knn_overlap(raw_u[:knn_n], projected_u[:knn_n]),
        _knn_overlap(raw_s[:knn_n], projected_s[:knn_n]),
    ]))
    geometry_u = _safe_spearman(raw_u_geometry, projected_u_geometry)
    geometry_s = _safe_spearman(raw_s_geometry, projected_s_geometry)
    raw_projected = _safe_spearman(raw_distance, projected_distance)
    distance_ratio = projected_iqr / max(raw_iqr, 1e-12)
    rank_ratio = projected_rank / max(raw_rank, 1e-12)
    eligible = bool(
        finite and coverage >= .999 and rank_ratio >= .9 and distance_ratio >= .5
        and overlap >= .9 and raw_projected >= .9
    )
    selection_score = float(np.nanmean([
        geometry_u, geometry_s, raw_projected, overlap,
        min(1.0, distance_ratio), min(1.0, rank_ratio),
        1 - min(1.0, abs(discriminator_accuracy - .5) * 2),
    ]))
    return {
        "loss_disc": discriminator_loss,
        "disc_accuracy": discriminator_accuracy,
        "effective_rank": projected_rank,
        "raw_effective_rank": raw_rank,
        "distance_median": float(np.median(projected_distance)),
        "raw_distance_median": float(np.median(raw_distance)),
        "distance_iqr": projected_iqr,
        "raw_distance_iqr": raw_iqr,
        "distance_iqr_ratio": distance_ratio,
        "knn_overlap": overlap,
        "geometry_u_spearman": geometry_u,
        "geometry_s_spearman": geometry_s,
        "raw_projected_distance_spearman": raw_projected,
        "matched_shuffle_effect": float(np.mean(shuffled_distance) - np.mean(projected_distance)),
        "coverage": coverage,
        "finite_qc": finite,
        "checkpoint_eligible": eligible,
        "checkpoint_selection_score": selection_score,
    }


def fit(cfg: dict, tag: str | None = None, device_name: str | None = None,
        epochs: int | None = None, steps_per_epoch: int | None = None) -> Path:
    if cfg.get('align', {}).get('mode') == 'joint_reconstruction':
        from .joint_training import fit as fit_joint
        return fit_joint(cfg, tag, device_name, epochs, steps_per_epoch)
    source = stage_directory(cfg, "zero-shot", tag)
    hu = np.load(source / "reservoir_hu.npy", mmap_mode="r")
    hs = np.load(source / "reservoir_hs.npy", mmap_mode="r")
    pair_cells = np.load(source / "reservoir_cell.npy", mmap_mode="r")
    cells = pd.read_parquet(stage_directory(cfg, "prepare", tag) / "cells.parquet").set_index("cell_index", drop=False)
    align_cfg = cfg.get("align", {})
    split_group = align_cfg.get("split_group", "embryo_id")
    if split_group not in cells:
        raise ValueError(f"Align split group '{split_group}' is absent from obs")
    groups = cells.loc[np.asarray(pair_cells), split_group].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError(f"Align requires at least two '{split_group}' groups; found {len(unique)}")
    rng = np.random.default_rng(cfg.get("seed", 42))
    rng.shuffle(unique)
    validation_groups = set(unique[:max(1, int(np.ceil(0.2 * len(unique))))])
    validation = np.isin(groups, list(validation_groups))
    train_pool, validation_pool = np.flatnonzero(~validation), np.flatnonzero(validation)
    device = torch.device(device_name or cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    projector = SharedResidualProjector(int(cfg.get("model", {}).get("embed_dim", 128))).to(device)
    discriminator = ModalityDiscriminator(int(cfg.get("model", {}).get("embed_dim", 128))).to(device)
    learning_rate = float(align_cfg.get("learning_rate", 5e-4))
    projector_optimizer = torch.optim.AdamW(projector.parameters(), lr=learning_rate, weight_decay=1e-4)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate, weight_decay=1e-4)
    epochs = int(epochs or align_cfg.get("epochs", 30))
    steps_per_epoch = int(steps_per_epoch or align_cfg.get("steps_per_epoch", 200))
    batch_pairs = int(align_cfg.get("batch_pairs", 65536))
    adversarial_weight = float(align_cfg.get("adversarial_weight", 0.1))
    warmup = float(align_cfg.get("warmup_epochs", 5))
    out = provenance(cfg, "align-fit", tag, {"encoder_frozen": True, "split_group": split_group, "validation_groups": sorted(validation_groups)})
    protocol = str(align_cfg.get("protocol", "legacy_adversarial_only"))
    rows, best_score, best_confusion, selected_eligible = [], -float("inf"), float("inf"), False
    for epoch in range(1, epochs + 1):
        projector.train(); discriminator.train()
        for step in range(steps_per_epoch):
            index = rng.choice(train_pool, min(batch_pairs, len(train_pool)), replace=len(train_pool) < batch_pairs)
            u = torch.as_tensor(np.asarray(hu[index], np.float32), device=device)
            s = torch.as_tensor(np.asarray(hs[index], np.float32), device=device)
            with torch.no_grad():
                zu, zs = projector(u), projector(s)
            labels = torch.cat([torch.zeros(len(index), dtype=torch.long, device=device), torch.ones(len(index), dtype=torch.long, device=device)])
            discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_loss = F.cross_entropy(discriminator(torch.cat([zu, zs])), labels)
            discriminator_loss.backward(); discriminator_optimizer.step()
            for parameter in discriminator.parameters(): parameter.requires_grad_(False)
            zu, zs = projector(u), projector(s)
            progress = ((epoch - 1) + step / max(1, steps_per_epoch)) / max(warmup, 1e-12)
            weight = adversarial_weight * min(1.0, max(0.0, progress))
            projector_loss = weight * _confusion(discriminator, torch.cat([zu, zs]))
            projector_optimizer.zero_grad(set_to_none=True); projector_loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0); projector_optimizer.step()
            for parameter in discriminator.parameters(): parameter.requires_grad_(True)
        index = rng.choice(validation_pool, min(len(validation_pool), 100_000), replace=False)
        with torch.no_grad():
            u = torch.as_tensor(np.asarray(hu[index], np.float32), device=device); s = torch.as_tensor(np.asarray(hs[index], np.float32), device=device)
            zu, zs = projector(u), projector(s)
            validation_loss = float(_confusion(discriminator, torch.cat([zu, zs])))
            matched_distance = float(cosine_distance(zu, zs).mean())
            qc = _alignment_qc(u, s, zu, zs, discriminator)
        row = {
            "stage": "align", "protocol": protocol, "seed": int(cfg.get("seed", 42)),
            "epoch": epoch, "global_step": epoch * steps_per_epoch, "split": "val",
            "loss_total": adversarial_weight * validation_loss,
            "loss_u": np.nan, "loss_s": np.nan,
            "loss_adv": validation_loss, "loss_disc": qc["loss_disc"],
            "loss_relational": np.nan, "loss_variance": np.nan,
            "loss_covariance": np.nan, "loss_identity": np.nan,
            "disc_accuracy": qc["disc_accuracy"], "learning_rate": learning_rate,
            "adv_weight": adversarial_weight, "adapter_alpha": float(projector.alpha.detach()),
            **{key: value for key, value in qc.items() if key not in {"loss_disc", "disc_accuracy"}},
            # Compatibility aliases for historical clients.
            "validation_confusion": validation_loss,
            "validation_matched_distance": matched_distance,
            "alpha": float(projector.alpha.detach()),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "training_metrics.csv", index=False)
        eligible = bool(qc["checkpoint_eligible"])
        score = float(qc["checkpoint_selection_score"])
        choose = (eligible and (not selected_eligible or score > best_score)) or (not selected_eligible and not eligible and validation_loss < best_confusion)
        if choose:
            selected_eligible = selected_eligible or eligible
            best_score = score if eligible else best_score
            best_confusion = min(best_confusion, validation_loss)
            torch.save({
                "projector": projector.state_dict(), "epoch": epoch,
                "validation_confusion": validation_loss,
                "checkpoint_selection_score": score,
                "checkpoint_eligible": eligible,
                "selection_rule": "eligible_qc_score" if eligible else "fallback_validation_confusion",
                "encoder_frozen": True, "name": "scUS-Align", "protocol": protocol,
            }, out / "best_projector.pt")
    write_json(out / "status.json", {
        "status": "complete", "train_pairs": len(train_pool), "validation_pairs": len(validation_pool),
        "best_validation_confusion": best_confusion, "best_selection_score": best_score if selected_eligible else None,
        "selected_checkpoint_eligible": selected_eligible, "encoder_frozen": True, "protocol": protocol,
    })
    return out


def project(cfg: dict, tag: str | None = None, start: int = 0, stop: int | None = None,
            device_name: str | None = None, fit_tag: str | None = None,
            prepare_tag: str | None = None) -> Path:
    if cfg.get('align', {}).get('mode') == 'joint_reconstruction':
        from .joint_projection import project as project_joint
        return project_joint(cfg,tag,start,stop,device_name,fit_tag,prepare_tag)
    checkpoint = torch.load(stage_directory(cfg, "align-fit", fit_tag or tag) / "best_projector.pt", map_location="cpu")
    if not checkpoint.get("encoder_frozen"):
        raise ValueError("Alignment checkpoint does not attest to a frozen encoder")
    projector = SharedResidualProjector(int(cfg.get("model", {}).get("embed_dim", 128)))
    projector.load_state_dict(checkpoint["projector"])
    device = torch.device(device_name or cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    projector.eval().to(device)
    return encode(cfg, tag, start, stop, str(device), projector, stage="align-project", prepare_tag=prepare_tag)

"""Primitives for falsifiable interpretation of frozen scUS U/S distances."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    key = ":".join(map(str, (seed, *parts))).encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(value)


def load_frozen_encoder(checkpoint: str | Path, device: torch.device, random_seed: int | None = None):
    """Load the historical architecture; random controls never load trained weights."""
    from .models import MaskedModel

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    hp = dict(payload["hyper_parameters"])
    if random_seed is None:
        model = MaskedModel(**hp)
        model.load_state_dict(payload["state_dict"], strict=True)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(random_seed))
            model = MaskedModel(**hp)
    model.eval().requires_grad_(False).to(device)
    return model


def collate_cells(cells: list[dict], device: torch.device):
    """Pad independent cells without changing their token order or values."""
    if not cells:
        raise ValueError("Cannot collate an empty cell list")
    lengths = np.asarray([int(x["value_bins"].numel()) for x in cells], dtype=np.int64)
    width = int(lengths.max())
    arrays = [torch.zeros((len(cells), width), dtype=torch.long, device=device) for _ in range(3)]
    for row, cell in enumerate(cells):
        for target, key in zip(arrays, ("gene_ids", "value_bins", "splice_flags")):
            value = cell[key].reshape(-1).to(device)
            target[row, : len(value)] = value
    return (*arrays, lengths)


def layerwise_hidden(model, gene_ids: torch.Tensor, value_bins: torch.Tensor, splice_flags: torch.Tensor):
    """Return input embedding and every Transformer layer under eval semantics."""
    hidden = model.embed(gene_ids, value_bins, splice_flags)
    outputs = [hidden]
    pad = value_bins.eq(0)
    for layer in model.encoder.layers:
        if hidden.is_cuda:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
        else:
            backend = nullcontext()
        with backend:
            hidden = layer(hidden, attn_mask=pad)
        outputs.append(hidden)
    return outputs


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1)


def paired_distance(hidden: torch.Tensor, value_bins: torch.Tensor, length: int | None = None):
    """Distance for every interleaved gene pair; missing pairs remain NaN."""
    n = int(length if length is not None else value_bins.numel())
    if n % 2:
        raise ValueError("Interleaved sequence length must be even")
    values = value_bins.reshape(-1)[:n]
    h = hidden.reshape(-1, hidden.shape[-1])[:n]
    valid = values[::2].gt(0) & values[1::2].gt(0)
    result = torch.full((n // 2,), float("nan"), dtype=torch.float32, device=h.device)
    result[valid] = cosine_distance(h[::2][valid], h[1::2][valid])
    return result, valid


def derangement(n: int, seed: int, *parts: object) -> np.ndarray:
    if n < 2:
        raise ValueError("A derangement requires at least two elements")
    order = stable_rng(seed, *parts).permutation(n)
    result = np.empty(n, dtype=np.int64)
    result[order] = np.roll(order, 1)
    if np.any(result == np.arange(n)):
        raise AssertionError("Failed to build derangement")
    return result


def paired_null_distances(hidden: torch.Tensor, value_bins: torch.Tensor, seed: int, cell_id: str):
    """Correct and within-cell cross-gene distances on the identical valid gene set."""
    values = value_bins.reshape(-1)
    h = hidden.reshape(-1, hidden.shape[-1])[: len(values)]
    valid = values[::2].gt(0) & values[1::2].gt(0)
    hu, hs = h[::2][valid], h[1::2][valid]
    if len(hu) < 2:
        raise ValueError("At least two paired genes are required")
    wrong = torch.as_tensor(derangement(len(hu), seed, cell_id, "within-gene"), device=h.device)
    return cosine_distance(hu, hs), cosine_distance(hu, hs[wrong])


def matched_cross_gene_indices(pair_bins: np.ndarray, seed: int, *parts: object) -> np.ndarray:
    """Select a nonself S-gene donor matched on the observed U/S bin pair.

    Donors are sampled from the nearest occupied (U-bin, S-bin) stratum.  The
    same stratum is preferred, ties are deterministic, and donor reuse is
    allowed.  Reuse avoids silently discarding singleton strata while the
    explicit no-self rule keeps this a genuine cross-gene null.
    """
    bins = np.asarray(pair_bins, dtype=np.int16)
    if bins.ndim != 2 or bins.shape[1] != 2 or len(bins) < 2:
        raise ValueError("pair_bins must contain at least two U/S pairs")
    if (bins <= 0).any():
        raise ValueError("matched cross-gene indices require observed U/S bins")
    groups: dict[tuple[int, int], list[int]] = {}
    for index, pair in enumerate(bins):
        groups.setdefault((int(pair[0]), int(pair[1])), []).append(index)
    occupied = np.asarray(list(groups), dtype=np.int16)
    result = np.empty(len(bins), dtype=np.int64)
    for index, pair in enumerate(bins):
        candidates = np.asarray([x for x in groups[(int(pair[0]), int(pair[1]))] if x != index], dtype=np.int64)
        if not len(candidates):
            costs = np.abs(occupied - pair[None]).sum(1)
            for stratum in np.flatnonzero(costs == costs.min()):
                possible = np.asarray([x for x in groups[tuple(map(int, occupied[stratum]))] if x != index],
                                      dtype=np.int64)
                if len(possible):
                    candidates = np.r_[candidates, possible]
            if not len(candidates):
                # The closest occupied stratum may be the singleton itself.
                # Move outward until a genuine donor becomes available.
                for cost in np.unique(costs):
                    possible = np.concatenate([
                        np.asarray([x for x in groups[tuple(map(int, occupied[s]))] if x != index], dtype=np.int64)
                        for s in np.flatnonzero(costs == cost)
                    ])
                    if len(possible):
                        candidates = possible
                        break
        if not len(candidates):
            raise AssertionError("Could not find a nonself matched gene donor")
        result[index] = int(candidates[stable_rng(seed, *parts, index, "matched-gene").integers(len(candidates))])
    if np.any(result == np.arange(len(result))):
        raise AssertionError("Matched cross-gene null contains self donors")
    return result


def paired_matched_null_distances(hidden: torch.Tensor, value_bins: torch.Tensor,
                                  seed: int, cell_id: str):
    """Correct and bin-matched cross-gene distances on identical valid genes."""
    values = value_bins.reshape(-1)
    h = hidden.reshape(-1, hidden.shape[-1])[: len(values)]
    valid = values[::2].gt(0) & values[1::2].gt(0)
    hu, hs = h[::2][valid], h[1::2][valid]
    bins = torch.stack([values[::2][valid], values[1::2][valid]], 1).cpu().numpy()
    donors = torch.as_tensor(matched_cross_gene_indices(bins, seed, cell_id), device=h.device)
    return cosine_distance(hu, hs), cosine_distance(hu, hs[donors]), donors


def select_target_genes(cells: Iterable[dict], count: int, seed: int = 42):
    """Coverage-first target selection with deterministic tie breaking."""
    coverage: dict[int, int] = {}
    for cell in cells:
        g = cell["gene_ids"].reshape(-1)[::2].cpu().numpy()
        v = cell["value_bins"].reshape(-1).cpu().numpy()
        for gene in g[(v[::2] > 0) & (v[1::2] > 0)]:
            coverage[int(gene)] = coverage.get(int(gene), 0) + 1
    ranked = sorted(coverage, key=lambda g: (-coverage[g], stable_rng(seed, g).random()))
    return np.asarray(ranked[: min(count, len(ranked))], dtype=np.int64), coverage


def target_positions(cell: dict, targets: np.ndarray, require_paired: bool = True):
    genes = cell["gene_ids"].reshape(-1)[::2].cpu().numpy()
    values = cell["value_bins"].reshape(-1).cpu().numpy().reshape(-1, 2)
    lookup = {int(g): i for i, g in enumerate(genes)}
    pos = np.asarray([lookup[int(g)] for g in targets if int(g) in lookup], dtype=np.int64)
    if require_paired:
        pos = pos[(values[pos] > 0).all(1)]
    return pos


def bin_intervention(cell: dict, pair_positions: np.ndarray, condition: str, severity: int = 1,
                     donor: dict | None = None, seed: int = 42):
    """Apply one registered target intervention; all unmentioned tokens stay exact."""
    result = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cell.items()}
    g = result["gene_ids"].reshape(-1)
    v = result["value_bins"].reshape(-1)
    s = result["splice_flags"].reshape(-1)
    positions = np.asarray(pair_positions, dtype=np.int64)
    if not len(positions):
        raise ValueError("No eligible target positions")
    uidx = torch.as_tensor(2 * positions, dtype=torch.long)
    sidx = uidx + 1
    if condition in {"u_shift", "s_shift", "both_shift"}:
        selected = {"u_shift": [uidx], "s_shift": [sidx], "both_shift": [uidx, sidx]}[condition]
        for idx in selected:
            original = v[idx]
            if (original <= 0).any():
                raise ValueError("Shift interventions require observed tokens")
            v[idx] = torch.clamp(original + int(severity), 1, 15)
    elif condition == "value_swap":
        old = v[uidx].clone(); v[uidx] = v[sidx]; v[sidx] = old
    elif condition == "modality_flag_swap":
        old = s[uidx].clone(); s[uidx] = s[sidx]; s[sidx] = old
    elif condition == "modality_full_swap":
        oldv, olds = v[uidx].clone(), s[uidx].clone()
        v[uidx], s[uidx] = v[sidx].clone(), s[sidx].clone()
        v[sidx], s[sidx] = oldv, olds
    elif condition == "u_low_bin":
        v[uidx] = 1
    elif condition == "s_low_bin":
        v[sidx] = 1
    elif condition == "gene_only_s":
        if len(positions) < 2:
            raise ValueError("Gene-only intervention needs at least two targets")
        permutation = derangement(len(positions), seed, cell["cell_id"], "gene-only")
        g[sidx] = g[sidx[torch.as_tensor(permutation)]]
    elif condition in {"donor_s", "donor_values"}:
        if donor is None or donor.get("cell_id") == cell.get("cell_id"):
            raise ValueError("A nonself donor is required")
        dg = donor["gene_ids"].reshape(-1)[::2].cpu().numpy()
        dv = donor["value_bins"].reshape(-1).cpu().numpy().reshape(-1, 2)
        lookup = {int(x): i for i, x in enumerate(dg)}
        own_genes = g[uidx].cpu().numpy()
        for local, gene in enumerate(own_genes):
            if int(gene) not in lookup:
                continue
            donor_values = dv[lookup[int(gene)]]
            if condition == "donor_s":
                if donor_values[1] > 0:
                    v[sidx[local]] = int(donor_values[1])
            elif (donor_values > 0).all():
                v[uidx[local]] = int(donor_values[0])
                v[sidx[local]] = int(donor_values[1])
    else:
        raise ValueError(f"Unknown intervention: {condition}")
    return result


def grouped_masks(pair_positions: np.ndarray, modality: int, groups: int, repeats: int,
                  seed: int, cell_id: str):
    """Yield deterministic target-token groups while never masking their partners."""
    positions = np.asarray(pair_positions, dtype=np.int64)
    for repeat in range(repeats):
        order = stable_rng(seed, cell_id, modality, repeat, "group-mask").permutation(len(positions))
        for group in np.array_split(order, groups):
            if len(group):
                yield repeat, 2 * positions[group] + int(modality)


@torch.inference_mode()
def conditional_surprisal(model, cell: dict, pair_positions: np.ndarray, modality: int,
                          mode: str = "exact", groups: int = 8, repeats: int = 2,
                          seed: int = 42):
    """Score masked target bins with the frozen historical prediction head."""
    if modality not in (0, 1):
        raise ValueError("modality must be 0 (U) or 1 (S)")
    device = next(model.parameters()).device
    g = cell["gene_ids"].to(device).clone()
    v = cell["value_bins"].to(device).clone()
    s = cell["splice_flags"].to(device).clone()
    positions = np.asarray(pair_positions, dtype=np.int64)
    token_positions = 2 * positions + modality
    if (v[0, torch.as_tensor(token_positions, device=device)] <= 0).any():
        raise ValueError("Surprisal targets must be observed")
    scores = {int(p): [] for p in positions}
    if mode == "exact":
        batches = [(0, np.asarray([p], dtype=np.int64)) for p in token_positions]
    elif mode == "grouped":
        batches = list(grouped_masks(positions, modality, groups, repeats, seed, str(cell["cell_id"])))
    else:
        raise ValueError("mode must be exact or grouped")
    for _, masked_tokens in batches:
        masked = v.clone()
        idx = torch.as_tensor(masked_tokens, dtype=torch.long, device=device)
        targets = v[0, idx].long()
        masked[0, idx] = model.mask_token
        logits = model.head_gene(model._encode(g, masked, s))[0, idx].float()
        values = -F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
        for token, value in zip(masked_tokens, values.cpu().tolist()):
            scores[int(token // 2)].append(float(value))
    return {p: float(np.mean(values)) for p, values in scores.items() if values}


def empirical_conditional_surprisal(source: np.ndarray, partner: np.ndarray, train_rows: np.ndarray,
                                    alpha: float = 1.0):
    """Per-gene Laplace-smoothed -log p(source_bin | partner_bin)."""
    source, partner = np.asarray(source), np.asarray(partner)
    if source.shape != partner.shape or source.ndim != 2:
        raise ValueError("source and partner must be matching cell x gene matrices")
    tables = np.full((source.shape[1], 16, 16), alpha, dtype=np.float64)
    for gene in range(source.shape[1]):
        valid = (source[train_rows, gene] > 0) & (partner[train_rows, gene] > 0)
        np.add.at(tables[gene], (partner[train_rows, gene][valid], source[train_rows, gene][valid]), 1)
    probabilities = tables / tables.sum(axis=2, keepdims=True)
    result = np.full(source.shape, np.nan, dtype=np.float32)
    for gene in range(source.shape[1]):
        valid = (source[:, gene] > 0) & (partner[:, gene] > 0)
        result[valid, gene] = -np.log(probabilities[gene, partner[valid, gene], source[valid, gene]])
    return result

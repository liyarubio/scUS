from __future__ import annotations

import hashlib
import numpy as np


def rank_bin(values: np.ndarray, bins: int = 15) -> np.ndarray:
    """Rank positive values into 1..bins; zero is reserved for padding."""
    x = np.asarray(values)
    out = np.zeros(x.shape, dtype=np.uint8)
    positive = x > 0
    if positive.any():
        vals = x[positive]
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(len(vals), dtype=np.int64)
        ranks[order] = np.arange(len(vals))
        out[positive] = np.minimum(bins, 1 + ranks * bins // max(1, len(vals)))
    return out


def select_genes(mu: np.ndarray, ms: np.ndarray, max_genes: int, cell_id: str, seed: int = 42, force=()) -> np.ndarray:
    available = np.flatnonzero((mu > 0) | (ms > 0))
    forced = np.asarray(list(force), dtype=np.int64)
    forced = forced[(forced >= 0) & (forced < len(mu))]
    available = np.unique(np.r_[available, forced])
    if len(available) <= max_genes:
        return available
    remaining = np.setdiff1d(available, forced)
    digest = hashlib.blake2b(f"{seed}:{cell_id}".encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    picked = rng.choice(remaining, max_genes - len(forced), replace=False)
    return np.sort(np.unique(np.r_[forced, picked]))


def tokenize_pair(mu, ms, vocab_ids, max_genes=1000, bins=15, cell_id="cell", seed=42, force=()):
    selected = select_genes(mu, ms, max_genes, cell_id, seed, force)
    values = np.empty(2 * len(selected), np.float32)
    values[0::2], values[1::2] = mu[selected], ms[selected]
    value_bins = rank_bin(values, bins).astype(np.int64)
    gene_ids = np.repeat(np.asarray(vocab_ids)[selected], 2).astype(np.int64)
    splice_flags = np.tile(np.array([0, 1], np.int64), len(selected))
    valid_pairs = (value_bins[0::2] > 0) & (value_bins[1::2] > 0)
    return selected, gene_ids, value_bins, splice_flags, valid_pairs

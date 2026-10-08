from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..runtime import provenance, write_json


def _vocabulary(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def build_mapping_audit(adata, cfg: dict) -> pd.DataFrame:
    """Audit every source gene and explain whether it enters the vocabulary."""
    vocab = _vocabulary(cfg["vocab"])
    species = cfg.get("species", "human").lower()
    source_genes = pd.Index(adata.var_names.astype(str))
    audit = pd.DataFrame({"source_gene": source_genes})
    if species == "mouse":
        if not cfg.get("ortholog"):
            raise ValueError("Mouse input requires an ortholog table")
        table = pd.read_csv(cfg["ortholog"], sep=None, engine="python", dtype=str).fillna("")
        required = {"Gene stable ID", "Human gene name", "Human homology type"}
        if not required.issubset(table):
            raise ValueError(f"Ortholog table is missing columns: {sorted(required-set(table))}")
        table["source_gene"] = table["Gene stable ID"].astype(str)
        table["target_symbol"] = table["Human gene name"].str.upper()
        table = table[table.source_gene.isin(source_genes)]
        strict = table[table["Human homology type"].eq("ortholog_one2one")].copy()
        first_target = strict.drop_duplicates("source_gene").set_index("source_gene").target_symbol
        audit["target_symbol"] = audit.source_gene.map(first_target).fillna("")
        has_record = audit.source_gene.isin(table.source_gene)
        has_strict = audit.source_gene.isin(strict.source_gene)
        source_ambiguous = strict.source_gene.duplicated(False)
        target_ambiguous = strict.target_symbol.ne("") & strict.target_symbol.duplicated(False)
        ambiguous_sources = set(strict.loc[source_ambiguous | target_ambiguous, "source_gene"])
        audit["reason"] = "mapped"
        audit.loc[~has_record, "reason"] = "no_ortholog_record"
        audit.loc[has_record & ~has_strict, "reason"] = "not_strict_one_to_one"
        audit.loc[has_strict & audit.target_symbol.eq(""), "reason"] = "missing_human_symbol"
        audit.loc[audit.source_gene.isin(ambiguous_sources), "reason"] = "ambiguous_source_or_target"
        audit.loc[audit.reason.eq("mapped") & ~audit.target_symbol.isin(vocab), "reason"] = "target_not_in_vocab"
    elif species == "human":
        symbol_column = cfg.get("gene_symbol_column")
        symbols = adata.var[symbol_column].astype(str) if symbol_column else pd.Series(adata.var_names.astype(str), index=adata.var_names)
        audit["target_symbol"] = symbols.str.upper().to_numpy()
        audit["reason"] = "mapped"
        audit.loc[audit.target_symbol.eq("") | audit.target_symbol.eq("NAN"), "reason"] = "missing_gene_symbol"
        audit.loc[audit.target_symbol.duplicated(False), "reason"] = "ambiguous_target"
        audit.loc[audit.reason.eq("mapped") & ~audit.target_symbol.isin(vocab), "reason"] = "target_not_in_vocab"
    else:
        raise ValueError(f"Unsupported species: {species}")
    audit.insert(1, "source_column", np.arange(adata.n_vars))
    audit["vocab_id"] = audit.target_symbol.map(vocab).astype("Int64")
    return audit


def build_mapping(adata, cfg: dict) -> pd.DataFrame:
    audit = build_mapping_audit(adata, cfg)
    result = audit[audit.reason.eq("mapped")].drop(columns="reason").copy()
    result["vocab_id"] = result.vocab_id.astype(int)
    return result.sort_values("source_column").reset_index(drop=True)


def prepare_data(cfg: dict, tag: str | None = None) -> Path:
    import anndata as ad

    data_cfg = cfg.get("data", cfg)
    adata = ad.read_h5ad(data_cfg["input"], backed="r")
    for layer in (data_cfg.get("mu_layer", "Mu"), data_cfg.get("ms_layer", "Ms")):
        if layer not in adata.layers:
            raise ValueError(f"Required layer '{layer}' is absent; available={list(adata.layers)}")
    out = provenance(cfg, "prepare", tag, {"shape": list(adata.shape)})
    audit = build_mapping_audit(adata, data_cfg)
    mapping = audit[audit.reason.eq("mapped")].drop(columns="reason").copy()
    mapping["vocab_id"] = mapping.vocab_id.astype(int)
    mapping.to_csv(out / "gene_mapping.csv", index=False)
    audit.to_csv(out / "gene_mapping_audit.csv", index=False)
    obs = adata.obs.copy()
    obs.insert(0, "cell_index", np.arange(adata.n_obs))
    if "cell_id" not in obs:
        obs.insert(1, "cell_id", adata.obs_names.astype(str))
    obs.to_parquet(out / "cells.parquet", index=False)
    align_cfg = cfg.get("align", {})
    strata = [x for x in align_cfg.get("strata", []) if x in obs]
    maximum = int(align_cfg.get("reservoir_max_cells", min(50_000, adata.n_obs)))
    if strata:
        hashed = pd.util.hash_pandas_object(obs[strata].astype(str), index=False).to_numpy() ^ np.uint64(cfg.get("seed", 42))
        table = obs[strata + ["cell_index"]].copy()
        table["_hash"] = hashed
        n_groups = max(1, table.groupby(strata, observed=True).ngroups)
        cap = max(1, int(np.ceil(maximum / n_groups)))
        reservoir = table.sort_values("_hash").groupby(strata, observed=True).head(cap).sort_values("_hash").head(maximum).cell_index.to_numpy()
    else:
        rng = np.random.default_rng(cfg.get("seed", 42))
        reservoir = rng.choice(adata.n_obs, min(maximum, adata.n_obs), replace=False)
    np.save(out / "reservoir_cells.npy", np.sort(reservoir))
    summary = {"status": "complete", "cells": adata.n_obs, "genes": adata.n_vars, "mapped_genes": len(mapping), "mapping_reasons": audit.reason.value_counts().to_dict(), "layers": list(adata.layers), "reservoir_cells": len(reservoir), "strata": strata}
    write_json(out / "status.json", summary)
    return out

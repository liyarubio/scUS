from __future__ import annotations

import bisect
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from lightning.pytorch import LightningDataModule
except ImportError:  # Base/zero-shot installations do not require Lightning.
    LightningDataModule = object

from .tokens import tokenize_pair


def prepare_pretraining_shards(cfg: dict) -> list[Path]:
    import anndata as ad

    data_cfg = cfg["data"]
    source = Path(data_cfg.get("input_dir", data_cfg.get("input", ".")))
    files = [source] if source.is_file() else sorted(source.glob(data_cfg.get("pattern", "*.h5ad")))
    if not files:
        raise FileNotFoundError(f"No H5AD files found under {source}")
    output = Path(data_cfg.get("token_dir", "data/tokens")); output.mkdir(parents=True, exist_ok=True)
    with Path(data_cfg["vocab"]).open() as handle: vocab = json.load(handle)
    paths = []
    for dataset_id, path in enumerate(files):
        target = output / f"{path.stem}.pt"
        if target.exists() and not data_cfg.get("force", False):
            paths.append(target); continue
        adata = ad.read_h5ad(path)
        mu_layer, ms_layer = data_cfg.get("mu_layer", "Mu"), data_cfg.get("ms_layer", "Ms")
        if data_cfg.get("compute_moments", False):
            import scanpy as sc
            import scvelo as scv
            sc.pp.filter_cells(adata, min_genes=int(data_cfg.get("min_genes", 100)))
            sc.pp.filter_genes(adata, min_cells=int(data_cfg.get("min_cells", 3)))
            scv.pp.filter_and_normalize(adata)
            scv.pp.moments(adata, use_highly_variable=bool(data_cfg.get("highly_variable_only", True)))
        if mu_layer not in adata.layers or ms_layer not in adata.layers:
            raise ValueError(f"{path} lacks {mu_layer}/{ms_layer}")
        symbols = adata.var[data_cfg["gene_symbol_column"]].astype(str).str.upper() if data_cfg.get("gene_symbol_column") else adata.var_names.astype(str).str.upper()
        keep = np.array([symbol in vocab for symbol in symbols])
        columns = np.flatnonzero(keep); vocab_ids = np.array([vocab[symbol] for symbol in symbols[keep]], np.int64)
        mu, ms = adata.layers[mu_layer][:, columns], adata.layers[ms_layer][:, columns]
        if hasattr(mu, "toarray"): mu = mu.toarray()
        if hasattr(ms, "toarray"): ms = ms.toarray()
        mu, ms = np.asarray(mu, np.float32), np.asarray(ms, np.float32)
        if data_cfg.get("log1p", False): mu, ms = np.log1p(mu), np.log1p(ms)
        records = {"gene_ids": [], "value_bins": [], "splice_flags": [], "batch_id": [], "cell_id": []}
        for row, cell_id in enumerate(adata.obs_names.astype(str)):
            _, genes, values, splice, _ = tokenize_pair(mu[row], ms[row], vocab_ids, int(data_cfg.get("top_k_genes", 1000)), int(cfg.get("model", {}).get("bin_size", 15)), cell_id, int(cfg.get("seed", 618)))
            records["gene_ids"].append(torch.as_tensor(genes, dtype=torch.int32)); records["value_bins"].append(torch.as_tensor(values, dtype=torch.uint8)); records["splice_flags"].append(torch.as_tensor(splice, dtype=torch.int8)); records["batch_id"].append(dataset_id); records["cell_id"].append(cell_id)
        torch.save(records, target); paths.append(target)
    (output / "index.json").write_text(json.dumps([str(path.resolve()) for path in paths], indent=2) + "\n")
    return paths


class TokenShardDataset(Dataset):
    def __init__(self, token_dir: str | Path):
        self.paths = [Path(path) for path in json.loads((Path(token_dir) / "index.json").read_text())]
        self.lengths = []
        for path in self.paths:
            shard = torch.load(path, map_location="cpu", weights_only=False); self.lengths.append(len(shard["gene_ids"])); del shard
        self.cumulative = np.cumsum(self.lengths).tolist(); self._cached_index = None; self._cached = None

    def __len__(self): return self.cumulative[-1]

    def __getitem__(self, index):
        shard_index = bisect.bisect_right(self.cumulative, index); offset = index - (self.cumulative[shard_index-1] if shard_index else 0)
        if self._cached_index != shard_index:
            self._cached = torch.load(self.paths[shard_index], map_location="cpu", weights_only=False); self._cached_index = shard_index
        return {name: self._cached[name][offset] for name in ("gene_ids", "value_bins", "splice_flags", "batch_id")}


def collate_tokens(records):
    length = max(len(record["gene_ids"]) for record in records); batch = len(records)
    result = {"gene_ids": torch.zeros((batch, length), dtype=torch.long), "value_bins": torch.zeros((batch, length), dtype=torch.long), "splice_flags": torch.zeros((batch, length), dtype=torch.long), "batch_id": torch.zeros(batch, dtype=torch.long)}
    for row, record in enumerate(records):
        n = len(record["gene_ids"])
        for name in ("gene_ids", "value_bins", "splice_flags"): result[name][row, :n] = record[name].long()
        result["batch_id"][row] = int(record["batch_id"])
    return result


class TokenDataModule(LightningDataModule):
    """Lightning data module for deterministic token-shard train/validation splits."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.validation_set = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is not None:
            return
        dataset = TokenShardDataset(self.cfg["data"]["token_dir"])
        fraction = float(self.cfg.get("pretrain", {}).get("validation_fraction", 0.02))
        n_validation = max(1, int(len(dataset) * fraction))
        if n_validation >= len(dataset):
            raise ValueError("Pretraining requires at least two cells for a train/validation split")
        generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 618)))
        self.train_set, self.validation_set = random_split(
            dataset,
            [len(dataset) - n_validation, n_validation],
            generator=generator,
        )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        options = self.cfg.get("pretrain", {})
        return DataLoader(
            dataset,
            batch_size=int(options.get("batch_size", 32)),
            shuffle=shuffle,
            num_workers=int(options.get("num_workers", 2)),
            collate_fn=collate_tokens,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.validation_set, False)


def make_loaders(cfg: dict):
    """Compatibility helper for callers that need concrete DataLoaders."""
    module = TokenDataModule(cfg)
    module.setup("fit")
    return module.train_dataloader(), module.val_dataloader()

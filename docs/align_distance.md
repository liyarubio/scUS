# Shared-residual Align and U/S distance

## Architecture

```text
H5AD U/S layers + vocabulary + pretrained checkpoint
  → unique gene mapping and interleaved U/S tokens
  → frozen Transformer
      ├─ paired token vectors → raw U/S distance
      └─ cached paired vectors → shared residual projector training
                                   ↓
                 frozen Transformer → trained projector → aligned U/S distance
```

`scus.align.SharedResidualProjector` applies the same function to both U and S:

`P(h) = normalize(LayerNorm(h + sigmoid(a) * MLP(h)))`.

The MLP is Linear(D,256) → GELU → Linear(256,D); sigmoid(a) starts at 0.05.
Its final linear weights start near zero. LayerNorm and output normalization
are active even at initialization, so the initialized projector is not an
exact identity map. The default encoder dimension is 128.

`ModalityDiscriminator` is Linear(D,128) → GELU → Linear(128,2). It predicts
whether a projected token came from U or S. Training alternates discriminator
cross-entropy updates with projector updates against a uniform U/S target:

`L_confusion = mean(-0.5 log p(U) - 0.5 log p(S))`.

The projector loss is the confusion loss multiplied by the configured
adversarial weight with a linear warmup. There is no direct paired-distance
minimization or joint reconstruction loss in this version. The Transformer
is frozen; Align trains from cached vectors without loading encoder weights.
This release supports `align.mode: shared_residual`; the experimental
`joint_reconstruction` branch is not included and is explicitly rejected.

## Split and checkpoint selection

Preparation chooses a bounded cell reservoir, optionally stratified by obs
columns. The encoder caches up to `reservoir_pairs_per_cell` valid pairs
per chosen cell (the first valid pairs in selected-gene order).
Align holds out 20% of distinct `split_group` values, rounded up and with
at least one held-out group. At least two represented groups are required.

Validation checks finite vectors, nonzero coverage, effective rank retention,
distance spread, neighbour overlap and raw/aligned distance correlation.
Checkpoint eligibility requires coverage ≥ 0.999, rank ratio ≥ 0.9,
distance-IQR ratio ≥ 0.5, neighbour overlap ≥ 0.9 and distance correlation
≥ 0.9, plus finite outputs. Among eligible candidates the composite QC score
is maximized. If none qualifies, the smallest validation confusion loss is
saved as a fallback. Check `selected_checkpoint_eligible` in `status.json`:
a saved checkpoint does not by itself establish alignment quality.
Training metrics are generated at runtime; no experiment results are bundled.

## Distance definition and masks

For cell c and gene g, `d(c,g) = 1 - cosine(h_U(c,g), h_S(c,g))`.
After Align, substitute `P(h_U)` and `P(h_S)`. Distances lie in [0,2]:
0 means identical directions, 1 orthogonal directions, 2 opposite directions.
Distances describe representation differences, not signed RNA velocity,
kinetic rates or causal effects. Only genes with positive observed expression
bins in both U and S produce a stored distance. Missing pairs are not zeros.
The cell score is the median of valid gene distances; cells with no valid
pairs receive NaN. The cell embedding averages all positive-bin token vectors.

For tensors already encoded with interleaved U/S tokens:

```python
from scus.distance import us_distance

# hidden: [B, 2G, D]; value_bins: [B, 2G], observed values (no mask tokens)
distances, valid_pairs = us_distance(hidden, value_bins)
# Both outputs have shape [B, G]; invalid distances are NaN.
```

`cosine_distance(u, s)` is also available for matched vectors of shape
`[..., D]`. It computes in float32 and clips rounding errors to [0,2].
Epsilon normalization gives distance 1 when a vector is zero; callers must
handle missingness separately. `us_distance` uses bin validity for that purpose.

## Files and Python entry points

`prepare/` contains ordered `gene_mapping.csv`, cell metadata and reservoir
cell indices. Human inputs map uppercase gene symbols uniquely. Mouse inputs
require the configured strict one-to-one ortholog table, as in the original
project; see `scus.data.prepare.build_mapping_audit` for its column contract.

`zero-shot/` and `align-project/` each contain CSR components `indptr.npy`,
`indices.npy`, `data.npy`; columns index rows of the preparation mapping,
not raw H5AD columns or vocabulary IDs. Rows follow `cell_scores.parquet`.
`pooled_embedding.npy` has one row per processed cell.

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# out and prep are pathlib.Path objects for a distance stage and prepare stage.
ptr = np.load(out / 'indptr.npy')
idx = np.load(out / 'indices.npy')
values = np.load(out / 'data.npy')
n_genes = len(pd.read_csv(prep / 'gene_mapping.csv'))
distance = csr_matrix((values, idx, ptr), shape=(len(ptr) - 1, n_genes))
valid = csr_matrix((np.ones(len(values), dtype=bool), idx, ptr), shape=distance.shape)
# Preserve `valid`: implicit missing entries and genuine zero distances differ.
```

`scus.data.prepare.prepare_data(cfg)`, `scus.zero_shot.encode(cfg)`,
`scus.align.fit(cfg)` and `scus.align.project(cfg)` expose the same workflow
in Python. `encode` supports start/stop ranges and `prepare_tag`; use
`merge_parts` to combine contiguous ranges produced with identical config,
checkpoint and preparation mapping. Projector checkpoints and raw caches
must match the encoder used during inference. Load only trusted checkpoints.

H5AD layers and the distance/reservoir accumulators can use substantial host
memory. Use bounded cell ranges for large inputs. GPU inference uses BF16
autocast and has not been validated in this release's CPU smoke tests.

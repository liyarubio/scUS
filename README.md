# scUS Model

scUS is a masked Transformer for paired unspliced/spliced single-cell inputs.
This repository contains the model, tokenization contract, and pretraining
pipeline. Analysis results, figures, datasets, and checkpoints are deliberately
not included.

## Architecture

```text
Mu/Ms expression values
  -> deterministic gene selection and rank bins
  -> interleaved [gene, value_bin, splice_flag] tokens
  -> summed embeddings + optional batch bias
  -> pre-LN Transformer encoder
       (FlashAttention 2 on CUDA, PyTorch SDPA fallback)
  -> linear value-bin prediction head
  -> masked cross-entropy reconstruction
```

The default configuration uses 128-dimensional embeddings, 8 attention heads,
6 Transformer blocks, 15 expression bins, and a 0.3 masking ratio.

## Input contract

The three token inputs are integer tensors with shape `[batch, sequence]`.
Optional `batch_id` has shape `[batch]`:

| Tensor | Meaning |
|---|---|
| `gene_ids` | vocabulary IDs; the same gene ID appears for U and S |
| `value_bins` | `0` padding, `1..bin_size` observed values, `bin_size+1` mask |
| `splice_flags` | `0` unspliced (U), `1` spliced (S) |
| `batch_id` | optional dataset ID for a trainable/frozen batch bias |

Tokens are interleaved as `U(gene_1), S(gene_1), U(gene_2), S(gene_2), ...`.
`MaskedModel.cell_embedding()` returns a pooled cell embedding; with
`split_splice=True` it additionally returns separate U and S pooled embeddings.

## Install and test

```bash
python -m pip install -e ".[train,dev]"
pytest
```

FlashAttention is optional. CPU or installations without FlashAttention use
PyTorch scaled dot-product attention. When FlashAttention is installed, CUDA
inputs must satisfy that backend's hardware and dtype requirements; use mixed
precision for training. Evaluation disables attention dropout in both backends.

## Pretraining

Prepare a vocabulary JSON mapping gene symbols to integer IDs, then update
`configs/pretrain.yaml` with input and output paths. The training components are
available through `scus.pretrain.train(cfg)` and the included token data module.
Token shards are created by `prepare_pretraining_shards(cfg)`.

Use uppercase gene symbols and reserve ID 0 for padding, for example
`{"<PAD>": 0, "GATA1": 1, "GATA2": 2}`. H5AD inputs must contain the
configured U/S layers and gene symbol column. Paths in YAML are relative to
the configuration file. Gene selection is seeded subsampling of expressed
genes when the limit is exceeded, not expression-sorted top-k selection.

```bash
scus-pretrain configs/pretrain.yaml --prepare-only
scus-pretrain configs/pretrain.yaml
scus-pretrain configs/pretrain.yaml --resume outputs/pretrain/main/pretrain/checkpoints/YOUR_CHECKPOINT.ckpt
```

For CPU runs, set `accelerator: cpu`, `devices: 1`, and `precision: 32-true`
under `pretrain`. Token preparation currently densifies each input dataset;
partition large datasets into smaller H5AD files to fit available RAM.

## Model details

Gene, expression-bin and splice embeddings are summed and passed through
dropout (0.1). Each pre-LN block applies non-causal multi-head attention with
a residual connection, then LayerNorm and a 4x-width GELU feed-forward network
with a second residual connection. There is no positional embedding.
The head produces `[batch, sequence, bin_size + 1]` logits.

Masking samples positive value bins, ensuring at least one masked token per
nonempty cell. Training uses `0.5 * (CE_U + CE_S)`; a splice group without
masked tokens contributes zero. Validation weights the two losses by their
masked-token counts within each batch. Padding targets are ignored.
Adam is used with the configured learning rate.

The YAML uses 6 blocks and learning rate 1e-4. Direct `MaskedModel(...)`
construction defaults to 4 blocks and learning rate 1e-3 for compatibility.
Other defaults are 128 embedding dimensions, 8 heads and mask ratio 0.3.

## Minimal inference

```python
import torch
from scus.models import MaskedModel

# Randomly initialized demonstration; use a trained checkpoint for real inputs.
model = MaskedModel(vocab_size=3, bin_size=15).eval()
batch = {
    "gene_ids": torch.tensor([[1, 1, 2, 2]]),
    "value_bins": torch.tensor([[2, 4, 6, 8]]),
    "splice_flags": torch.tensor([[0, 1, 0, 1]]),
}
with torch.inference_mode():
    logits = model(batch)
    embeddings = model.cell_embedding(batch, split_splice=True)

# Load only checkpoints you trust; vocabulary IDs must match training.
# model = MaskedModel.load_from_checkpoint("model.ckpt", map_location="cpu").eval()
```

Run `pytest` for model/token checks; `pip install -e ".[train,dev]"` also
enables the synthetic H5AD → token shards → CPU training → checkpoint test.

Checkpoints are intentionally excluded from Git. Store released weights as
GitHub Release assets or in external artifact storage.

## License

MIT. See [LICENSE](LICENSE).

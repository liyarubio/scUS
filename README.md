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

Inputs are integer tensors with shape `[batch, sequence]`:

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

FlashAttention is optional. CPU and unsupported CUDA environments use PyTorch
scaled dot-product attention automatically.

## Pretraining

Prepare a vocabulary JSON mapping gene symbols to integer IDs, then update
`configs/pretrain.yaml` with input and output paths. The training components are
available through `scus.pretrain.train(cfg)` and the included token data module.
Token shards are created by `prepare_pretraining_shards(cfg)`.

Checkpoints are intentionally excluded from Git. Store released weights as
GitHub Release assets or in external artifact storage.

## License

MIT. See [LICENSE](LICENSE).

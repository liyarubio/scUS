from __future__ import annotations

import torch.nn as nn

__all__ = ["GeneValueEmbedding"]


class GeneValueEmbedding(nn.Module):
    """Joint embedding of (gene_id, value_bin, splice_flag).

    Parameters
    ----------
    vocab_size : int
        Number of unique gene IDs.
    n_value_bins : int
        Expression bins *including* the `<mask>` token (see design table
        above), but **excluding padding**.  Full value vocabulary size will be
        ``n_value_bins + 1`` (add `<pad>`).
    embed_dim : int
        Dimension of the returned vectors.
    dropout : float, optional
        Dropout rate applied to the summed embedding, by default ``0.1``.
    """

    def __init__(
        self,
        vocab_size: int,
        n_value_bins: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── tables ────────────────────────────────────────────────────
        self.gene_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # value 0 = pad, 1..n_bins = real, n_bins+1 = <mask>
        self.value_embedding = nn.Embedding(n_value_bins + 1, embed_dim, padding_idx=0)
        # splice_flag 0 = unspliced, 1 = spliced
        self.splice_embedding = nn.Embedding(2, embed_dim)

        # ── regularisation ────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, gene_ids, value_bins, splice_flags):  # type: ignore[override]
        """Embed & sum; shape preservation.

        Parameters
        ----------
        gene_ids, value_bins, splice_flags : torch.Tensor, shape ``[B,L]``

        Returns
        -------
        torch.Tensor, shape ``[B,L,embed_dim]``
        """
        emb = (
            self.gene_embedding(gene_ids)
            + self.value_embedding(value_bins)
            + self.splice_embedding(splice_flags)
        )
        return self.dropout(emb)
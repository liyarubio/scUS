from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from lightning.pytorch import LightningModule as _LightningModule
except ImportError:
    class _LightningModule(nn.Module):
        """Inference-only fallback used by base installations."""

        def save_hyperparameters(self) -> None:
            return None

        def log_dict(self, *args, **kwargs) -> None:
            return None

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, map_location=None, **overrides):
            payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            settings = dict(payload.get("hyper_parameters", {}))
            settings.update(overrides)
            model = cls(**settings)
            model.load_state_dict(payload.get("state_dict", payload))
            return model

from .attention import SpatialTransformer
from .embedding_encoder import GeneValueEmbedding

__all__ = ["MaskedModel"]


class MaskedModel(_LightningModule):
    """Masked value-bin Transformer for interleaved U/S gene tokens.

    Padding is value bin 0, observed expression uses bins ``1..bin_size``, and
    ``bin_size + 1`` is the mask token. Splice flags are 0 for U and 1 for S.
    The module names intentionally match the released v1 checkpoint.
    """

    def __init__(
        self,
        vocab_size: int,
        bin_size: int,
        *,
        embed_dim: int = 128,
        lr: float = 1e-3,
        num_heads: int = 8,
        num_layers: int = 4,
        mask_ratio: float = 0.3,
        num_datasets: int | None = None,
        use_batch_embed: bool = False,
        freeze_batch_bias: bool = True,
        predict_mode: str = "gene",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.bin_size = bin_size
        self.mask_token = bin_size + 1
        self.predict_mode = predict_mode
        self.embed = GeneValueEmbedding(vocab_size, bin_size + 1, embed_dim)
        if use_batch_embed and num_datasets is not None:
            self.batch_bias = nn.Embedding(num_datasets, embed_dim)
            self.batch_bias.weight.requires_grad_(not freeze_batch_bias)
        else:
            self.batch_bias = None
        self.encoder = SpatialTransformer(embed_dim, num_heads, num_layers)
        self.head_gene = nn.Linear(embed_dim, bin_size + 1)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)

    def _rand_mask(self, value_bins: torch.Tensor) -> torch.Tensor:
        candidates = value_bins > 0
        mask = torch.rand_like(value_bins, dtype=torch.float).lt(self.mask_ratio) & candidates
        need_mask = (mask.sum(1) == 0) & (candidates.sum(1) > 0)
        if need_mask.any():
            column = torch.multinomial(candidates[need_mask].float(), 1).squeeze(1)
            mask[need_mask, column] = True
        return mask

    def _encode(
        self,
        gene_ids: torch.Tensor,
        value_bins: torch.Tensor,
        splice_flags: torch.Tensor,
        batch_ids: torch.Tensor | None = None,
        *,
        return_all: bool = False,
    ) -> torch.Tensor:
        tokens = self.embed(gene_ids, value_bins, splice_flags)
        if self.batch_bias is not None and batch_ids is not None:
            tokens = tokens + self.batch_bias(batch_ids).unsqueeze(1)
        # return_all is retained for checkpoint/client API compatibility.
        return self.encoder(tokens, mask=value_bins == 0)

    @staticmethod
    def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denominator = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (value * mask.unsqueeze(-1)).sum(dim=1) / denominator

    @torch.inference_mode()
    def cell_embedding(
        self,
        batch: dict[str, Any],
        *,
        split_splice: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        gene_ids = batch.get("gene_ids", batch.get("topk"))
        value_bins = batch.get("value_bins", batch.get("bins"))
        splice_flags = batch["splice_flags"]
        hidden = self._encode(gene_ids, value_bins, splice_flags, batch.get("batch_id"), return_all=True)
        valid = value_bins > 0
        if not split_splice:
            return self._safe_mean(hidden, valid)
        return {
            "emb_all": self._safe_mean(hidden, valid),
            "emb_u": self._safe_mean(hidden, valid & (splice_flags == 0)),
            "emb_s": self._safe_mean(hidden, valid & (splice_flags == 1)),
        }

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        gene_ids = batch.get("gene_ids", batch.get("topk"))
        value_bins = batch.get("value_bins", batch.get("bins"))
        hidden = self._encode(gene_ids, value_bins, batch["splice_flags"], batch.get("batch_id"))
        return self.head_gene(hidden)

    def _masked_loss(self, batch: dict[str, Any], *, prefix: str) -> torch.Tensor:
        gene_ids = batch.get("gene_ids", batch.get("topk"))
        original = batch.get("value_bins", batch.get("bins"))
        splice_flags = batch["splice_flags"]
        masked = original.clone()
        prediction_mask = self._rand_mask(original)
        masked[prediction_mask] = self.mask_token
        logits = self.head_gene(self._encode(gene_ids, masked, splice_flags, batch.get("batch_id")))
        mask_u = prediction_mask & (splice_flags == 0)
        mask_s = prediction_mask & (splice_flags == 1)
        loss_u = self.loss_ce(logits[mask_u], original[mask_u].long()) if mask_u.any() else 0.0 * logits.sum()
        loss_s = self.loss_ce(logits[mask_s], original[mask_s].long()) if mask_s.any() else 0.0 * logits.sum()
        count_u, count_s = mask_u.sum(), mask_s.sum()
        if prefix == "train":
            # Preserve the v1 objective: equal U/S weighting per training batch.
            loss = 0.5 * (loss_u + loss_s)
        else:
            count = (count_u + count_s).clamp(min=1)
            loss = (loss_u * count_u + loss_s * count_s) / count
        metrics = {f"{prefix}_loss": loss, f"{prefix}_loss_u": loss_u, f"{prefix}_loss_s": loss_s}
        if prefix == "val":
            metrics.update({"val_n_u": count_u.float(), "val_n_s": count_s.float()})
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=gene_ids.size(0),
            sync_dist=prefix == "val",
        )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._masked_loss(batch, prefix="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._masked_loss(batch, prefix="val")

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        embedding = self.cell_embedding(batch, split_splice=getattr(self, "split_splice", False))
        output = {name: value.cpu() for name, value in embedding.items()} if isinstance(embedding, dict) else {"emb": embedding.cpu()}
        if "batch_id" in batch:
            value = batch["batch_id"]
            output["batch_id"] = value.cpu() if torch.is_tensor(value) else value
        for name, value in batch.items():
            if name.startswith("obs_"):
                output[name] = value.cpu() if torch.is_tensor(value) else value
        return output

    @torch.inference_mode()
    def get_attention_maps(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        gene_ids = batch.get("gene_ids", batch.get("topk"))
        value_bins = batch.get("value_bins", batch.get("bins"))
        tokens = self.embed(gene_ids, value_bins, batch["splice_flags"])
        if self.batch_bias is not None and batch.get("batch_id") is not None:
            tokens = tokens + self.batch_bias(batch["batch_id"]).unsqueeze(1)
        _, weights = self.encoder(tokens, mask=value_bins == 0, return_weights=True)
        return weights

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Flash‑Attention v2 (optional). If unavailable, fall back to torch attention.
_FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn.flash_attn_interface import (  # type: ignore
        flash_attn_qkvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
    from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore

    _FLASH_ATTN_AVAILABLE = True
except Exception:
    _FLASH_ATTN_AVAILABLE = False

__all__ = [
    "FlashMHA",
    "TransformerBlock",
    "SpatialTransformer",
]


# ---------------------------------------------------------------------
# Flash‑Attention based multi‑head attention (padding‑aware)
# ---------------------------------------------------------------------
class FlashMHA(nn.Module):
    """Multi‑Head Attention powered by flash‑attn 2.x kernels.

    • **Supports `attn_mask`**: `(B,N)` Bool where **True = pad / ignore**.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # QKV → (B,N,3H*Dh)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # shared kernel hyper‑params
        self.dropout_p = dropout
        self.softmax_scale = None   # default 1/√Dh inside kernel
        self.causal = False         # encoder mode only

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                    # (B,N,D)
        attn_mask: torch.Tensor | None = None,  # (B,N) True = pad / ignore
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape

        # 1. project to packed QKV
        qkv_lin = self.qkv_proj(x)

        # Explicit attention path for diagnostic weight extraction.
        if return_weights:
            q, k, v = qkv_lin.chunk(3, dim=-1)
            q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

            scale = self.softmax_scale or (1.0 / math.sqrt(self.head_dim))
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale

            if attn_mask is not None:
                mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
                attn_scores = attn_scores.masked_fill(mask_expanded, float("-inf"))

            attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, N, N)

            out = torch.matmul(attn_weights, v)  # (B, H, N, Dh)
            out = out.transpose(1, 2).reshape(B, N, self.embed_dim)
            out = self.out_proj(out)
            return out, attn_weights

        # Efficient training and standard inference path.
        if not _FLASH_ATTN_AVAILABLE or not x.is_cuda:
            # ---- fallback: torch scaled dot-product attention -----------------
            # qkv_lin: (B, N, 3*D) -> q,k,v: (B, H, N, Dh)
            q, k, v = qkv_lin.chunk(3, dim=-1)
            q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

            # attn_mask: (B,N) True = pad. Use an additive mask because SDPA's
            # boolean-mask convention differs from MultiheadAttention.
            am = None
            if attn_mask is not None:
                am = torch.zeros((B, 1, 1, N), dtype=q.dtype, device=q.device)
                am = am.masked_fill(attn_mask[:, None, None, :].to(torch.bool), float("-inf"))

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=am, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False
            )  # (B,H,N,Dh)
            out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
            return self.out_proj(out)

        qkv = qkv_lin.view(B, N, 3, self.num_heads, self.head_dim).contiguous()  # (B,N,3,H,Dh)

        # 2. choose kernel path -------------------------------------------------
        if attn_mask is not None and attn_mask.any():
            # ---- with padding -------------------------------------------------
            keep = (~attn_mask).int()  # 1 = keep, 0 = pad
            qkv_upd, indices, cu_seqlens, max_seqlen, _ = unpad_input(qkv, keep)

            out_upd = flash_attn_varlen_qkvpacked_func(
                qkv_upd,
                cu_seqlens,
                max_seqlen,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )  # (total_tokens, H, Dh)

            out = pad_input(out_upd, indices, B, N)  # (B,N,H,Dh)
        else:
            # ---- contiguous sequence (no pad) --------------------------------
            out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )  # (B,N,H,Dh)

        # 3. reshape & output proj --------------------------------------------
        out = out.reshape(B, N, self.embed_dim)  # (B,N,D)
        return self.out_proj(out)


# ---------------------------------------------------------------------
# Transformer Block (LN → MHA → FFN) using FlashMHA
# ---------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre‑LN Transformer block with Flash‑Attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FlashMHA(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,                       # (B,N,D)
        attn_mask: torch.Tensor | None = None,  # (B,N)
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_weights:
            attn_out, attn_weights = self.attn(
                self.norm1(x), attn_mask=attn_mask, return_weights=True
            )
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
            x = x + self.ffn(self.norm2(x))
            return x


# ---------------------------------------------------------------------
# Stacked Spatial Transformer (unchanged public API)
# ---------------------------------------------------------------------
class SpatialTransformer(nn.Module):
    """A stack of *num_layers* TransformerBlock with Flash Attention."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,                     # (B,N,D)
        mask: torch.Tensor | None = None,    # (B,N) pad mask
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        all_attn_weights: list[torch.Tensor] = []
        for layer in self.layers:
            if return_weights:
                x, attn_w = layer(x, attn_mask=mask, return_weights=True)
                all_attn_weights.append(attn_w)
            else:
                x = layer(x, attn_mask=mask)
        if return_weights:
            return x, all_attn_weights
        return x

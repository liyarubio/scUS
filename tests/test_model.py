import torch

from scus.models import GeneValueEmbedding, MaskedModel, SpatialTransformer


def batch(batch_size=2, length=6, bins=15):
    return {
        "gene_ids": torch.randint(1, 20, (batch_size, length)),
        "value_bins": torch.randint(1, bins + 1, (batch_size, length)),
        "splice_flags": torch.tensor([[0, 1, 0, 1, 0, 1]] * batch_size),
    }


def test_embedding_shape_and_padding():
    layer = GeneValueEmbedding(32, 16, 8)
    out = layer(torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long))
    assert out.shape == (2, 4, 8)


def test_transformer_padding_and_attention_weights():
    model = SpatialTransformer(embed_dim=8, num_heads=2, num_layers=2)
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[False, False, True, True], [False, False, False, True]])
    out, weights = model(x, mask=mask, return_weights=True)
    assert out.shape == x.shape
    assert len(weights) == 2 and weights[0].shape == (2, 2, 4, 4)


def test_masked_model_loss_backward_and_embeddings():
    model = MaskedModel(vocab_size=32, bin_size=15, embed_dim=8, num_heads=2, num_layers=1, mask_ratio=0.5)
    data = batch()
    loss = model._masked_loss(data, prefix="train")
    loss.backward()
    assert torch.isfinite(loss)
    embedding = model.cell_embedding(data)
    assert embedding.shape == (2, 8)


def test_checkpoint_round_trip(tmp_path):
    model = MaskedModel(vocab_size=16, bin_size=15, embed_dim=8, num_heads=2, num_layers=1)
    path = tmp_path / "model.ckpt"
    settings = dict(vocab_size=16, bin_size=15, embed_dim=8, num_heads=2, num_layers=1)
    torch.save(
        {
            "pytorch-lightning_version": "2.5.0",
            "hyper_parameters": settings,
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored = MaskedModel.load_from_checkpoint(path, map_location="cpu")
    assert restored.embed.gene_embedding.weight.shape == model.embed.gene_embedding.weight.shape
    model.eval()
    restored.eval()
    data = batch()
    data['gene_ids'].clamp_(max=15)
    torch.testing.assert_close(model(data), restored(data))

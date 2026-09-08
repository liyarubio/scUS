from __future__ import annotations

import json
from pathlib import Path

from .runtime import provenance, write_json


def main() -> None:
    import argparse
    from .config import load_config

    parser = argparse.ArgumentParser(description="Pretrain the scUS masked Transformer")
    parser.add_argument("config", help="YAML training configuration")
    parser.add_argument("--resume", default=None, help="Optional Lightning checkpoint")
    parser.add_argument("--prepare-only", action="store_true", help="Create token shards without training")
    args = parser.parse_args()
    if args.prepare_only:
        from .data.pretraining import prepare_pretraining_shards
        prepare_pretraining_shards(load_config(args.config))
        return
    train(load_config(args.config), resume=args.resume)


def train(cfg: dict, resume: str | None = None):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    from .data.pretraining import TokenDataModule
    from .models import MaskedModel

    pl.seed_everything(int(cfg.get("seed", 618)), workers=True)
    data_module = TokenDataModule(cfg)
    with Path(cfg["data"]["vocab"]).open() as handle:
        vocab = json.load(handle)
    vocab_size = max(vocab.values()) + 1
    model_cfg, options = cfg.get("model", {}), cfg.get("pretrain", {})
    model = MaskedModel(vocab_size=vocab_size, bin_size=int(model_cfg.get("bin_size", 15)), embed_dim=int(model_cfg.get("embed_dim", 128)), num_heads=int(model_cfg.get("num_heads", 8)), num_layers=int(model_cfg.get("num_layers", 6)), mask_ratio=float(model_cfg.get("mask_ratio", 0.3)), lr=float(options.get("learning_rate", 1e-4)), num_datasets=options.get("num_datasets"), use_batch_embed=bool(options.get("use_batch_embed", False)))
    out = provenance(cfg, "pretrain")
    checkpoint = ModelCheckpoint(dirpath=out / "checkpoints", filename="scus-{epoch:02d}-{val_loss_epoch:.4f}", monitor="val_loss_epoch", mode="min", save_top_k=int(options.get("save_top_k", 3)))
    trainer = pl.Trainer(max_epochs=int(options.get("epochs", 200)), accelerator=options.get("accelerator", "gpu"), devices=options.get("devices", "auto"), strategy=options.get("strategy", "auto"), precision=options.get("precision", "16-mixed"), gradient_clip_val=1.0, callbacks=[checkpoint, LearningRateMonitor("epoch")], logger=[CSVLogger(out, name="csv"), TensorBoardLogger(out, name="tensorboard")])
    trainer.fit(model, datamodule=data_module, ckpt_path=resume)
    write_json(out / "status.json", {"status": "complete", "best_checkpoint": checkpoint.best_model_path, "best_score": checkpoint.best_model_score})
    return out


if __name__ == "__main__":
    main()

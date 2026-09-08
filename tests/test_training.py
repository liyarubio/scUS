import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_training_module_imports():
    from scus.pretrain import main
    assert callable(main)


def test_cpu_training_pipeline(tmp_path):
    pytest.importorskip('lightning.pytorch')
    ad = pytest.importorskip('anndata')
    pytest.importorskip('tensorboard')
    from scus.data.pretraining import prepare_pretraining_shards, TokenDataModule
    from scus.pretrain import train
    from scus.models import MaskedModel

    matrix = np.arange(1, 17, dtype=np.float32).reshape(8, 2)
    data = ad.AnnData(matrix)
    data.var_names = ['G1', 'G2']
    data.layers['Mu'] = matrix
    data.layers['Ms'] = matrix + 1
    source = tmp_path / 'tiny.h5ad'
    data.write_h5ad(source)
    vocab = tmp_path / 'vocab.json'
    vocab.write_text(json.dumps({'<PAD>': 0, 'G1': 1, 'G2': 3}))
    cfg = {
        'seed': 42, 'output_dir': str(tmp_path / 'out'),
        'data': {'input': str(source), 'vocab': str(vocab), 'token_dir': str(tmp_path / 'tokens')},
        'model': {'embed_dim': 8, 'num_heads': 2, 'num_layers': 1, 'bin_size': 15},
        'pretrain': {'accelerator': 'cpu', 'devices': 1, 'precision': '32-true',
                     'epochs': 1, 'batch_size': 4, 'num_workers': 0},
    }
    prepare_pretraining_shards(cfg)
    out = train(cfg)
    status = json.loads((out / 'status.json').read_text())
    checkpoint = Path(status['best_checkpoint'])
    assert checkpoint.is_file()
    model = MaskedModel.load_from_checkpoint(checkpoint, map_location='cpu').eval()
    module = TokenDataModule(cfg)
    module.setup()
    batch = next(iter(module.val_dataloader()))
    assert torch.isfinite(model(batch)).all()
    assert torch.isfinite(model.cell_embedding(batch)).all()
    assert status['status'] == 'complete'

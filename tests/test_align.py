import copy
import json

import numpy as np
import pytest
import torch

ad = pytest.importorskip('anndata')
pd = pytest.importorskip('pandas')
pytest.importorskip('pyarrow')
pytest.importorskip('scipy')

from scus.align import SharedResidualProjector, fit, project
from scus.data.prepare import prepare_data
from scus.models import MaskedModel
from scus.zero_shot import encode, load_encoder, merge_parts


@pytest.fixture
def cfg(tmp_path):
    rng = np.random.default_rng(42)
    matrix = rng.uniform(1, 10, (12, 4)).astype('float32')
    data = ad.AnnData(matrix, obs=pd.DataFrame({'sample': ['A'] * 6 + ['B'] * 6},
                                             index=[f'c{i}' for i in range(12)]))
    data.var_names = ['G1', 'G2', 'G3', 'G4']
    data.layers['Mu'] = matrix
    data.layers['Ms'] = matrix[:, ::-1].copy()
    data.layers['Mu'][0, 0] = 0
    source = tmp_path / 'tiny.h5ad'
    data.write_h5ad(source)
    vocab = tmp_path / 'vocab.json'
    vocab.write_text(json.dumps({'<PAD>': 0, 'G1': 1, 'G2': 2, 'G3': 3, 'G4': 5}))
    settings = dict(vocab_size=6, bin_size=15, embed_dim=8, num_heads=2, num_layers=1)
    model = MaskedModel(**settings)
    checkpoint = tmp_path / 'model.ckpt'
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': settings,
                'pytorch-lightning_version': '2.5.0'}, checkpoint)
    return {'seed': 42, 'device': 'cpu', 'output_dir': str(tmp_path / 'out'),
            'data': {'input': str(source), 'vocab': str(vocab)},
            'model': {'checkpoint': str(checkpoint), 'embed_dim': 8, 'bin_size': 15},
            'zero_shot': {'max_genes': 4, 'batch_size': 3},
            'align': {'split_group': 'sample', 'reservoir_max_cells': 12,
                      'reservoir_pairs_per_cell': 4, 'epochs': 1,
                      'steps_per_epoch': 2, 'batch_pairs': 16}}


def test_full_distance_align_pipeline(cfg):
    prepare_data(cfg)
    encoder = load_encoder(cfg, torch.device('cpu'))
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    raw = encode(cfg)
    original = np.load(raw / 'data.npy').copy()
    checkpoint_before = open(cfg['model']['checkpoint'], 'rb').read()
    fitted = fit(cfg)
    assert (fitted / 'best_projector.pt').exists()
    aligned = project(cfg)
    for directory in [raw, aligned]:
        values = np.load(directory / 'data.npy')
        ptr = np.load(directory / 'indptr.npy')
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 2)).all()
        assert len(ptr) == 13 and ptr[-1] == len(values) == 47
    np.testing.assert_array_equal(np.load(raw / 'indices.npy'), np.load(aligned / 'indices.npy'))
    np.testing.assert_array_equal(original, np.load(raw / 'data.npy'))
    assert checkpoint_before == open(cfg['model']['checkpoint'], 'rb').read()
    projector = SharedResidualProjector(dim=8)
    output = projector(torch.randn(6, 8))
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(6))


def test_chunked_distance_and_missing_split(cfg):
    prepare_data(cfg)
    whole = encode(cfg)
    encode(cfg, 'part1', start=0, stop=6, prepare_tag='main')
    encode(cfg, 'part2', start=6, stop=12, prepare_tag='main')
    merged = merge_parts(cfg, 'zero-shot', 'merged', ['part2', 'part1'])
    for name in ['indptr', 'indices', 'data', 'pooled_embedding']:
        np.testing.assert_allclose(np.load(whole / f'{name}.npy'), np.load(merged / f'{name}.npy'), atol=1e-6)
    invalid = copy.deepcopy(cfg)
    invalid['align']['split_group'] = 'missing'
    with pytest.raises(ValueError, match='absent'):
        fit(invalid)
    invalid['align']['mode'] = 'joint_reconstruction'
    with pytest.raises(ValueError, match='shared_residual'):
        fit(invalid)

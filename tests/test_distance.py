import torch
import pytest

from scus.distance import cosine_distance, us_distance


def test_cosine_distance_known_vectors():
    u = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 0.]])
    s = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [1., 0.]])
    torch.testing.assert_close(cosine_distance(u, s), torch.tensor([0., 1., 2., 1.]))


def test_us_distance_masks_missing_pairs():
    hidden = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [1., 0.]]])
    distance, valid = us_distance(hidden, torch.tensor([[1, 2, 0, 3]]))
    assert valid.tolist() == [[True, False]]
    assert distance[0, 0] == 0 and torch.isnan(distance[0, 1])
    with pytest.raises(ValueError):
        us_distance(hidden[:, :3], torch.ones(1, 3))

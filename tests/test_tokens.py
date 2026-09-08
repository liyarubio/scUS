import numpy as np
from scus.data.tokens import rank_bin, select_genes, tokenize_pair


def test_rank_bin_reserves_zero_and_uses_valid_range():
    result = rank_bin(np.array([0.0, 3.0, 1.0, 2.0]), 15)
    assert result[0] == 0
    assert np.all((result[1:] >= 1) & (result[1:] <= 15))


def test_selection_is_deterministic_and_forced():
    mu = np.arange(30, dtype=float); ms = mu.copy()
    first = select_genes(mu, ms, 7, "cell-a", 42, [19])
    second = select_genes(mu, ms, 7, "cell-a", 42, [19])
    assert np.array_equal(first, second)
    assert 19 in first


def test_interleaved_token_contract():
    result = tokenize_pair(np.array([1., 2., 0.]), np.array([2., 3., 1.]), np.array([10, 11, 12]), 3)
    assert result[1].tolist() == [10, 10, 11, 11, 12, 12]
    assert result[3].tolist() == [0, 1, 0, 1, 0, 1]
    assert result[4].tolist() == [True, True, False]


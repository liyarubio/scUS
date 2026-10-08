from __future__ import annotations

import numpy as np


def perturb_pair_bins(
    value_bins: np.ndarray,
    pair_index: int,
    mode: str,
    maximum_bin: int = 15,
) -> np.ndarray:
    """Return a copy with exactly one interleaved Mu/Ms token pair changed.

    ``KO`` assigns both tokens to the lowest non-padding bin (1), while ``OE``
    assigns both to ``maximum_bin``. Re-encoding this array measures contextual
    representation sensitivity; it does not imply expression direction.
    """
    result = np.asarray(value_bins).copy()
    start = 2 * int(pair_index)
    if start < 0 or start + 1 >= len(result):
        raise IndexError(f"pair_index {pair_index} is outside {len(result) // 2} pairs")
    mode = mode.upper()
    if mode == "KO":
        result[start:start + 2] = 1
    elif mode == "OE":
        result[start:start + 2] = maximum_bin
    else:
        raise ValueError("mode must be 'KO' or 'OE'")
    return result

# Fast lookups on the cached component maps.
from __future__ import annotations

import numpy as np


def bilinear(xs, ys, Z, x, y, fill=np.nan):
    # Bilinear interpolation at one point, ``fill`` outside the grid.
    if not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
        return fill
    i = min(max(np.searchsorted(xs, x) - 1, 0), len(xs) - 2)
    j = min(max(np.searchsorted(ys, y) - 1, 0), len(ys) - 2)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    return (Z[i, j] * (1 - tx) * (1 - ty) + Z[i + 1, j] * tx * (1 - ty)
            + Z[i, j + 1] * (1 - tx) * ty + Z[i + 1, j + 1] * tx * ty)

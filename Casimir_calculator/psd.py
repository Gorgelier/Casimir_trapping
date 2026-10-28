from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal


def _eta(order: int, z: np.ndarray) -> np.ndarray:
    """Expansion coefficients for the Hu-Xu-Yan PSD poles."""
    am = 1.0 / 4.0
    a = 5.0 / 4.0
    bm = 3.0
    b = 15.0 + z / 4.0
    dbm = 0.0
    db = 1.0 / 4.0

    for m in range(3, order + 1):
        amm = am
        bmm = bm
        dbmm = dbm

        am = a
        bm = b
        dbm = db

        db = (2 * m + 1) * dbm + bmm / 4.0 + z * dbmm / 4.0
        a = (2 * m + 1) * am + z * amm / 4.0
        b = (2 * m + 1) * bm + z * bmm / 4.0

        am /= a
        bm /= a
        b /= a
        dbm /= a
        db /= a
        a = 1.0

    return a / (2.0 * db)


def psd(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return poles and weights for Pade spectrum decomposition.

    This is the same PSD construction used by `califorcia` and
    `casimir_cylinder`, based on Hu, Xu, and Yan, J. Chem. Phys. 133,
    101106 (2010).
    """
    # Distance scans reuse the same frequency order.  Cache only this
    # material-independent construction, never a physical result.  Copies
    # preserve the public writable-array API without exposing cached state.
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or order < 1:
        raise ValueError("PSD order must be a positive integer.")
    poles, weights = _cached_psd(int(order))
    return poles.copy(), weights.copy()


@lru_cache(maxsize=32)
def _cached_psd(order: int) -> tuple[np.ndarray, np.ndarray]:

    size = 2 * order
    diagonal = np.zeros(size)
    index = np.arange(1, size)
    off_diagonal = 1.0 / np.sqrt((2 * index + 1) * (2 * index + 3))
    eigenvalues = eigvalsh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, order - 1),
    )
    poles = np.atleast_1d(-2.0 / eigenvalues)
    weights = np.atleast_1d(_eta(size, -poles**2))
    poles.setflags(write=False)
    weights.setflags(write=False)
    return poles, weights


__all__ = ["psd"]

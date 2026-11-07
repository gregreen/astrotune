"""Quality-cut and model-selection helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .transforms import unmask


def finite_mask(t, cols: Iterable[str]) -> np.ndarray:
    """Boolean mask selecting rows where every listed column is finite."""
    mask = np.ones(len(t), dtype=bool)
    for col in cols:
        mask &= np.isfinite(unmask(t[col]))
    return mask


def snr_chisq_relation(snr, chisq, *, factors: Sequence[float] = (1, 2, 4, 8, 16)):
    """Fit ``chisq(snr) = a (1 + b snr^2)``.

    The coefficients are set by matching the median chi^2 in the 1st-2nd and
    98th-99th SNR percentile ranges, which is robust to the bulk of the
    distribution and to catastrophic outliers.

    Returns a callable ``f(snr)``; ``f.a`` and ``f.b`` hold the coefficients
    so that the fit can be plotted or recorded.
    """
    snr = np.asarray(snr, dtype="f8")
    chisq = np.asarray(chisq, dtype="f8")

    ok = np.isfinite(snr) & np.isfinite(chisq) & (snr > 0) & (chisq > 0)
    snr, chisq = snr[ok], chisq[ok]

    snr_pct = np.reshape(np.nanpercentile(snr, [1.0, 2.0, 98.0, 99.0]), (2, 2))

    x, y = [], []
    for snr_low, snr_high in snr_pct:
        sel = (snr > snr_low) & (snr < snr_high)
        x.append(np.median(snr[sel]))
        y.append(np.median(chisq[sel]))

    b = (y[1] - y[0]) / (y[0] * x[1] ** 2 - y[1] * x[0] ** 2)
    a = y[0] / (1 + b * x[0] ** 2)

    def chisq_of_snr(snr_new):
        return a * (1 + b * np.asarray(snr_new, dtype="f8") ** 2)

    chisq_of_snr.a = a
    chisq_of_snr.b = b
    chisq_of_snr.snr_percentiles = snr_pct
    return chisq_of_snr


def snr_chisq_mask(snr, chisq, *, factor: float = 1.0) -> tuple[np.ndarray, callable]:
    """Convenience wrapper: select ``chisq < factor * f(factor * snr)``.

    Returns the mask and the fitted relation.
    """
    f = snr_chisq_relation(snr, chisq)
    mask = np.asarray(chisq) < factor * f(factor * np.asarray(snr))
    return mask, f

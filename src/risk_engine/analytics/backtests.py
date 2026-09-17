"""Likelihood-ratio VaR diagnostics; non-rejection does not establish model validity."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2


@dataclass(frozen=True)
class KupiecResult:
    n_obs: int
    n_breaches: int
    alpha: float
    lr_stat: float
    p_value: float


def _ll(ones, zeros, p):
    return float(xlogy(ones, p) + xlogy(zeros, 1 - p))


def kupiec_pof_test(n_breaches: int, n_obs: int, alpha: float) -> KupiecResult:
    if not isinstance(n_obs, (int, np.integer)) or n_obs <= 0:
        raise ValueError("n_obs must be a positive integer.")
    if not isinstance(n_breaches, (int, np.integer)) or not 0 <= n_breaches <= n_obs:
        raise ValueError("n_breaches must be an integer between 0 and n_obs.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    x, n = n_breaches, n_obs
    lr = max(0., 2 * (_ll(x, n-x, x/n) - _ll(x, n-x, 1-alpha)))
    return KupiecResult(n, x, alpha, lr, float(chi2.sf(lr, 1)))


@dataclass(frozen=True)
class ChristoffersenIndependenceResult:
    n_00: int
    n_01: int
    n_10: int
    n_11: int
    lr_stat: float | None
    p_value: float | None


def breach_transitions(breaches):
    b = np.asarray(breaches)
    if b.ndim != 1 or not np.isin(b, [0, 1]).all():
        raise ValueError("Breaches must be a one-dimensional binary sequence.")
    return tuple(int(np.sum((b[:-1] == i) & (b[1:] == j))) for i, j in [(0,0), (0,1), (1,0), (1,1)])


def christoffersen_independence_test(breaches):
    if len(breaches) < 2:
        raise ValueError("Need at least two observations.")
    a, b, c, d = breach_transitions(breaches)
    # With no observed transitions out of one state, independence is not identifiable.
    if a+b == 0 or c+d == 0:
        return ChristoffersenIndependenceResult(a, b, c, d, None, None)
    p = (b+d)/(a+b+c+d)
    lr = max(0., 2 * (_ll(b, a, b/(a+b)) + _ll(d, c, d/(c+d)) - _ll(b+d, a+c, p)))
    return ChristoffersenIndependenceResult(a, b, c, d, lr, float(chi2.sf(lr, 1)))


@dataclass(frozen=True)
class ConditionalCoverageResult:
    lr_stat: float | None
    p_value: float | None


def conditional_coverage_test(kupiec_lr, christoffersen_lr):
    if christoffersen_lr is None:
        return ConditionalCoverageResult(None, None)
    lr = kupiec_lr + christoffersen_lr
    return ConditionalCoverageResult(lr, float(chi2.sf(lr, 2)))


def align_forecasts(pnl: pd.Series, **forecasts: pd.Series) -> pd.DataFrame:
    """Drop unavailable forecasts before creating breach indicators.

    Only leading/trailing unavailability is allowed: gaps within evaluation would
    make an independence test incorrectly treat non-adjacent observations as adjacent.
    """
    if not forecasts or pnl.empty:
        raise ValueError("Supply returns and forecasts.")
    series = [pnl, *forecasts.values()]
    if any(not s.index.is_unique or not s.index.is_monotonic_increasing for s in series):
        raise ValueError("Indexes must be unique and chronological.")
    frame = pd.DataFrame({"return": pnl, **{k: v.reindex(pnl.index) for k, v in forecasts.items()}})
    valid = frame.notna().all(axis=1)
    positions = np.flatnonzero(valid.to_numpy())
    if len(positions) < 2:
        raise ValueError("Need at least two common valid forecasts.")
    if not valid.iloc[positions[0]:positions[-1]+1].all():
        raise ValueError("Internal missing forecasts would break breach adjacency.")
    frame = frame.loc[valid]
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Evaluation values must be finite.")
    return frame

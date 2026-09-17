"""VaR is the signed loss quantile: -quantile(return, 1-alpha)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def validate_sample(pnl: pd.Series, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if pnl.empty or not np.isfinite(pnl.to_numpy()).all():
        raise ValueError("Returns must be non-empty and finite.")


def portfolio_pnl(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Exact simple return for a portfolio rebalanced to these weights each observation.

    Inputs must be simple asset returns. Output is fractional return, not currency P&L.
    This educational implementation requires a fully invested, long-only portfolio.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or returns.shape[1] != len(weights):
        raise ValueError("weights length must match number of assets.")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must be finite, non-negative and sum to one.")
    if returns.empty or not np.isfinite(returns.to_numpy()).all() or (returns < -1).any().any():
        raise ValueError("Simple returns must be finite and at least -1.")
    return pd.Series(returns.to_numpy() @ weights, index=returns.index, name="portfolio_return")


def var_historical(pnl: pd.Series, alpha: float) -> float:
    validate_sample(pnl, alpha)
    return float(-np.quantile(pnl.to_numpy(), 1 - alpha))


def es_historical(pnl: pd.Series, alpha: float) -> float:
    validate_sample(pnl, alpha)
    q = np.quantile(pnl.to_numpy(), 1 - alpha)
    return float(-pnl[pnl <= q].mean())


def var_parametric_gaussian(pnl: pd.Series, alpha: float) -> float:
    validate_sample(pnl, alpha)
    if len(pnl) < 2:
        raise ValueError("Gaussian VaR needs at least two observations.")
    return float(-(pnl.mean() + pnl.std(ddof=1) * norm.ppf(1 - alpha)))


def rolling_var_historical(pnl: pd.Series, alpha: float, window: int) -> pd.Series:
    """Forecast at t uses the window observations strictly before t."""
    validate_sample(pnl, alpha)
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least two.")
    return (-pnl.shift(1).rolling(window, min_periods=window).quantile(1 - alpha)).rename("rolling_var")

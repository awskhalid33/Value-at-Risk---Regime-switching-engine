"""Validated date-indexed price input; missing observations are never filled here."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("Prices must contain observations and assets.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices need a DatetimeIndex.")
    if prices.index.hasnans or not prices.index.is_unique:
        raise ValueError("Price dates must be non-missing and unique.")
    if not prices.columns.is_unique:
        raise ValueError("Asset names must be unique.")
    prices = prices.apply(pd.to_numeric, errors="raise").sort_index()
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite, non-missing and strictly positive.")
    return prices


def load_prices_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dates = [col for col in df.columns if col.lower() == "date"]
    if len(dates) != 1:
        raise ValueError("CSV must contain exactly one date column (date, Date or DATE).")
    df = df.rename(columns={dates[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return validate_prices(df.set_index("date"))

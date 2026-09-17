"""Asset return transformations. Portfolio aggregation uses simple returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from risk_engine.io.prices import validate_prices


def to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return validate_prices(prices).pct_change(fill_method=None).iloc[1:]


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(validate_prices(prices)).diff().iloc[1:]

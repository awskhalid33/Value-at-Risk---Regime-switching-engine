"""Download local-currency index levels; use common observation dates."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
from risk_engine.io.prices import validate_prices


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Supply distinct tickers.")
    data = yf.download(tickers=tickers, start=start, end=end, interval="1d",
                       auto_adjust=True, progress=False, group_by="column")
    if data.empty:
        raise ValueError("Yahoo returned no prices; check dates, tickers and network access.")
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.reindex(columns=tickers)
    if prices.isna().all().any():
        raise ValueError("One or more tickers returned no prices.")
    # No forward filling: avoid manufacturing zero returns on local holidays.
    prices = prices.dropna(how="any")
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return validate_prices(prices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["^FTSE", "^GSPC", "^STOXX50E"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive end date")
    parser.add_argument("--output", type=Path, default=Path("data/market_prices.csv"))
    args = parser.parse_args()
    prices = fetch_prices(args.tickers, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.output, index_label="date")
    print(f"Saved {len(prices)} common-date observations, {prices.shape[1]} assets to {args.output}")


if __name__ == "__main__":
    main()

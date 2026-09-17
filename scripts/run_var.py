"""Reproducible fixed-training, sequential-holdout VaR experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from risk_engine.io.prices import load_prices_csv
from risk_engine.analytics.returns import to_simple_returns
from risk_engine.analytics.var_es import (portfolio_pnl, var_historical, es_historical,
    var_parametric_gaussian, rolling_var_historical)
from risk_engine.analytics.regimes import holdout_hmm_forecasts, hmm_params
from risk_engine.analytics.backtests import (align_forecasts, kupiec_pof_test,
    christoffersen_independence_test, conditional_coverage_test)


def backtest(breaches, alpha):
    n, x = len(breaches), int(breaches.sum())
    kup = kupiec_pof_test(x, n, alpha)
    ind = christoffersen_independence_test(breaches.tolist())
    cc = conditional_coverage_test(kup.lr_stat, ind.lr_stat)
    return dict(observations=n, breaches=x, breach_rate=x/n, expected_breaches=n*(1-alpha),
                kupiec=asdict(kup), independence=asdict(ind), conditional_coverage=asdict(cc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=Path("data/market_prices.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/latest"))
    parser.add_argument("--train-size", type=int, default=750)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=.99)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["bisection", "mc"], default="bisection")
    parser.add_argument("--n-sims", type=int, default=50_000)
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.window > args.train_size:
        parser.error("--window must not exceed --train-size.")
    prices = load_prices_csv(args.prices)
    weights = np.array(args.weights) if args.weights else np.ones(prices.shape[1])/prices.shape[1]
    pnl = portfolio_pnl(to_simple_returns(prices), weights)
    hmm_var, probs, model = holdout_hmm_forecasts(pnl, args.train_size, args.alpha,
        args.states, args.seed, args.method, args.n_sims)
    rolling = rolling_var_historical(pnl, args.alpha, args.window)
    frame = align_forecasts(pnl, rolling_var=rolling, hmm_var=hmm_var)
    tests = {}
    for name in ["rolling_var", "hmm_var"]:
        breaches = (frame["return"] < -frame[name]).astype(int)
        tests[name] = backtest(breaches, args.alpha)

        # VaR is a loss magnitude; the forecast return quantile is -VaR.
        tau = 1.0 - args.alpha
        error = frame["return"] + frame[name]
        quantile_loss = error * (tau - (error < 0).astype(int))

        tests[name]["average_var"] = float(frame[name].mean())
        tests[name]["mean_quantile_loss"] = float(quantile_loss.mean())

        frame[name + "_breach"] = breaches
        frame[name + "_quantile_loss"] = quantile_loss
    mus, sigmas = hmm_params(model)
    training = pnl.iloc[:args.train_size]
    summary = dict(
        design="Fixed initial HMM training; sequential one-observation-ahead holdout; no refitting",
        input_sha256=hashlib.sha256(args.prices.read_bytes()).hexdigest(),
        assets=list(prices.columns), weights=weights.tolist(), return_observations=len(pnl),
        training=dict(observations=len(training), start=str(training.index[0]), end=str(training.index[-1])),
        evaluation=dict(start=str(frame.index[0]), end=str(frame.index[-1]), observations=len(frame)),
        configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        training_descriptives=dict(historical_var=var_historical(training,args.alpha),
            historical_es=es_historical(training,args.alpha), gaussian_var=var_parametric_gaussian(training,args.alpha)),
        hmm=dict(means=mus.tolist(), sigmas=sigmas.tolist(), transition_matrix=model.transmat_.tolist(),
                 start_probabilities=model.startprob_.tolist(), fit_diagnostics=model.fit_diagnostics_),
        backtests=tests,
        environment=dict(python=platform.python_version(), packages={p:version(p) for p in
            ["numpy","pandas","scipy","matplotlib","hmmlearn","scikit-learn","yfinance"]}),
        limitations=["Local-currency equity-index proxy; no FX, trading costs or dividend reinvestment model",
          "Common-date intervals can span holidays; market closing times differ",
          "Fixed HMM parameters may become stale; no superiority claim follows from non-rejection",
          "Sparse tail events limit asymptotic tests; null test values mean independence was not identifiable"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    frame.join(probs.add_prefix("forecast_")).to_csv(args.output/"forecasts.csv", index_label="date")
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(frame.index, frame["return"], label="Portfolio simple return", linewidth=.7)
        for name in tests:
            ax.plot(frame.index, -frame[name], label=f"{args.alpha:.0%} {name}", linewidth=.9)
        ax.set(title="Holdout returns and forecast loss thresholds", ylabel="Fractional return")
        ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(args.output/"var.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10,4))
        for i, col in enumerate(probs):
            ax.plot(probs.index, probs[col], label=f"State {i}: sigma={sigmas[i]:.2%}")
        ax.set(title="Holdout forecast regime probabilities (states ordered by volatility)", ylabel="Probability", ylim=(0,1))
        ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(args.output/"regimes.png", dpi=150); plt.close(fig)
    print(json.dumps(tests, indent=2))
    print(f"Saved reproducibility metadata and forecasts to {args.output}")


if __name__ == "__main__":
    main()

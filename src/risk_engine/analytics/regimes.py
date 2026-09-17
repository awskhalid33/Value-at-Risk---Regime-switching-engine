"""Univariate Gaussian HMM and signed Gaussian-mixture loss quantiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from risk_engine.analytics.var_es import validate_sample


def fit_gaussian_hmm(pnl: pd.Series, n_states: int, random_state: int = 42,
                     n_iter: int = 1000, n_starts: int = 3) -> GaussianHMM:
    """Fit on caller-supplied training data only; select converged fit by training score.

    Fit in percentage units for numerical conditioning, then convert parameters back
    to fractional-return units. States are consistently ordered by increasing sigma.
    """
    validate_sample(pnl, .99)
    if n_states < 1 or n_starts < 1 or n_iter < 2 or len(pnl) < max(20, 10 * n_states):
        raise ValueError("Insufficient training observations or invalid fit configuration.")
    x = pnl.to_numpy().reshape(-1, 1) * 100
    if np.std(x) == 0:
        raise ValueError("Cannot fit regimes to constant returns.")
    candidates = []
    diagnostics = []
    for seed in range(random_state, random_state + n_starts):
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter,
                            random_state=seed, tol=1e-4, min_covar=1e-4)
        try:
            model.fit(x)
            history = list(model.monitor_.history)
            delta = history[-1] - history[-2] if len(history) >= 2 else np.inf
            score = float(model.score(x))
            # hmmlearn's monitor also calls reaching n_iter 'converged'; check improvement.
            converged = bool(np.isfinite(score) and -1e-6 <= delta < model.tol)
            diagnostics.append(dict(seed=seed, converged=converged, iterations=model.monitor_.iter,
                                    log_likelihood=score, last_improvement=float(delta)))
            if converged:
                candidates.append((score, model))
        except (ValueError, FloatingPointError) as exc:
            diagnostics.append(dict(seed=seed, converged=False, error=str(exc)))
    if not candidates:
        raise RuntimeError(f"No HMM fit converged; revise training/configuration. Diagnostics: {diagnostics}")
    _, model = max(candidates, key=lambda pair: pair[0])
    means = model.means_.reshape(-1) / 100
    variances = model.covars_.reshape(-1) / 100**2
    order = np.argsort(variances)
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.means_ = means[order, None]
    model.covars_ = variances[order, None]
    model.fit_diagnostics_ = diagnostics
    return model


def hmm_params(model: GaussianHMM) -> tuple[np.ndarray, np.ndarray]:
    return model.means_.reshape(-1), np.sqrt(model.covars_.reshape(-1))


def hmm_posterior_probabilities(model: GaussianHMM, pnl: pd.Series) -> pd.DataFrame:
    """Smoothed probabilities use future observations. For retrospective analysis only."""
    validate_sample(pnl, .99)
    probs = model.predict_proba(pnl.to_numpy().reshape(-1, 1))
    return pd.DataFrame(probs, index=pnl.index, columns=[f"regime_{k}" for k in range(probs.shape[1])])


def hmm_filtered_probabilities(model: GaussianHMM, pnl: pd.Series) -> pd.DataFrame:
    """Causal forward filter conditional on fixed parameters; training must be separate.

    Log-domain recursion avoids emission underflow and uses public fitted parameters.
    """
    validate_sample(pnl, .99)
    mus, sigmas = hmm_params(model)
    log_emission = norm.logpdf(pnl.to_numpy()[:, None], mus, sigmas)
    probs = np.empty_like(log_emission)
    prior = model.startprob_
    for t, emission in enumerate(log_emission):
        with np.errstate(divide="ignore"):
            log_weight = np.log(prior) + emission
        normalizer = logsumexp(log_weight)
        if not np.isfinite(normalizer):
            raise ValueError("Non-finite filter normalizer; inspect returns and HMM parameters.")
        probs[t] = np.exp(log_weight - normalizer)
        prior = probs[t] @ model.transmat_
    return pd.DataFrame(probs, index=pnl.index, columns=[f"regime_{k}" for k in range(len(mus))])


def hmm_forecast_probabilities(model: GaussianHMM, filtered_probs: pd.DataFrame) -> pd.DataFrame:
    """Row dated t describes S_(t+1); shift once before comparing with returns."""
    values = filtered_probs.to_numpy() @ model.transmat_
    return pd.DataFrame(values / values.sum(axis=1, keepdims=True),
                        index=filtered_probs.index, columns=filtered_probs.columns)


def _mixture_inputs(weights, mus, sigmas, alpha):
    w, m, s = [np.asarray(x, dtype=float) for x in (weights, mus, sigmas)]
    if any(x.ndim != 1 for x in (w, m, s)) or not (len(w) == len(m) == len(s) > 0):
        raise ValueError("Mixture parameters must be non-empty vectors of equal length.")
    if not all(np.isfinite(x).all() for x in (w, m, s)):
        raise ValueError("Mixture parameters must be finite.")
    if (w < 0).any() or w.sum() <= 0 or (s <= 0).any() or not 0 < alpha < 1:
        raise ValueError("Invalid weights, sigmas or alpha.")
    return w / w.sum(), m, s


def mixture_var_bisection(weights, mus, sigmas, alpha, tol=1e-10, max_iter=200) -> float:
    """Deterministic bracketed root solve (Brent) for the mixture quantile.

    Legacy function name is retained. Component quantiles bracket the mixture quantile.
    """
    w, m, s = _mixture_inputs(weights, mus, sigmas, alpha)
    q = m + s * norm.ppf(1 - alpha)
    lo, hi = float(q.min()), float(q.max())
    if lo == hi:
        return -lo
    return float(-brentq(lambda x: np.dot(w, norm.cdf((x - m) / s)) - (1 - alpha),
                        lo, hi, xtol=tol, maxiter=max_iter))


def mixture_var_monte_carlo(weights, mus, sigmas, alpha, n_sims=50_000, seed=123) -> float:
    w, m, s = _mixture_inputs(weights, mus, sigmas, alpha)
    if not isinstance(n_sims, int) or n_sims < 1:
        raise ValueError("n_sims must be a positive integer.")
    rng = np.random.default_rng(seed)
    states = rng.choice(len(w), size=n_sims, p=w)
    return float(-np.quantile(rng.normal(m[states], s[states]), 1 - alpha))


def hmm_var_series(posterior_probs, mus, sigmas, alpha, method="bisection", n_sims=50_000, seed=123):
    if method not in ("mc", "bisection"):
        raise ValueError("method must be mc or bisection.")
    values = []
    for i, row in enumerate(posterior_probs.to_numpy()):
        if method == "mc":
            value = mixture_var_monte_carlo(row, mus, sigmas, alpha, n_sims, seed + i)
        else:
            value = mixture_var_bisection(row, mus, sigmas, alpha)
        values.append(value)
    return pd.Series(values, index=posterior_probs.index, name="hmm_var")


def holdout_hmm_forecasts(pnl, train_size, alpha=.99, n_states=2, seed=42,
                          method="bisection", n_sims=50_000):
    """Fit once on the initial training block; forecast all subsequent observations.

    Evaluation return at t affects forecasts from t+1 onwards, never the forecast at t.
    No holdout observation participates in parameter fitting or restart selection.
    """
    validate_sample(pnl, alpha)
    if not pnl.index.is_unique or not pnl.index.is_monotonic_increasing:
        raise ValueError("Returns must have unique, chronological indexes.")
    if not isinstance(train_size, int) or not 20 <= train_size < len(pnl):
        raise ValueError("train_size must leave both sufficient training data and a holdout.")
    model = fit_gaussian_hmm(pnl.iloc[:train_size], n_states, random_state=seed)
    filtered = hmm_filtered_probabilities(model, pnl)
    probs = hmm_forecast_probabilities(model, filtered).shift(1).iloc[train_size:]
    mus, sigmas = hmm_params(model)
    var = hmm_var_series(probs, mus, sigmas, alpha, method, n_sims, seed)
    return var, probs, model

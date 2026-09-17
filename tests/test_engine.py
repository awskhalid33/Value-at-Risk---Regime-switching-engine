import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk_engine.io.prices import load_prices_csv
from risk_engine.analytics.returns import to_simple_returns
from risk_engine.analytics.var_es import portfolio_pnl, rolling_var_historical
from risk_engine.analytics.backtests import (align_forecasts, kupiec_pof_test,
    christoffersen_independence_test, breach_transitions)
from risk_engine.analytics.regimes import (holdout_hmm_forecasts, mixture_var_bisection,
    mixture_var_monte_carlo, hmm_filtered_probabilities, hmm_posterior_probabilities)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("column", ["date", "Date", "DATE"])
def test_date_column_roundtrip(tmp_path, column):
    path = tmp_path / "prices.csv"
    path.write_text(f"{column},A\n2024-01-02,110\n2024-01-01,100\n")
    prices = load_prices_csv(path)
    assert prices.index.is_monotonic_increasing
    assert to_simple_returns(prices).iloc[0, 0] == pytest.approx(.1)


@pytest.mark.parametrize("body", [
    "missing,A\n2024-01-01,100\n", "date,A\n2024-01-01,0\n",
    "date,A\n2024-01-01,100\n2024-01-01,101\n", "date,A\n2024-01-01,\n"])
def test_invalid_prices_rejected(tmp_path, body):
    path = tmp_path / "prices.csv"; path.write_text(body)
    with pytest.raises(ValueError): load_prices_csv(path)


def test_exact_portfolio_return():
    prices = pd.DataFrame([[100,100],[110,90]], index=pd.date_range("2024-01-01", periods=2))
    assert portfolio_pnl(to_simple_returns(prices), np.array([.5,.5])).iloc[0] == pytest.approx(0, abs=1e-15)
    with pytest.raises(ValueError): portfolio_pnl(to_simple_returns(prices), np.array([.5,.6]))


def test_warmup_excluded_before_breaches():
    pnl = pd.Series(np.linspace(-.03,.03,300))
    rolling = rolling_var_historical(pnl,.99,250)
    table = align_forecasts(pnl, rolling=rolling, hmm=pd.Series(.02,index=pnl.index))
    assert len(table) == 50
    assert table.index[0] == 250
    changed = pnl.copy(); changed.iloc[260:] = -.5
    other = rolling_var_historical(changed,.99,250)
    pd.testing.assert_series_equal(rolling.iloc[:261], other.iloc[:261])


def test_missing_internal_forecast_rejected():
    with pytest.raises(ValueError, match="Internal"):
        align_forecasts(pd.Series([.01,.02,.03]), var=pd.Series([.02,np.nan,.02]))


def test_reference_statistics_and_degenerate_cases():
    result = kupiec_pof_test(17,1546,.99)
    assert result.lr_stat == pytest.approx(.1500982689)
    assert result.p_value == pytest.approx(.6984414667)
    assert np.isfinite(kupiec_pof_test(0,100,.99).lr_stat)
    assert christoffersen_independence_test([0]*100).p_value is None
    ind = christoffersen_independence_test([0,1]*20 + [0]*1506)
    assert ind.lr_stat == pytest.approx(.5246052029)
    assert ind.p_value == pytest.approx(.4688838136)
    assert breach_transitions([0,0,1,1,0]) == (1,1,1,1)
    with pytest.raises(ValueError): breach_transitions([0,2,1])


def test_mixture_quantile_against_normal_and_monte_carlo():
    exact = -.01 + .02*norm.ppf(.99)
    assert mixture_var_bisection([1],[.01],[.02],.99) == pytest.approx(exact)
    w,m,s = [.8,.2], [.001,-.01], [.01,.035]
    q = mixture_var_bisection(w,m,s,.99)
    assert np.dot(w, norm.cdf((-q-np.array(m))/s)) == pytest.approx(.01,abs=1e-9)
    mc = mixture_var_monte_carlo(w,np.array(m),np.array(s),.99,n_sims=300_000,seed=123)
    assert mc == pytest.approx(q,abs=.0015)
    with pytest.raises(ValueError): mixture_var_bisection([1],[0],[0],.99)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(2026)
    state = 0; values = []
    for _ in range(520):
        if rng.random() < (.04 if state == 0 else .12): state = 1-state
        values.append(rng.normal([.0005,-.002][state],[.007,.03][state]))
    return pd.Series(values,index=pd.bdate_range("2020-01-01",periods=len(values)))


def test_hmm_future_invariance_and_boundary(synthetic):
    pnl = synthetic
    var, probs, model = holdout_hmm_forecasts(pnl,300)
    changed = pnl.copy(); changed.iloc[400:] = -.12
    var2, probs2, model2 = holdout_hmm_forecasts(changed,300)
    np.testing.assert_allclose(model.means_,model2.means_)
    pd.testing.assert_series_equal(var.loc[:pnl.index[400]],var2.loc[:pnl.index[400]])
    pd.testing.assert_frame_equal(probs.loc[:pnl.index[400]],probs2.loc[:pnl.index[400]])
    filtered = hmm_filtered_probabilities(model,pnl.iloc[:300])
    np.testing.assert_allclose(probs.iloc[0],filtered.iloc[-1].to_numpy() @ model.transmat_)
    # At the final observation, smoothed and filtered probabilities agree.
    smooth = hmm_posterior_probabilities(model,pnl.iloc[:300])
    np.testing.assert_allclose(filtered.iloc[-1],smooth.iloc[-1],atol=1e-10)
    assert np.all(np.diff(model.covars_.reshape(-1)) >= 0)
    np.testing.assert_allclose(probs.sum(axis=1),1)
    assert len(var) == 220 and var.index[0] == pnl.index[300]


def test_download_cleaning_and_cli_entry(monkeypatch,tmp_path):
    spec = importlib.util.spec_from_file_location("fetch",ROOT/"scripts/fetch_prices.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    dates = pd.date_range("2024-01-01",periods=3)
    raw = pd.DataFrame([[100,200],[101,np.nan],[102,202]],index=dates,
        columns=pd.MultiIndex.from_product([["Close"],["A","B"]]))
    monkeypatch.setattr(module.yf,"download",lambda **kwargs: raw)
    target = tmp_path/"nested"/"prices.csv"
    monkeypatch.setattr(sys,"argv",["fetch_prices.py","--tickers","B","A","--output",str(target)])
    module.main()
    prices=load_prices_csv(target)
    assert list(prices.columns) == ["B","A"] and len(prices) == 2
    result=subprocess.run([sys.executable,str(ROOT/"scripts/fetch_prices.py"),"--help"],capture_output=True,text=True)
    assert result.returncode == 0 and "--tickers" in result.stdout


@pytest.mark.parametrize("method", ["bisection", "mc"])
def test_cli_end_to_end(tmp_path,synthetic,method):
    prices=pd.DataFrame({"A":100*(1+synthetic).cumprod(),"B":100*(1+synthetic*.8).cumprod()})
    csv=tmp_path/"prices.csv"; prices.to_csv(csv,index_label="date")
    output=tmp_path/"result"
    result=subprocess.run([sys.executable,str(ROOT/"scripts/run_var.py"),"--prices",str(csv),
        "--train-size","300","--method",method,"--n-sims","2000","--output",str(output)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    summary=json.loads((output/"summary.json").read_text())
    assert summary["evaluation"]["observations"] == 219
    forecasts=pd.read_csv(output/"forecasts.csv")
    assert forecasts.notna().all().all()
    assert len(forecasts) == summary["backtests"]["hmm_var"]["observations"]
    assert (output/"var.png").stat().st_size > 1000
    assert (output/"regimes.png").stat().st_size > 1000

# Validation notes

This change replaces the original full-sample evaluation with a fixed-training chronological holdout experiment. The previous performance table is withdrawn; it must not be used as evidence that the HMM outperforms rolling historical VaR.

## Checks performed

- Installed the project using `python -m pip install -e ".[dev]"` in a virtual environment (Python 3.12).
- Executed the regression suite, including CSV roundtrips and rejection of invalid prices; exact portfolio simple-return arithmetic; exclusion of warmup forecasts; refusal to bridge missing evaluation dates; reference likelihood-ratio values; and Gaussian-mixture quantile checks.
- Changed returns from a chosen holdout date onwards and confirmed that the forecast for that date and all earlier dates remained identical. Fitted HMM parameters also remained identical.
- Compared the first holdout forecast probabilities with the final training filter projected through the transition matrix.
- Ran the command-line pipeline on synthetic prices with both deterministic and Monte Carlo quantiles, checking saved JSON, CSV and PNG outputs. Inspected the generated VaR chart.
- Tested downloader cleaning and directory creation with a controlled Yahoo-shaped response, and checked the executable `--help` entry point.

These synthetic checks validate software behaviour. They do not establish financial forecasting performance. The synthetic generator is in `tests/test_engine.py` and uses a fixed seed.

## Subsequent market-data evaluation

The live Yahoo download failed during the original implementation checks. A later market-data evaluation was completed and its reports are now committed under `docs/results/`; see the README for configurations, results and limitations. The earlier statement that no replacement performance table existed is therefore superseded.

The retained summaries cover 95% and 99% VaR with historical windows of 250 and 500 days. The reported HMM forecasts use deterministic mixture quantiles. Monte Carlo quantiles remain an optional implementation method, not the method behind those reported figures.

The results do not establish universal HMM superiority. In particular, the 95% HMM calibration tests reject, and 99% inference is limited by few breaches. Changing the historical window does not produce an independent HMM experiment. Preserve the input CSV, its SHA-256, configuration, dependency versions and forecast table when reproducing results. Further tuning requires a separate final test period.

## Automated checks

The GitHub Actions workflow installs the package and runs the suite on Python 3.10 and 3.12. It does not depend on Yahoo availability. Its status should be checked on the pull request; local execution alone is not evidence that the hosted workflow passed.


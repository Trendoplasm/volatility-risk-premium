# Decomposing the Volatility Risk Premium

Option-implied volatility is persistently higher than the volatility underlyings go on to
deliver. This study measures that gap on real market data, then simulates the delta-hedged
straddle that harvests it and decomposes the profit into the Greeks that produced it.

[![CI](https://github.com/Trendoplasm/volatility-risk-premium/actions/workflows/ci.yml/badge.svg)](https://github.com/Trendoplasm/volatility-risk-premium/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-1f5082)](https://mypy-lang.org/)

## Read this first: what is and is not measured

| Component | Status |
|---|---|
| Implied volatility | **Observed.** Cboe VIX9D / VIX / VIX3M / VIX1Y and VXAPL / VXAZN daily history |
| Underlying returns | **Observed.** Daily closes for the S&P 500, Apple, and Amazon |
| Option prices | **Modelled.** Black-Scholes marks from the observed inputs, not historical quotes |
| Securities measured | **3 of 21.** The other 18 have no free implied-volatility history |
| Capital denominator | **A research normalisation**, not broker margin |

The implied and realised legs are both real market data, so the measured premium is a genuine
empirical result. The *trade* built on top of it is a simulation: it marks options with
Black-Scholes rather than with historical bid and ask, so its profits exclude the spread of a real
option chain beyond a modelled entry half-spread. Returns below are therefore an upper bound on
what was achievable, and the report is explicit about the gates a live deployment would have to
clear. Every exported table repeats this note.

## Headline results

Study period **7 January 2011 to 30 June 2026** — 11,602 matched observations at the one-month
horizon and 555 non-overlapping delta-hedged trades.

### The premium exists, on real data

| Security | Horizon | Mean implied vol | Mean realised vol | Premium | Positive |
|---|---:|---:|---:|---:|---:|
| S&P 500 | 21d | 18.2% | 14.6% | **+3.61 pts** | 84.4% |
| Apple | 21d | 29.8% | 25.9% | **+3.89 pts** | 75.0% |
| Amazon | 21d | 34.3% | 30.2% | **+4.03 pts** | 74.9% |
| Pooled | 21d | 27.4% | 23.6% | **+3.84 pts** | 78.1% |
| Pooled | 42d | 29.7% | 24.1% | **+5.54 pts** | 82.4% |
| Pooled | 63d | 30.3% | 24.5% | **+5.89 pts** | 80.8% |

The block-bootstrap 95% interval for the pooled one-month premium is **+3.68 to +3.99 volatility
points** — comfortably clear of zero. The S&P 500 figure of about 3.6 points is the study's
external sanity check: it reproduces a widely replicated result, from scratch, on independently
downloaded data.

### Where the profit comes from

Averaged over the 555 trades, as a share of the premium paid for the straddle:

| Component | Share of premium | Reading |
|---|---:|---|
| Theta | **−57.1%** | The buyer pays time decay almost every single day |
| Gamma | **+40.5%** | Realised movement gives most of it back |
| Vega | **+3.4%** | Implied volatility drifts slightly in the buyer's favour |
| Residual | **−2.0%** | Higher-order convexity the first three terms miss |
| **Net (long straddle)** | **−15.2%** | What the seller collects |

This is the economic mechanism, not a restatement of the premium. Theta arrives steadily and
gamma losses arrive in bursts, which is why the strategy shows a high win rate and a long left
tail at the same time. The four components sum to the total by construction, and the exported
table carries a `check` column reporting the floating-point residual of that identity, so it can
be verified rather than trusted.

### The trade, and its tail

| Group | Trades | Mean return per trade | Annualised | Win rate | Worst trade | Worst 5% average |
|---|---:|---:|---:|---:|---:|---:|
| S&P 500 | 185 | +4.45% | +53.4% | 84.3% | −19.8% | −11.0% |
| Apple | 185 | +3.80% | +45.7% | 68.6% | −42.7% | −21.2% |
| Amazon | 185 | +3.92% | +47.1% | 69.7% | −41.6% | −26.8% |
| Pooled | 555 | +4.06% | +48.7% | 74.2% | −42.7% | −21.6% |

Those annualised figures are large and should be read with the modelling caveats above firmly in
mind. The tail columns are the point: a single trade lost 42.7% of its capital proxy, and the
worst 5% of trades averaged −21.6% against a +4.06% mean. Transaction costs consume about 1.7% of
the entry premium.

### Selling into high volatility paid more — and risked more

| Regime | Trades | Mean entry VIX | Mean return | Win rate | Worst 5% average |
|---|---:|---:|---:|---:|---:|
| Low | 186 | 12.7% | +2.17% | 69.4% | −22.6% |
| Middle | 186 | 16.6% | +4.12% | 73.1% | −14.7% |
| High | 183 | 25.4% | +5.92% | 80.3% | −24.4% |

### Robustness: one dimension moved at a time

| Horizon | Moneyness | Hedge | Trades | Mean return | Win rate | Worst 5% average |
|---:|---:|---:|---:|---:|---:|---:|
| 21d | 1.00 | daily | 555 | +4.06% | 74.2% | −21.6% |
| 21d | 0.95 | daily | 555 | +0.45% | 57.5% | −24.9% |
| 21d | 1.05 | daily | 555 | +4.38% | 75.9% | −18.3% |
| 21d | 1.00 | 5-day | 555 | +4.01% | 69.9% | **−32.2%** |
| 42d | 1.00 | daily | 276 | +8.71% | 80.4% | −24.2% |
| 63d | 1.00 | daily | 183 | +15.58% | 90.7% | −21.2% |

Two findings worth drawing out. Hedging weekly instead of daily barely changed the average return
(+4.01% against +4.06%) but made the tail half again as bad (−32.2% against −21.6%): **hedging
frequency buys risk control, not return.** And struck 5% below the forward, the same trade earned
almost nothing — the premium is concentrated at and above the money.

### A negative result: the obvious entry filter does not work

The intuitive rule is to sell only when implied volatility looks rich against what the underlying
has recently delivered. Thresholds were selected on 2011–2018 alone and then applied forward:

| Rule | In sample (to 2018) | Out of sample (2019+) |
|---|---:|---:|
| Take every trade | +3.43% | +4.73% |
| Filter on implied-minus-trailing-realised | +3.33% | +4.65% |

The filter is no better than doing nothing in either period. Worse, the relationship *reverses
sign* between them: in sample, tightening the threshold monotonically degraded returns (+3.3%
down to −0.3% at the tightest setting); out of sample, tightening improved them (+4.6% up to
+6.7%). The in-sample regression agrees that there is nothing there — the coefficient on richness
is −0.13 with a t-statistic of −1.56 (p = 0.12).

An unstable sign across periods is what an absent effect looks like. Reporting it is more useful
than presenting a filter fitted to whichever half of the sample flattered it.

## How the term structure is handled

Cboe publishes a single-name 30-day index and nothing else, so measuring a 42- or 63-day premium
needs a maturity adjustment. Rather than assume a shape, the study **measures** one: Cboe
publishes four points on the S&P 500 curve every day, and interpolating that observed curve — in
total variance, the only convention free of calendar arbitrage — gives the shape to apply.

| Index | Horizon | Mean level over the study period |
|---|---:|---:|
| VIX9D | 9 days | 17.5% |
| VIX | 30 days | 18.2% |
| VIX3M | 93 days | 20.1% |
| VIX1Y | 365 days | 22.7% |

Applying the index's *relative* shape to a single name's own 30-day index is an approximation, and
a documented one: single-name term structure is not observable free of charge and is generally
flatter than the index's. The approximation is mild in practice because the study's horizons land
close to points Cboe already publishes — 21 trading days is about 30 calendar days, and 63 is
about 91.

## Quickstart

Requires Python 3.11 or newer; developed and validated on 3.13.

```bash
make setup      # install Python 3.13 and dependencies
make data       # download the Cboe volatility history and the price history
make reproduce  # run the study, writing to outputs/
```

Or with plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/fetch_cboe_data.py
python scripts/fetch_price_data.py
vrp --output-dir outputs
```

A successful run prints:

```text
Completed 555 delta-hedged trades over 11602 matched observations: mean variance risk premium 3.84 volatility points (positive 78.1%), mean short return on capital 4.06% per trade, win rate 74.2%.
```

`vrp --help` documents every option. The study is importable too:

```python
from pathlib import Path
from vrp import CostModel, StudyConfig, run_study

results = run_study(
    Path("data/raw"), Path("data/reference/study_universe.csv"), StudyConfig(), CostModel()
)
print(results.pooled_panel["mean_vrp_vol_points"])
```

### Development tasks

```bash
make lint       # ruff check + format check
make typecheck  # mypy, strict
make test       # 213 tests
make verify     # re-run and diff against outputs/
make help       # list every target
```

## Method

### Measuring the premium

For every trading day and horizon, implied variance is compared against the variance the
underlying actually delivered over exactly that horizon:

- **Implied variance** — the security's observed 30-day index, scaled to the horizon by the
  observed S&P 500 term structure, squared.
- **Realised variance** — annualised sum of squared close-to-close log returns over the following
  window.
- **Premium** — implied minus realised.

The realised leg is not knowable at the observation date, so no panel row can serve as a trading
signal. The test suite enforces that separation directly: it verifies that altering prices *after*
an observation cannot change any signal input, and that altering prices *before* it cannot change
the forward realised variance.

### Simulating the trade

One call plus one put at a common strike set from the entry forward. The package is marked daily
with the security's observed implied volatility and observed close, the Black-Scholes delta hedge
is rebalanced, the position is financed, and it is held to cash settlement. Entries do not
overlap — the next begins when the previous expires — because overlapping trades would share most
of one price path and inflate the sample without adding information.

Each day's profit is decomposed by the identity

```text
dV − Delta·dS + financing  =  Theta·dt  +  ½·Gamma·dS²  +  Vega·dIV  +  residual
```

Read left to right that is the definition of a delta-hedged gain; read right to left it is the
economics of the trade.

### Costs and capital

Entry half-spread of 1% of premium, $0.65 per contract per leg, and hedge turnover at 1 basis
point for single names and 0.5 for the index. The capital denominator is the greater of 20% of
spot notional and 1.5 times entry premium — transparent and conservative, but explicitly **not**
broker margin, which depends on offsets, concentration, stress scans, liquidity, and house rules.

## How the code is organised

| Module | Responsibility |
|---|---|
| [`config.py`](src/vrp/config.py) | Study period, horizons, cost model, and the data contract |
| [`loaders.py`](src/vrp/loaders.py) | Reading volatility, price, and universe inputs, failing loudly |
| [`blackscholes.py`](src/vrp/blackscholes.py) | Option marks and Greeks |
| [`termstructure.py`](src/vrp/termstructure.py) | Matching implied volatility to the study horizons |
| [`realized.py`](src/vrp/realized.py) | Realised variance, forward and trailing |
| [`panel.py`](src/vrp/panel.py) | The implied-versus-realised variance panel |
| [`hedged.py`](src/vrp/hedged.py) | One delta-hedged trade and its decomposition |
| [`strategy.py`](src/vrp/strategy.py) | Which trades to run, including the robustness variants |
| [`signals.py`](src/vrp/signals.py) | Conditioning signals with in-sample-only selection |
| [`aggregate.py`](src/vrp/aggregate.py) | Summaries, regimes, tails, bootstrap |
| [`figures.py`](src/vrp/figures.py) | The four figures |
| [`verify.py`](src/vrp/verify.py) | Tolerance-based comparison of two result sets |
| [`pipeline.py`](src/vrp/pipeline.py) | End-to-end orchestration |
| [`cli.py`](src/vrp/cli.py) | Command-line interface |

## Reproducibility

`outputs/` holds the committed result set and the test suite checks the study still produces it:

```bash
make test    # includes the end-to-end reproduction check across all 15 tables
make verify  # re-run and print the largest difference found
```

**The study period ends on a fixed date on purpose.** Both data providers extend their series
every trading day, so an open-ended sample would give a slightly different answer every time it
was downloaded. Freezing the end date is what lets a download taken months later reproduce the
published numbers exactly.

Bit-for-bit equality is still not the target, and cannot be. IEEE 754 requires `+ - * / sqrt` to
be correctly rounded, so those agree everywhere, but it deliberately imposes no such requirement
on `exp`, `log` or `erf` — each platform's maths library may use its own approximation. The
least-squares and interpolation routines add to this by depending on the host's linear-algebra
library, where floating-point addition is not associative and a different summation order gives a
different last digit. Identical code on macOS and on Linux therefore disagrees at around `1e-13`
relative.

Two values are treated as agreeing when

```text
|a - b| <= atol + rtol * max(|a|, |b|)
```

with `rtol = 1e-9` and `atol = 1e-10`. The absolute term is not decoration. Several exported
columns are the *residual of an identity* whose correct value is zero — `attribution_error`
reports how far the Greek decomposition missed the realised profit, and a correct run puts it at
`1e-14`. Comparing `1e-14` against `0.0` relatively gives a difference of 100%, so a relative-only
check fails a study that in fact reproduced perfectly. The floor sits seven orders of magnitude
below the smallest quantity this study reports, so it cannot mask a real difference. Both
behaviours are covered by tests in `tests/test_verify.py`.

`outputs/summary.json` records the configuration, the cost model, and which securities were
measured, so the exact basis of the committed results is documented.

## Data provenance

Neither input is redistributed here.

- **Implied volatility** — Cboe, under Cboe's terms of use. `scripts/fetch_cboe_data.py`
  downloads VIX9D, VIX, VIX3M, VIX1Y, VXAPL, and VXAZN from
  <https://cdn.cboe.com/api/global/us_indices/daily_prices/>.
- **Prices** — Yahoo Finance's public chart endpoint, via `scripts/fetch_price_data.py`. It needs
  no API key, which is why it is used, but it is undocumented and can change. If it breaks, any
  provider's export in the same `date,close` shape will be read unchanged.

The study universe and each security's evidence status are version-controlled, in
`data/reference/study_universe.csv`.

## Limitations

- Three securities are measured, not the twenty-one in the universe. Free implied-volatility
  history is the binding constraint.
- Option prices are Black-Scholes marks from an at-the-money volatility, not historical quotes. A
  real chain has a skew, a bid-ask spread at every strike, and finite depth.
- The capital denominator is a research normalisation. Real margin would rise precisely when
  losses occur, which is exactly when the reported returns look worst.
- Daily closes cannot represent intraday entry, exit, or hedging.
- The single-name term-structure adjustment borrows the index's shape.
- A high win rate with a long left tail is characteristic of short-volatility strategies. The mean
  is not the risk.
- No claim is made that this is implementable at these returns after real execution costs.

**Results are research findings, not investment advice.**

## Origin

This repository reimplements a study that previously existed only as a Word report and an Excel
workbook, both preserved in `deliverables/` (kept out of version control as large binaries). In
that original, only three volatility indexes were observed; all returns, surfaces, and option P&L
were, in its own words, "deterministic fixtures". Its analysis code was never delivered, and the
fixture generator was never specified, so its exact figures cannot be regenerated.

This implementation replaces the simulated returns with real observed prices, which is why the
numbers here differ from the report's — they now rest on market data rather than on generated
paths.

## License

[MIT](LICENSE).

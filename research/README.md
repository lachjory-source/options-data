# Research

A record of strategy tests run in August 2026. Most are null results. They are
here because null results are the point: each one cost an evening instead of an
evaluation fee.

Every script validates itself against synthetic data with a known answer before
touching real data. Several caught bugs in my own code that way.

---

## Method

Three rules applied throughout:

1. **Compare against a null, not against zero.** Random entries with the same
   stop and holding period, or a bootstrap from random dates. "Beat buy and
   hold" is not a benchmark.
2. **Validate the engine before trusting the result.** Run the strategy on data
   with no structure. If it finds an edge there, the code is broken.
3. **Replicate across markets.** A single-market finding survived twice in this
   repo and failed both times under wider testing.

---

## Results

### 01 — Liquidity sweeps

Wick through a prior swing low, close back inside, trade the reversal.

| Market | Trades | Expectancy | vs random entries |
|---|---|---|---|
| BTCUSDT 1h | 1,061 | -0.468R | 39th percentile |
| SPY daily | 371 | -0.074R | 69th percentile |

**Null in both.** Indistinguishable from entering at random moments.

Incidental finding that mattered more: the worst single SPY trade lost
**-14.63R** after gapping through its stop. At any risk above 0.34% per trade,
that one trade ends a prop firm evaluation on its own.

### 02 — Prop firm evaluation simulator

Monte Carlo of a 10% target against a 5% daily loss limit and 10% max drawdown.

- Zero edge, sized optimally: **37% pass rate**, not 50%. The daily limit is a
  one-way barrier.
- Minimum expectancy for better-than-coinflip odds: **+0.10R per trade**
- **Position size dominates edge.** +0.05R at 0.5% risk beats +0.30R at 2% risk.
- Fat tails from gaps cost 12 to 25 percentage points of pass probability.
- At 2% risk the daily loss limit causes 78% of failures, which reproduces the
  published industry figure without assuming any skill in the population.

### 03 — Realized volatility forecasting

HAR (Corsi 2009) against naive benchmarks, expanding-window out-of-sample.

- **23.2% out-of-sample R²** on BTC 10-day forward volatility
- 28.1% against a trailing-average benchmark
- Adding jump, semivariance and vol-of-vol features: **-3.6% versus HAR**

Six features lost to three, on both simulated and real data.

### 04 — Implied volatility encompassing regression

Does the HAR forecast contain anything the options market has not priced?

```
log(realised vol) = a + b·log(IV) + c·log(HAR)
```

| Term | Coefficient | Newey-West t |
|---|---|---|
| log(IV) | 0.847 | 6.42 |
| log(HAR) | **-0.011** | **-0.04** |

IV alone: R² 44.0%. IV plus HAR: R² 44.0%. **The model adds nothing.**

Variance risk premium confirmed as a sanity check: implied 52.4% against
subsequent realised 47.3%, t = 3.84.

### 05 — Opening range breakout: measuring what optimisation invents

Ran an ORB at all 24 hourly slots on BTC, a market with no opening bell, so any
edge found is manufactured.

| | Fixed parameters | Best of 12 combos |
|---|---|---|
| Profitable slots | 0 of 24 | 8 of 24 |
| Mean expectancy | -0.150R | -0.013R |

**Optimisation lift: +0.137R on random data, +0.190R on real BTC, +0.151R on
the S&P 500.** Stable across three unrelated datasets.

That number is the bar any backtest has to clear before it means anything.

### 06 — Opening range breakout: cross-market replication

The real opening bell against 47 fabricated ones in the same session.

| Market | Real open | Null mean | Rank | z |
|---|---|---|---|---|
| S&P 500 | -0.021R | -0.136R | 1/14 | +1.85 |
| DAX | -0.022R | -0.152R | 4/18 | +1.25 |
| FTSE 100 | -0.122R | -0.252R | 1/18 | +1.68 |
| Nikkei 225 | +0.017R | -0.153R | 2/13 | +1.68 |
| ASX 200 | -0.205R | -0.268R | 4/13 | +0.91 |

All five z-scores positive: sign test p = 0.031. Fisher combined p = 0.028.

**The opening effect is real and worth about +0.12R** relative to a random
session slot. It is also smaller than the cost of trading it.

### 07 — Opening range breakout: walk-forward redesign

Wider stops and targets, parameters chosen on training windows and traded blind.

Profitable out-of-sample in **1 of 5** markets. The aggregate +0.019R edge came
entirely from the Nikkei; excluding it, the edge is **-0.004R**.

**Failed.** Logged and stopped per the pre-registered rule.

### 08 — Intraday momentum (Baltussen et al., JFE 2021)

Last 30 minutes predicted by the rest of the day.

Betas 0.005 to 0.015, none significant. The closing window ranked 4th to 10th
of twelve, so no concentration at the close.

**Inconclusive rather than null.** The mechanism operates in the closing
auction, and CFD data almost certainly excludes the auction print. Testing this
properly needs futures data with settlement prices.

### 09 — OpEx gamma: does volatility expand after expiry?

The claim, sold commercially, is that gamma rolls off at monthly expiry and
volatility expands afterwards.

| Market | Change in daily range | t | Bootstrap percentile |
|---|---|---|---|
| S&P 500 | **-10.2%** | -3.13 | 0.1st |
| DAX | -4.0% | -1.23 | 10.5th |
| FTSE 100 | -7.1% | -2.24 | 0.7th |
| Nikkei 225 | -7.5% | -2.55 | 0.9th |
| ASX 200 | -4.7% | -1.80 | 3.5th |

**Volatility contracts.** Opposite to the marketed direction, in all five
markets, concentrated on the third Friday rather than other Fridays, and larger
in triple-witching months in every market.

### 10 — OpEx gamma: robustness

Medians made the effect **larger** in four of five markets, which rules out a
handful of CPI or FOMC days driving it.

The real story is a structural break:

| Market | 2012-2018 | 2019-2025 |
|---|---|---|
| S&P 500 | -0.324 | -0.003 |
| DAX | -0.158 | +0.020 |
| FTSE 100 | -0.211 | -0.004 |
| Nikkei 225 | -0.191 | -0.031 |
| ASX 200 | -0.075 | +0.002 |

Strong before 2019, gone after, in all five markets simultaneously.

This argues **against** a macro-calendar confound, since the CPI and FOMC
calendar has not changed, and **for** an options-structure explanation: weekly
and zero-day options grew over exactly this period, so gamma now rolls off
continuously instead of piling into one monthly cliff. That is a hypothesis
fitted to the timing, not something tested here.

### 11 — Gamma exposure: is the collected snapshot fit to use?

Not a strategy test, and not an analysis of the collected data. Collection
started 12 August 2026, so there is one snapshot and nothing can be concluded
from it about gamma or about price.

This asks whether the collector is configured correctly, which is a question
that needs exactly one snapshot and gains nothing from the second. It was run on
day one because a config problem found now is a config change, and the same
problem found in six months costs six months of collection.

Engine validated on synthetic chains first: **31 known-answer tests**, checking
both directions. Detectors must fire on injected defects and stay silent on
clean data.

| | SPY | QQQ | IWM |
|---|---|---|---|
| Contracts | 5,736 | 5,299 | 1,940 |
| Expiries | 17 | 17 | 15 |
| Median strikes/expiry | 177 | 193 | 85 |
| Strike range (x spot) | 0.39 - 1.30 | 0.29 - 1.53 | 0.30 - 1.33 |
| Parity regression median R2 | 0.9999 | 0.9999 | 1.0000 |
| Implied carry (r - q) | +2.89% | +3.17% | +1.65% |
| Gross gamma at edge strikes | 0.010% | 0.019% | 0.014% |
| Unusable IV | 2.51% | 0.68% | 1.60% |
| Call-vs-put IV gap, gamma-weighted | 3.25 pts | 3.32 pts | 3.59 pts |

**The collector does not need a config change.** Coverage, open interest and
quote integrity are all fine.

**But the stored implied volatility field should not be used.** Put-call parity
says a call and a put on the same strike must have identical implied volatility.
In yfinance's `impliedVolatility` column they disagree by over 3 volatility
points even after weighting by gross gamma, so the gap is not confined to
illiquid wings. Gamma is a function of IV and inherits the error. Bid and ask are
already stored, so this is fixed at analysis time by re-inverting the mid price:
an argument for the store-raw-derive-later design rather than against it.

The implied carry is the strongest independent check. Nothing in the pipeline is
told the interest rate or the dividend yield; both fall out of a put-call parity
regression on the quotes. Getting +2.9% for SPY against a roughly 4.2% policy
rate and a 1.1% distribution yield is a number that would not appear if the
pricer, the forward recovery or the expiry parsing were wrong.

**One real defect, and it is a timing discontinuity rather than a config
problem.** The 2026-08-12 snapshot was the manual first run, taken at 02:23 UTC,
which is 10:23pm New York on the 11th. At that hour spot is the 11 August close
but open interest is still 10 August, because the OCC publishes overnight. Every
scheduled 13:00 UTC run has spot and open interest aligned to the same prior
session. **Row one of the dataset uses a different date offset from every row
after it.** Same shape as the Binance millisecond-to-microsecond switch: a
convention change mid-series that produces plausible numbers instead of an error.

**A negative result on the dealer sign convention, before any data was needed.**
If convention B is the exact negation of convention A, the zero-gamma flip level
is identical under both and only the interpretation flips. So "test A against B"
is one correlation whose sign you read, not two competing models. The only
implemented convention that genuinely moves the flip level is the
moneyness-conditional one, and on real data it produces **8 zero crossings on
SPY and 12 on QQQ**, because each strike flips sign as spot sweeps past it. Any
vendor quoting "the" flip level under that convention is picking one crossing
out of eight.

**And the six-month plan does not have the sample size.** Net GEX is a stock
variable, so daily observations are heavily autocorrelated.

| True correlation | Independent observations needed |
|---|---|
| 0.10 | 783 |
| 0.20 | 194 |
| 0.40 | 47 |

125 trading days at AR(1) = 0.95 is an effective sample size of **3.2**. Test
changes in GEX rather than levels: differencing turns the stock into a flow, and
monthly expiry is the largest natural experiment in the dataset, which connects
directly to tests 09 and 10.

---

## What this adds up to

Four independent approaches, all on free and publicly accessible data, all null
or unusable. That is consistent with the view that accessible data is the
picked-over part of the space.

The most reusable output is not a strategy. It is the +0.15R optimisation lift:
a measured quantity for how much apparent edge a small parameter search
fabricates from nothing.

Test 11 added a second one, about the method rather than the markets. Every
script here validates against synthetic data before touching real data, and that
caught real bugs. But test 11's engine passed 28 synthetic tests and then broke
on first contact with a live SPY chain, because the generator and the engine
shared a European-options assumption that real American ETF options violate.
**Synthetic validation cannot catch an error the generator also makes.** It is a
necessary step, not a sufficient one.

---

## Running these

Each file is standalone Python. Paste into a Colab notebook and run.

Data sources: Binance public archives (crypto, free), Dukascopy via
`dukascopy-python` (index CFDs, free), Deribit DVOL API (crypto implied
volatility, free), and `data/` in this repo for option chains, which exists
because per-strike open interest history is not sold cheaply and no free source
of it was found.

Some scripts need `!pip install dukascopy-python` in a cell first. Test 11 needs
nothing installed and no clone: it reads `data/` from this repo over https, so
pasting it into a cell reproduces the run. Point `SNAPSHOT_DIR` at a different
date folder to read a different day, or at a local path to work offline. If the
snapshots are unreachable it falls back to synthetic chains, so the self-test
and both controls still run with no network.

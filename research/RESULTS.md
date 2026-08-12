# Results Log

Actual output from every script in `research/`, August 2026.

Each test validates itself on synthetic data with a known answer before running
on real data. Where a validation step is shown, it passed unless stated.

---

## 01 — Liquidity sweeps

### BTCUSDT, 1-hour bars, Jan 2024 to Jun 2025

**Engine validation** (sweep on a pure random walk, must not beat random entries)

```
Trades              2531
Win rate            24.5%
Expectancy          -0.5614R
Random-entry control on same data: -0.5419R
Sweep sits at the 28th percentile  ->  PASS, no lookahead bias
```

**Real data**

```
Trades              1061
Win rate            31.7%
Avg win / loss      +1.54R / -1.40R
Expectancy          -0.4677R
Max drawdown        500.5R
t-stat              -9.73
Median stop size    0.428% of price
Cost per trade      0.234R   (23% of risk, per trade)
```

**Verdict test**

```
Random entries, same risk/target structure:
  mean -0.4578R  (5th-95th pct: -0.5614 to -0.3783)
Sweep expectancy  -0.4677R
Sweep beats 39.0% of random runs.
```

**NULL.** Indistinguishable from entering at random.

Note: costs alone were -0.234R. Gross expectancy was roughly a coinflip, so
fees made a bad strategy worse rather than causing the result.

---

### SPY, daily bars, 1998 to 2017 (in-sample), gap-aware fills

**Exit breakdown**

| Exit | Count | Mean |
|---|---|---|
| Normal stop | 187 | -1.07R |
| **Gapped through stop** | **55** | **-2.30R** |
| Gapped past target | 27 | +4.02R |
| Target | 98 | +1.94R |
| Timeout | 4 | +0.08R |

**By year** (20 years, 11 to 24 trades each)

Twelve of twenty years profitable. The strategy still lost money overall,
because 2007, 2008, 2014 and 2015 were bad enough to swamp the winners.

**Verdict test**

```
Random entries, same structure: -0.1789R  (5th-95th: -0.6342 to +0.2205)
Sweep expectancy: -0.0735R
Sweep beats 69.0% of random runs.
```

**NULL**, though a weaker test than BTC: the control band is nearly five times
wider, so only a very large edge could have been detected.

**Incidental finding that mattered more than the strategy result:** worst
single trade **-14.63R**, from a roughly 5% overnight gap against a 0.35% stop.

| Risk per trade | That one trade costs | Result |
|---|---|---|
| 0.25% | -3.7% | Survivable |
| 0.50% | -7.3% | Breaches a 5% daily limit |
| 1.00% | -14.6% | Breaches the full 10% drawdown |

---

## 02 — Prop firm evaluation simulator

Rules: 10% target, 5% daily loss limit, 10% max drawdown, 3 trades/day,
60-day limit, minimum 5 days.

**Validation** (symmetric barriers, no edge, no daily limit → must be 50%)

```
pass 50.2%   dd 49.8%   timeout 0.0%   ->  PASS
```

### Baseline: zero edge, realistic fat tails

| Risk/trade | Pass | Daily limit | Max DD | Timeout |
|---|---|---|---|---|
| 0.25% | 8.1% | 0.0% | 8.2% | 83.8% |
| 0.50% | 36.3% | 3.3% | 34.5% | 25.9% |
| 1.00% | 36.9% | 42.0% | 21.1% | 0.0% |
| 2.00% | 16.7% | 78.4% | 5.0% | 0.0% |
| 3.00% | 7.9% | 82.8% | 9.3% | 0.0% |

Zero edge, sized optimally, passes **37%** of the time. Not 50%. The daily
limit is a one-way barrier.

At 2% risk it causes 78% of failures, which reproduces the published industry
figure without assuming any skill in the population.

### How much edge is actually needed

Pass probability by expectancy and position size:

| Expectancy | 0.25% | 0.50% | 1.00% | 2.00% |
|---|---|---|---|---|
| -0.05R | 4.1% | 24.4% | 30.9% | 15.6% |
| +0.00R | 8.2% | 35.8% | 37.5% | 17.3% |
| +0.05R | 15.6% | **48.7%** | 44.2% | 18.7% |
| +0.10R | 25.5% | **62.3%** | 51.2% | 20.0% |
| +0.15R | 38.4% | **73.7%** | 57.9% | 31.2% |
| +0.20R | 52.4% | **83.5%** | 64.0% | 34.7% |
| +0.30R | 80.1% | **94.9%** | 74.3% | **40.8%** |

**Minimum expectancy for better-than-coinflip odds: +0.10R.**

**Position size dominates edge.** +0.05R at 0.5% risk (48.7%) beats +0.30R at
2% risk (40.8%). Six times the edge, sized wrong, loses.

### What fat tails cost

| Expectancy | Clean -1R losses | With gap tail | Cost |
|---|---|---|---|
| +0.00R | 48.6% | 36.9% | -11.6pp |
| +0.10R | 72.0% | 50.5% | -21.5pp |
| +0.20R | 88.9% | 63.5% | -25.4pp |

The better the strategy, the more the tails cost, because there was more to lose.

### Run against the actual backtested SPY trades

```
Loaded 371 trades, expectancy -0.0735R, worst -14.63R

risk 0.50%  ->  pass 27.3%   daily-limit 21.9%   maxDD 44.9%
risk 1.00%  ->  pass 28.3%   daily-limit 31.6%   maxDD 40.1%
risk 2.00%  ->  pass 12.8%   daily-limit 78.6%   maxDD  8.6%
```

Best case 28%, against 37% for a zero-edge strategy. The tested strategy
performed **worse than having no strategy**, because it combined slightly
negative expectancy with a fatter tail.

---

## 03 — Realized volatility forecasting

BTCUSDT, hourly bars building daily realized measures, 2023 to 2025.
Expanding-window out-of-sample, 665 forecasts, 67 non-overlapping.

**Validation on simulated GARCH(1,1)**: HAR beat naive, extended features added
nothing (t = 0.08). Both PASS.

**Real data, 10-day forward volatility**

| Model | RMSE | QLIKE | R² vs mean | R² vs trailing avg | R² vs HAR |
|---|---|---|---|---|---|
| Naive 1-day | 1.0180 | 0.9816 | -180.9% | -162.9% | -265.8% |
| Naive 10-day | 0.6279 | 0.2269 | -6.9% | 0.0% | -39.2% |
| **HAR** | **0.5323** | **0.1660** | **23.2%** | **28.1%** | 0.0% |
| Extended (6 features) | 0.5418 | 0.1623 | 20.4% | 25.5% | **-3.6%** |

**Diebold-Mariano vs HAR** (67 independent observations)

| Model | Squared error | QLIKE |
|---|---|---|
| Naive 1-day | 3.76 | 2.87 |
| Naive 10-day | 2.11 | 1.90 |
| Extended | -0.51 | -0.99 |

**Volatility is genuinely forecastable.** HAR achieves 23.2% out-of-sample R².

Adding jump components, downside semivariance and vol-of-vol **subtracted**
value. Six features lost to three, on both simulated and real data.

Note: HAR beating the fair trailing-average benchmark sits at t = -2.11 on
squared error and -1.90 on QLIKE, so it is marginal on 67 observations.

Note also that the trailing 10-day average scored -6.9% against the mean, i.e.
it is *worse than a constant*. BTC volatility mean-reverted at this horizon, so
extrapolating recent volatility was actively counterproductive.

---

## 04 — Does the forecast beat the market? (encompassing regression)

BTC, 30-day horizon to match Deribit's DVOL construction.
745 aligned days, 24 independent windows.

**Sanity check: does the variance risk premium appear?**

```
Mean IV:                    52.4%
Mean subsequent realised:   47.3%
Mean log gap (the VRP):     +0.1117   t = 3.84
->  PASS, positive as the literature says
```

**Univariate forecasts**

| Predictor | R² | Slope | t |
|---|---|---|---|
| Implied volatility (the market) | **44.0%** | 0.84 | 7.53 |
| HAR (the model) | 15.5% | 0.98 | 3.49 |

**Encompassing regression**, Newey-West standard errors, lag 30

```
log(realised vol) = a + b·log(IV) + c·log(HAR)

        term      coef     NW se        t
   intercept     0.532     0.773     0.69
     log(IV)     0.847     0.132     6.42
    log(HAR)    -0.011     0.239    -0.04

Combined R2: 44.0%
```

**IV alone gives 44.0%. IV plus HAR gives 44.0%.** The model adds nothing the
options market has not already priced.

**Trading version** (does the IV-minus-forecast gap time the premium?)

```
slope 0.164   NW t 1.29   R2 1.9%   ->  no timing value

Mean realised premium by signal tercile:
  market looks cheap: +0.0997  (n=249)
              middle: +0.0783  (n=247)
   market looks rich: +0.1569  (n=249)
```

Non-monotonic, and all three terciles positive. The premium exists regardless
of the signal, so you would earn it by always selling, not by timing.

---

## 05 — How much edge does optimisation manufacture?

An opening range breakout run at all 24 hourly slots on BTC, a 24/7 market with
no opening bell, so any edge found is fabricated by the search.

### Pure random walk (control)

| | Fixed parameters | Best of 12 combos |
|---|---|---|
| Mean expectancy | -0.1501R | -0.0131R |
| Best single hour | -0.0224R | +0.0873R |
| Hours showing a profit | **0 of 24** | **8 of 24** |
| Hours significantly positive | 0 | 0 |

```
OPTIMIZATION LIFT: +0.1369R per trade
```

Second seed: lift +0.1416R. Consistent.

### Real BTC, 5-minute bars, 2024 to 2025

```
mean expectancy across hours   baseline -0.5199   optimized -0.0158
OPTIMIZATION LIFT: +0.5041R
```

**Contaminated.** Hour 22 returned -7.75R, a divide-by-tiny-range artefact from
a near-flat opening range. Excluding it:

| | Random walk | Real BTC (ex hour 22) |
|---|---|---|
| Baseline mean | -0.150R | -0.206R |
| After optimisation | -0.013R | **-0.016R** |
| Optimisation lift | +0.137R | **+0.190R** |
| Hours profitable, before → after | 0 → 8 | 2 → 8 |
| Hours significantly positive | 0 | **0** |

Real Bitcoin behaved almost identically to a random walk. Post-optimisation
expectancy matched to within 0.003R.

**Later measured at +0.1509R on the S&P 500.** Three unrelated datasets,
+0.137R / +0.190R / +0.151R.

---

## 06 — Opening range breakout: cross-market replication

The real opening bell against fabricated opens at other slots in the same
session, five markets, 2021 to 2025, 5-minute bars.

### Single market first (S&P 500)

The confound, made explicit:

| | Mean expectancy | Median opening range | Slots |
|---|---|---|---|
| Cash session | -0.1280R | 0.154% | 14 |
| Outside session | -0.2941R | 0.073% | 31 |

Overnight ranges are half as wide and expectancy is twice as bad. An early
version compared the open against 3am and reported it beating 100% of fake
opens, which was measuring liquidity, not an opening effect.

Against a **liquidity-matched null** (cash-session slots only):

```
Cash-session null: mean -0.1363R, sd 0.0625, n = 13 slots
REAL OPEN:         -0.0209R
Rank:              1 of 14   (1 = best)
z-score:           +1.85
```

Monday test (mechanism predicts Monday strongest after a weekend of news):

```
Mon +0.0329R  Tue -0.0521R  Wed -0.1885R  Thu +0.0030R  Fri +0.1043R
```

No gradient. Friday strongest. **The proposed mechanism failed its own test.**

### Five markets

| Market | Real open | Null mean | Rank | z | p |
|---|---|---|---|---|---|
| S&P 500 | -0.0209 | -0.1363 | 1/14 | +1.85 | 0.071 |
| DAX | -0.0218 | -0.1521 | 4/18 | +1.25 | 0.222 |
| FTSE 100 | -0.1216 | -0.2522 | 1/18 | +1.68 | 0.056 |
| Nikkei 225 | +0.0171 | -0.1526 | 2/13 | +1.68 | 0.154 |
| ASX 200 | -0.2053 | -0.2682 | 4/13 | +0.91 | 0.308 |

```
Markets where the open ranked FIRST:    2 of 5
Markets where the open was PROFITABLE:  1 of 5
Fisher combined p-value:                0.0277
```

**All five z-scores positive**, which is a straight binomial at **p = 0.031**
and does not depend on the rank-to-p conversion.

**The opening effect is real, worth about +0.12R** relative to a random session
slot (real opens average -0.0705R, their nulls average -0.1923R).

It is also smaller than the cost of trading it.

Caveat: indices are correlated, so these are not five independent tests and the
combined p-value is optimistic.

---

## 07 — Opening range breakout: walk-forward redesign

16 combinations of stop width and target, parameters chosen on 500 training
sessions and traded blind on the next 125, rolling. Only blind results reported.
Controls at 4 fake opens per market.

**Validation**: on random data, walk-forward returned -0.005R (t = -0.16). In-sample
best was +0.007R, so the in-sample inflation for this specific search was
**+0.012R**, much smaller than the +0.15R from the larger 24-slot search.

**Out-of-sample results**

| Market | Real (OOS) | t | Controls | Beats all? |
|---|---|---|---|---|
| S&P 500 | -0.0359 | -0.54 | -0.0045 | No |
| DAX | -0.0179 | -0.25 | -0.0556 | Yes |
| FTSE 100 | -0.0673 | -3.07 | -0.0533 | No |
| **Nikkei 225** | **+0.0910** | 0.86 | -0.0205 | No |
| ASX 200 | -0.0569 | -2.38 | -0.0482 | No |

```
markets profitable out-of-sample : 1 of 5
markets beating all controls     : 1 of 5
mean real -0.0174R   mean control -0.0364R   edge +0.0191R
```

**FAILED.** The +0.019R aggregate edge came entirely from the Nikkei. Excluding
it, the edge is **-0.004R**. Real beat control in only 2 of 5 markets, down from
5 of 5 on the gate 1 z-scores.

Stopped per the pre-registered rule.

Cost decomposition established here, on random data:

| Stop width | Cost in R | Expectancy |
|---|---|---|
| 1.0× range | 0.125R | -0.209R |
| 1.5× | 0.084R | -0.101R |
| 2.0× | 0.063R | -0.031R |
| 3.0× | 0.043R | -0.005R |

Cost in R terms is spread divided by stop distance. Double the stop, halve the
cost. Targets made results worse at every level.

---

## 08 — Intraday momentum (Baltussen et al., JFE 2021)

Does the last 30 minutes get predicted by the rest of the day? Five markets,
2021 to 2025. The paper's sample ends 2020, so this is entirely post-publication.

**Validation**: injected beta of 0.25 recovered as +0.2506 (t = 29.6); no effect
injected returned +0.0013 (t = 0.15). Detects real effects, does not invent them.

| Market | Beta | t | Window rank | Vol gradient | Net bps | Sharpe |
|---|---|---|---|---|---|---|
| S&P 500 | 0.0147 | 1.18 | 7 | +0.0239 | -1.96 | -1.19 |
| DAX | 0.0104 | 0.64 | 8 | -0.0048 | -0.72 | -0.66 |
| FTSE 100 | 0.0133 | 0.98 | 4 | -0.0229 | -1.39 | -1.48 |
| Nikkei 225 | 0.0047 | 0.16 | 7 | +0.0096 | -1.91 | -1.55 |
| ASX 200 | -0.0101 | -1.28 | 10 | +0.0215 | -1.88 | -3.07 |

```
beta positive:          4 of 5
beta significant (t>2): 0 of 5
vol gradient positive:  3 of 5
net profitable (t>2):   0 of 5
```

Betas 20 to 50 times smaller than the machinery can clearly detect. The closing
window ranked mid-pack in every market, so there is no concentration at the close.

**INCONCLUSIVE, not null.** The mechanism operates in the closing auction via
market-on-close orders and leveraged ETF rebalancing. The data used is broker
CFD quotes whose final bar covers 15:55 to 16:00, so the auction print is
almost certainly absent. A null from data that structurally cannot contain the
effect is not evidence against it.

---

## 09 — OpEx gamma: does volatility expand after expiry?

The commercially marketed claim is that dealer gamma rolls off at monthly
expiry and volatility expands afterwards. Tested on daily bars, 2012 to 2025,
five markets. Metric is the log ratio of mean daily range in the 5 sessions
after expiry versus the 5 before.

**Validation**: no effect returned -0.0312 (t = -1.27); +25% injected returned
+0.1919 (t = 7.79) against a target of +0.2231. Both PASS.

| Market | Log ratio | As % | t | Friday rank | Bootstrap pct | TW gradient |
|---|---|---|---|---|---|---|
| S&P 500 | -0.1072 | **-10.2%** | -3.13 | 4/4 | **0.1%** | -0.0947 |
| DAX | -0.0405 | -4.0% | -1.23 | 2/4 | 10.5% | -0.0724 |
| FTSE 100 | -0.0738 | -7.1% | -2.24 | 4/4 | **0.7%** | -0.0789 |
| Nikkei 225 | -0.0785 | -7.5% | -2.55 | 3/4 | **0.9%** | -0.0398 |
| ASX 200 | -0.0478 | -4.7% | -1.80 | 4/4 | **3.5%** | -0.0799 |

**Volatility contracts after expiry. All five markets. Opposite to the claim.**

- Three of five significant at t < -2
- Four of five below the 5th percentile of a bootstrap null built from 2000 random anchor dates
- Friday rank 4 of 4 in three markets: the third Friday shows the *most*
  contraction of any Friday, so the placebo control confirms expiry is special
- **Triple-witching gradient negative in all five markets**, and larger in
  magnitude (S&P -0.170 vs -0.075). More gamma rolling off means more
  contraction, which is a dose-response relationship

Absolute-return version agreed: S&P -15.8% (t = -3.38), FTSE -12.6%, Nikkei -11.2%.

---

## 10 — OpEx gamma: robustness

Attacking the rival explanation that mid-month CPI and FOMC prints sit in the
pre-expiry window and drive the whole result.

**Validation, all three aggregators** (a robustness check is useless if it is
merely less sensitive)

```
   mean: null -0.0312 (t -1.27)   with -20% injected -0.2544 (t -10.33)
 median: null -0.0028 (t -0.09)   with -20% injected -0.2259 (t  -7.02)
   trim: null -0.0247 (t -0.88)   with -20% injected -0.2479 (t  -8.77)
                                            target of injection: -0.2231
```

The median version is the best-behaved estimator and recovers the injected
effect almost exactly.

| Market | Mean | **Median** | Trimmed | Ex-2020 | 2012-2018 | 2019-2025 | % down |
|---|---|---|---|---|---|---|---|
| S&P 500 | -0.1072 | **-0.1617** | -0.1424 | -0.1886 | **-0.3244** | -0.0030 | 57% |
| DAX | -0.0405 | **-0.0651** | -0.1111 | -0.0971 | **-0.1583** | +0.0203 | 50% |
| FTSE 100 | -0.0738 | **-0.1029** | -0.0930 | -0.1238 | **-0.2112** | -0.0035 | 55% |
| Nikkei 225 | -0.0785 | **-0.1070** | -0.1069 | -0.1280 | **-0.1908** | -0.0312 | 59% |
| ASX 200 | -0.0478 | -0.0331 | -0.0541 | -0.0586 | **-0.0745** | +0.0024 | 58% |

**Medians made the effect larger in four of five markets.** If a handful of CPI
prints were responsible, medians would have shrunk it toward zero. The outlier
explanation is dead.

**The real finding is the structural break.** Strong in 2012-2018, gone in
2019-2025, in all five markets simultaneously. Every late-period estimate sits
between -0.031 and +0.020.

This argues **against** the macro-calendar confound, since the CPI and FOMC
calendar has not changed, and **for** an options-structure explanation: weekly
and zero-day options grew from marginal to dominant over exactly that period,
so gamma now rolls off continuously rather than piling into a monthly cliff.
That is a hypothesis fitted to the timing, not something tested here.

Effect size caveat: 50 to 59% of individual expiries contracted, so it was a
modest tilt rather than a reliable per-cycle event, and early-period t-stats on
medians run -1.3 to -2.4.

---

## Corrections and bugs found

Listed because they changed conclusions, and because a results log without them
is not honest.

**Timestamp parsing.** Binance switched from millisecond to microsecond
timestamps in January 2025, mid-window. The first version guessed the unit once
for the whole dataset, mapping every 2024 bar to January 1970. Prices and
ordering survived, so the headline result was unaffected, but every session and
time-of-day breakdown was noise. Caught by a printed date range reading "1970".

**A strawman benchmark.** The volatility forecaster originally compared HAR
against a 1-day naive forecast when the target was a 10-day average. That is
not a benchmark, and it inflated HAR's apparent advantage. Fixing it changed
the interpretation more than the number.

**Divide-by-tiny-range.** A near-flat opening range makes the risk denominator
almost zero, producing a -7.75R average for an entire hour from one session.
Fixed with a minimum range filter plus a cap, and the cap was then noted as
biasing against the fat right tail that breakout strategies depend on.

**Gap-blind fills.** Early versions assumed stops always filled at the stop
price. True enough for 24/7 crypto, false for anything with an overnight
session. Adding gap-aware fills revealed that 15% of SPY trades blew through
their stop, averaging -2.30R against a planned -1.00R.

**An off-by-one in a control.** Two windows were flagged as "the closing
window", which silently removed a legitimate comparison from the null set.

**A wrong automated verdict.** The robustness script concluded "PARTIAL,
large days are doing some of the work" because median t-stats fell. In fact the
median *point estimates* rose in four of five markets. Falling t-stats reflected
medians being noisier estimators, not a weaker effect. The summary logic read it
backwards and the underlying columns were right.

---

## Summary

| # | Test | Outcome |
|---|---|---|
| 01 | Liquidity sweeps, BTC and SPY | Null, twice |
| 02 | Prop firm evaluation simulator | Zero edge passes 37%; sizing dominates edge |
| 03 | Volatility forecasting | HAR works, 23.2% OOS R²; extra features do not |
| 04 | Forecast vs implied volatility | Redundant, t = -0.04 |
| 05 | Optimisation lift | **+0.15R fabricated, stable across three datasets** |
| 06 | Opening effect, five markets | Real, ~+0.12R, all five z positive |
| 07 | Walk-forward monetisation | Failed, edge was one market |
| 08 | Intraday momentum | Inconclusive, data lacks the closing auction |
| 09 | OpEx gamma | **Contracts, not expands. Opposite to the claim** |
| 10 | OpEx robustness | Survives medians; structural break at 2019 |

Four independent approaches on free public data, all null or unusable. The two
findings that stand are a measurement of how much edge tuning invents, and a
five-market contradiction of a commercially sold claim.

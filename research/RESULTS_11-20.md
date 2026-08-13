# Results Log, tests 11 onward (12+ unlogged)

Actual output from every script in `research/` numbered 11 and above. Tests 01
to 10 are in `RESULTS_1-10.md`.

Each test validates itself on synthetic data with a known answer before running
on real data. Where a validation step is shown, it passed unless stated. Bugs
that changed a conclusion are logged inside the test that found them, because a
results log without them is not honest.

---

## 11 — Gamma exposure: snapshot data QA

Run on `data/2026-08-12/{SPY,QQQ,IWM}.csv.gz`, the first snapshot. The script
reads these straight from the repo over https, so it reproduces from a bare
Colab cell with no clone and no upload.

**This is a test of the collector, not of the market.** It asks whether the
snapshots are fit to build a gamma dataset on. That question needs exactly one
snapshot and gains nothing from the second, which is why it was run on day one
rather than deferred: a config problem found now is a config change, the same
problem found in six months costs six months of collection.

Nothing here is a finding about gamma, dealer positioning or price. The dollar
figures and flip levels below are **engine diagnostics on a single day**,
included to show the code produces sane output and to demonstrate one structural
property of the sign conventions. Do not read them as results. With n = 1 they
cannot be.

**Engine validation** (synthetic chains with analytically known answers, both
directions: detectors must fire on injected defects AND stay silent on clean data)

```
PASS  gamma == finite difference (call)          fd=0.0342995179 analytic=0.0342997488
PASS  gamma == finite difference (put)           fd=0.0342995179 analytic=0.0342997488
PASS  call gamma == put gamma
PASS  IV inversion round-trips                   true=0.22 recovered=0.2200000000
PASS  IV inversion returns nan below intrinsic
PASS  parity recovers forward                    true=100.6928 fit=100.6928 r2=1.000000
PASS  parity recovers discount factor            true=0.990185 fit=0.990185
PASS  parity recovers carry r-q                  true=0.0280 fit=0.0280
PASS  clean chain fires no QA flags              fired: []
PASS  detects defect: zero_iv
PASS  detects defect: absurd_iv
PASS  detects defect: crossed_quotes
PASS  detects defect: zero_oi
PASS  detects defect: truncated_strikes
PASS  detects defect: duplicate_rows
PASS  detects defect: iv_field_wrong
PASS  detects defect: dropped_strikes
PASS  detects defect: yfinance_iv_sentinel
PASS  mixed $1/$5 spacing does NOT flag missing strikes
PASS  narrow chain flags gamma at edge strikes   edge share=0.3457
PASS  wide chain does not flag edge gamma        edge share=0.000004
PASS  symmetric chain: flip near spot            flip=99.9
PASS  zero OI gives zero gross gamma
PASS  zero OI gives no flip level (returns None)
PASS  short_call_long_put == -long_call_short_put
PASS  negated convention has IDENTICAL flip level
PASS  single-strike OI puts peak gamma at that strike
PASS  gross gamma is linear in OI                ratio=2.000000000000
PASS  sticky-strike vs sticky-moneyness flip differ on skewed chain
PASS  sticky-moneyness smile x is strictly increasing (no ties)
PASS  sticky-moneyness flip is reproducible

31/31 passed
```

### SPY, spot 770.56, 5,736 contracts

```
n_expiries                             17
strikes_per_expiry_median              177
parity_median_r2                       0.9999
parity_n_expiries_ok                   14 of 17
carry_annual_median                    0.0289
coverage_min_abs_z                     2.2607
gamma_share_at_edge_strikes            0.0001
iv_recompute_median_abs_err            0.0138   (n=231)
iv_call_put_gap_median                 0.0780   (n=2,236)
iv_call_put_gap_gamma_wtd              0.0325
iv_frac_below_1pct                     0.0251
oi_total                               13,014,716
oi_frac_zero                           0.0547
oi_put_call_ratio                      2.5592
quote_frac_crossed                     0.0000
spread_median_rel                      0.0362
```

Engine output, single snapshot, **diagnostic only**:

```
GEX TOTALS  ($ gamma per 1% move in spot)
  gross_gamma_dollars                    42,023,534,707
  gex_long_call_short_put                 3,628,788,235
  gex_short_call_long_put                -3,628,788,235
  gex_short_otm_long_itm                -19,801,776,908

ZERO-GAMMA FLIP LEVEL (re-priced across a spot grid, sticky-strike)
  long_call_short_put              768.23  (-0.30% from spot)   [1 crossing]
  short_call_long_put              768.23  (-0.30% from spot)   [1 crossing]
  short_otm_long_itm               787.87  (+2.25% from spot)   [8 crossings]
```

**Two flags.** `iv_zero_or_missing`: 144 contracts (2.51%) carry yfinance's
failed-inversion sentinel of 1e-5, excluded from gamma. And
`call_put_iv_inconsistent`, which is the section below.

The 8 crossings under `short_otm_long_itm` are the one line in that block that
carries information beyond this day. It is a property of the convention, not of
the 12 August chain: each strike flips sign as spot sweeps past it, so the net
curve is jagged and has no single zero. That holds for any chain.

### QQQ, spot 718.45, 5,299 contracts

Same caveat: collector diagnostics, not market findings.

```
parity_median_r2                       0.9999    (17 of 17 fits ok)
carry_annual_median                    0.0317
gamma_share_at_edge_strikes            0.0002
iv_recompute_median_abs_err            0.0145   (n=264)
iv_call_put_gap_median                 0.0531   (n=2,062)
iv_call_put_gap_gamma_wtd              0.0332
gross_gamma_dollars                    16,987,377,458
gex_long_call_short_put                 1,244,322,177
flip, long_call_short_put              715.24  (-0.45%)   [1 crossing]
flip, short_otm_long_itm               749.74  (+4.36%)   [12 crossings]
```

**One flag: `call_put_iv_inconsistent`.**

### IWM, spot 300.99, 1,940 contracts

```
parity_median_r2                       1.0000    (15 of 15 fits ok)
carry_annual_median                    0.0165
gamma_share_at_edge_strikes            0.0001
iv_recompute_median_abs_err            0.0134   (n=285)
iv_call_put_gap_median                 0.0474   (n=722)
iv_call_put_gap_gamma_wtd              0.0359
gross_gamma_dollars                     5,153,638,422
gex_long_call_short_put                  -382,397,738
flip, long_call_short_put              301.98  (+0.33%)   [1 crossing]
flip, short_otm_long_itm               314.91  (+4.63%)   [4 crossings]
```

**Two flags.** `call_put_iv_inconsistent`, below, plus
`missing_strikes_in_grid`: single holes at 275, 285, 290 and
314-323, all in the thin wings, carrying about 0.01% of gross gamma. A snapshot
cannot distinguish "Yahoo listed it and the collector dropped it" from "never
listed", and IWM is illiquid enough that the latter is likelier.

### The stored implied volatility field should not be used

Put-call parity implies a call and a put on the same strike and expiry have
**identical** implied volatility. In yfinance's `impliedVolatility` field they do
not.

| | SPY | QQQ | IWM |
|---|---|---|---|
| Strikes carrying both legs | 2,236 | 2,062 | 722 |
| Median call-vs-put IV gap | 7.80 pts | 5.31 pts | 4.74 pts |
| Same, weighted by gross gamma | **3.25 pts** | **3.32 pts** | **3.59 pts** |

The raw median is inflated by illiquid wings, where a large gap costs nothing
because there is no gamma there. Weighting by gross gamma measures the
disagreement where it can move a number, and it is still over 3 volatility points
on all three tickers. Gamma is a function of IV, so this propagates straight
through.

This is a sharper test than the earlier near-the-money check, which compared the
stored field against an inversion of the mid price and found only 1.3 to 1.5
points of disagreement. That check was restricted to strikes inside 0.85 to 1.15
of spot with two-sided quotes. The call-versus-put test needs no restriction and
no reference price, because parity supplies the answer: the gap should be zero.

**This does not require a collector change.** Bid and ask are already stored, so
IV can be recomputed at analysis time. It is an argument for the store-raw-derive-
later design, not against it.

### Verdict

**The collector is fit for purpose and needs no config change.** That is the
whole claim. No claim is made here about gamma or about SPY.

One analysis-time change it does argue for: recompute implied volatility from the
stored bid and ask rather than reading the `impliedVolatility` column.

The implied carry is the load-bearing check. No interest rate or dividend yield
is supplied anywhere in the pipeline; both are recovered from a put-call parity
regression on the quotes. +2.89% for SPY and +3.17% for QQQ against a roughly
4.2% policy rate and 1.1% / 0.5% distribution yields is a result that could not
appear if the pricer, the forward recovery or the expiry parsing were wrong.

**The one real problem is snapshot 2026-08-12 itself.**

| | |
|---|---|
| `meta.json` timestamp | `2026-08-12T02:23:26Z` |
| Workflow cron | `0 13 * * 1-5` |
| Latest `lastTradeDate` in the file | `2026-08-11 20:14:59Z` (4:14pm New York) |

02:23 UTC is 10:23pm New York on the 11th, hours after that session closed. This
was the manual `workflow_dispatch` first run.

At 13:00 UTC (9am New York, pre-open) `fast_info.last_price` returns the previous
close and the OCC has published the previous session's settled open interest
overnight, so spot and open interest are both as of day D-1. Aligned.

At 02:23 UTC spot is the 11 August close but open interest is still 10 August,
because the 11th had not published yet. **Spot and open interest are one session
apart in this file and in no other file that will ever be collected.**

`collect.py` states "whatever time you run this, the OI you get is yesterday's".
That is false before the OCC publishes, and it is precisely the belief that made
this file look correct.

### Postscript: the first scheduled run broke the same alignment, differently

Written the morning after. The collector's first automated run was due at 13:00 UTC,
before the US open. It fired at **16:13 UTC**, 3h13m late, which is 12:13pm in New York.
Mid-session.

GitHub Actions queues scheduled jobs on free runners and multi-hour delays are ordinary.
The consequence is the defect above with a different cause: the stored spot was a live
intraday price while the open interest was still the previous day's, so the pair was a
session apart again. The run also wrote to `data/2026-08-12`, the same UTC-dated folder as
the manual run, and **silently replaced it**. Spot changed from 770.56 to 772.39 under an
unchanged folder name, with no warning. The original survives only in git history at
`fdfe51d`.

The fix is to stop depending on the clock. The OCC publishes settled open interest
overnight, so throughout any New York calendar day D the open interest on hand is D-1's,
at 9am and at 5pm alike. The spot to pair with it is therefore the close of the last
trading day strictly before today's New York date: a calendar fact, not a timing one. Not
"the last completed session", which rolls forward at 4pm ET while open interest does not.
The folder is named for that session rather than the wall clock, so a delayed run lands in
the right place and cannot silently collide with an earlier one.

Tested across 13:00 to 23:30 UTC, a 10.5-hour window of possible delay: same spot, same
folder, every time. The test that mattered was not the stability one but the one checking
the rule is *correct*. The real 02:23 UTC manual snapshot carried 10 August open interest,
and the rule independently selects 10 August, so it reproduces something observed rather
than merely being self-consistent.

Worth recording that this took **27 hours** to surface, on the first automated run, in a
collector that had just passed a full QA review. The review was right about the data and
wrong about the system. It measured 5,736 SPY contracts and found them sound; what it
could not measure was an execution environment that had never executed anything yet.

**Validating a system's output is not the same as validating the system.** Third limit of
the method, after the two above, and the one most likely to recur.

### Bugs found

Four caught by the self-test before any real data was touched:

1. **The flip solver manufactured a level from nothing.** With zero open interest
   the net-GEX curve is identically zero, and the crossing detector reported a
   flip at every grid point, returning a confident `100.0` for a chain with no
   positions in it. Same failure mode as the -7.75R artefact in test 05: a null
   input producing a plausible number. Fixed with a degenerate-curve guard that
   returns `None`.
2. **The missing-strike detector false-alarmed on every realistic chain.** It
   assumed uniform strike spacing, so the legitimate $1-near-the-money /
   $5-in-the-tails ladder read as hundreds of dropped strikes. Rewritten to flag
   a gap only when it is large relative to the spacing on both sides.
3. **A threshold compared a strike count against a fraction of the row count.**
   Dimensionally incoherent.
4. **Duplicate contracts expanded the parity join into a cartesian product**,
   misaligning the call and put legs.

Five more that only appeared on real data:

5. **The IV sentinel test looked for `IV == 0`, but yfinance writes 1e-5.** It
   walked past 144 bad SPY contracts and fed a 0.001% volatility into a
   denominator.
6. **The parity fit rejected any discount factor above 1.0001.** SPY, QQQ and IWM
   options are **American**. Early exercise breaks put-call parity into an
   inequality band and biases the fitted discount factor above 1. This rejected
   15 of 17 SPY expiries whose regressions were R2 = 0.9999.
7. **The carry median included 0-DTE expiries.** Dividing a sub-cent price
   difference by T of about 0.001 gave +18% annualised carry for SPY. The correct
   answer, from 14 DTE and longer, is +2.89%.
8. **The stored-IV comparison sampled only from expiries passing the broken
   parity gate**, so it ran on 34 contracts and reported a verdict on yfinance's
   IV field. The correct sample is 231 and the answer changes.
9. **The loader dropped `dte <= 0`**, silently deleting the 0-DTE chain, which is
   the single largest gamma concentration in the file.

And one that neither could catch, found only by running the script on a second
machine:

10. **The sticky-moneyness smile had duplicate x-values.** Every strike carries a
    call and a put at the same log-moneyness, so the smile was built with 577
    points at 321 distinct positions. `np.interp` requires strictly increasing x
    and is undefined on ties, and `np.argsort` defaults to a **non-stable**
    quicksort, so which duplicate landed first depended on the numpy build. Same
    input, same code, different flip level: 766.82 in one environment against
    767.71 in another. Fixed by collapsing to one point per log-moneyness and
    averaging the call and put IV, which parity says should be equal anyway.

    This is the one bug in the list that no amount of testing on a single machine
    could have found, because the failure is *between* environments. It surfaced
    the first time someone else ran it.

Bugs 5 through 8 have the same shape: a threshold defensible in the abstract and
wrong against real market structure. Only bug 6 is a knowledge error. The rest
are the cost of validating against synthetic data generated by the same
assumptions the engine makes.

**Synthetic validation cannot catch an error shared by the generator and the
engine.** That is a real limit of the method used throughout this repo, and it
showed up on the first contact with live data.

Bug 10 adds a second limit: **validation on a single machine cannot catch an
error that lives between machines.** Reproducing a run somewhere else is not
ceremony. It is the only test that covers this class.

### Two predictions that were wrong

Recorded because logging wrong predictions is the point.

**Strike coverage was predicted to be a problem. It is not.** From "5,736
contracts, all within 90 days" the estimate was roughly 60 expiries and therefore
about 48 strikes each, or plus or minus 3% of spot. The real file has 17 expiries
with a median of 177 strikes, covering 0.39 to 1.30 of spot, and the measured
share of gross gamma on the outermost strikes is **0.010%**.

**The coverage threshold was derived from a false premise.** It assumed open
interest was uniform in log-moneyness, giving "gamma beyond |z| is about
2(1-Phi(z))" and a threshold of z = 2.5. Real open interest is concentrated near
the money, so far-dated wings carry far less gamma than that implies. On real SPY
the test fired at min |z| = 2.26 while measured edge gamma was 0.010%. Both
coverage flags are now anchored to the measured quantity, with z demoted to
corroboration.

### Not tested, and why

Which dealer sign convention forecasts realised volatility. Net GEX is a stock
variable, so daily observations are heavily autocorrelated.

| True correlation | Independent observations needed (alpha 0.05, power 0.80) |
|---|---|
| 0.10 | 783 |
| 0.15 | 347 |
| 0.20 | 194 |
| 0.30 | 85 |
| 0.40 | 47 |

Six months is about 125 trading days. At AR(1) = 0.95 the effective sample size
is **3.2**; at 0.90 it is 6.6. Aggregate open interest typically runs above 0.95.
Even three years of daily levels does not reach 47. Using overlapping forward
realised-volatility windows would inflate apparent n by the window length and
manufacture significance, which is the same multiple-comparisons failure this
repo is built to avoid.

Testing **changes** in GEX rather than levels turns the stock into a flow and
removes the autocorrelation problem. It is also the better mechanism test, and
monthly expiry is the largest discrete gamma change in the dataset, which links
straight to tests 09 and 10. The structural break found there predicts the
effect should be weak in post-2019 data, which is falsifiable rather than a
fishing expedition.

### Data sources checked

No free source of per-strike historical open interest was found, so the
collector's premise survives. Cboe's free downloads are aggregate volume and
put/call ratios only.

The DoltHub repo `post-no-preference/options` is free to clone, covers
**2019-02-09 to 2026-08-10**, still updating, with per-strike bid, ask, IV and
greeks. It has **no open interest column** so it cannot do GEX, and it is thin:
SPY had 210 rows on 2026-08-10 against 5,736 in the yfinance snapshot. It is
still 7.5 years of free equity implied volatility, which is the input test 04
needed and did not have.

---

## Summary

| # | Test | Outcome |
|---|---|---|
| 11 | GEX snapshot QA | Collector sound; row 1 has a mismatched date offset |

The transferable result is about the method rather than the market. This
engine passed 28 synthetic known-answer tests and then broke five ways on first
contact with a live option chain, because the generator and the engine shared
assumptions that real markets violate. The load-bearing one: both priced
European options, while SPY, QQQ and IWM options are American.

Every script in this repo validates against synthetic data before touching real
data, and that has caught real bugs. This is the first case where the step
passed and was still not enough. **Synthetic validation is necessary and it is
not sufficient.** Where the generator and the engine can share an assumption,
something external has to check it: here it was implied carry recovered from
put-call parity landing near the real policy rate minus the real dividend yield,
which is a number neither the generator nor the engine was told.

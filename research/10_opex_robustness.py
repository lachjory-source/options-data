"""
IS THE POST-EXPIRY VOLATILITY DROP REAL, OR IS IT A FEW BIG DAYS?
=================================================================

WHAT YOU FOUND
    Daily range contracts 4-10% in the five sessions after monthly expiry,
    across five markets, concentrated on the third Friday specifically, with a
    larger effect in triple-witching months. Opposite to the marketed claim.

THE RIVAL EXPLANATION
    US CPI lands around the 10th-13th. FOMC often falls mid-month. Monthly
    expiry is the 15th-21st. So the pre-expiry window is systematically
    EVENT-RICH and the post-expiry window is systematically EVENT-POOR.

    A handful of CPI prints could produce the entire result with no options
    involved. You cannot claim the finding until you rule that out.

FIVE WAYS TO ATTACK IT, none needing an external event calendar

    1. MEDIAN instead of mean. One enormous day cannot move a median of five.
       If the effect survives on medians, it is not a few prints doing it.

    2. TRIMMED mean, dropping the single largest day in each window. Same
       logic, slightly gentler.

    3. EXCLUDE 2020. COVID produced the wildest volatility in decades. If the
       result is driven by a handful of 2020 cycles, it is not a stable effect.

    4. SPLIT THE SAMPLE. 2012-2018 against 2019-2025. A real structural effect
       should appear in both halves. Two independent confirmations beat one.

    5. COUNT THE SIGN. What fraction of individual expiries show contraction?
       If the mean is driven by outliers, the fraction sits near 50%. If most
       cycles individually contract, the effect is broad rather than lumpy.

HOW TO READ THE OUTPUT
    Effect survives on medians AND in both halves  -> the finding is real
    Effect dies on medians                         -> it was a few event days
    Effect only in one half                        -> not stable, not usable

HOW TO RUN
    Put  !pip install dukascopy-python -q  as the FIRST LINE of this cell.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, date

import dukascopy_python as dk
from dukascopy_python.instruments import (
    INSTRUMENT_IDX_AMERICA_E_SANDP_500,
    INSTRUMENT_IDX_EUROPE_E_DAAX,
    INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    INSTRUMENT_IDX_ASIA_E_N225JAP,
    INSTRUMENT_IDX_ASIA_E_XJO_ASX,
)

MARKETS = [
    ("S&P 500", INSTRUMENT_IDX_AMERICA_E_SANDP_500),
    ("DAX", INSTRUMENT_IDX_EUROPE_E_DAAX),
    ("FTSE 100", INSTRUMENT_IDX_EUROPE_E_FUTSEE_100),
    ("Nikkei 225", INSTRUMENT_IDX_ASIA_E_N225JAP),
    ("ASX 200", INSTRUMENT_IDX_ASIA_E_XJO_ASX),
]


@dataclass
class Config:
    start: datetime = datetime(2012, 1, 1)
    end: datetime = datetime(2026, 1, 1)
    window: int = 5
    n_boot: int = 1500
    seed: int = 7
    split_year: int = 2019


# =============================================================================
# DATA + CALENDAR
# =============================================================================

def load_daily(inst, cfg):
    try:
        d = dk.fetch(inst, dk.INTERVAL_DAY_1, dk.OFFER_SIDE_BID, cfg.start, cfg.end)
    except Exception as e:
        print(f"    fetch failed: {type(e).__name__}")
        return None
    if d is None or len(d) < 500:
        return None
    d = d[["open", "high", "low", "close"]].dropna()
    d = d[(d["high"] >= d["low"]) & (d["close"] > 0)]
    ix = pd.to_datetime(d.index)
    ix = ix.tz_localize(None) if ix.tz is not None else ix
    d.index = ix.normalize()
    d = d[~d.index.duplicated()].sort_index()
    d["range"] = (d["high"] - d["low"]) / d["close"]
    return d.dropna()


def nth_friday(y, m, n):
    dd = date(y, m, 1)
    day = 1 + ((4 - dd.weekday()) % 7) + 7 * (n - 1)
    try:
        return pd.Timestamp(date(y, m, day))
    except ValueError:
        return None


def anchors(cfg, n=3):
    out = []
    for y in range(cfg.start.year, cfg.end.year):
        for m in range(1, 13):
            f = nth_friday(y, m, n)
            if f is not None and cfg.start <= f.to_pydatetime() < cfg.end:
                out.append(f)
    return out


# =============================================================================
# THE MEASUREMENT, WITH SWAPPABLE AGGREGATOR
# =============================================================================

def agg(v, how):
    if how == "mean":
        return float(np.mean(v))
    if how == "median":
        return float(np.median(v))
    if how == "trim":                      # drop the single largest day
        return float(np.mean(np.sort(v)[:-1])) if len(v) > 1 else float(np.mean(v))
    raise ValueError(how)


def ratios(d, anchor_dates, cfg, how="mean", keep=None):
    """log(post/pre) per expiry. `keep` optionally filters anchors by year."""
    idx, v = d.index, d["range"].values
    out, dates = [], []
    for a in anchor_dates:
        if keep is not None and not keep(a):
            continue
        pos = idx.searchsorted(a)
        if pos < cfg.window + 1 or pos + cfg.window + 1 >= len(idx):
            continue
        pre = v[pos - cfg.window:pos]
        post = v[pos + 1:pos + 1 + cfg.window]
        if len(pre) < cfg.window or len(post) < cfg.window:
            continue
        a_, b_ = agg(pre, how), agg(post, how)
        if a_ > 0 and b_ > 0:
            out.append(np.log(b_ / a_))
            dates.append(a)
    return np.array(out), dates


def summ(x):
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def bootstrap(d, cfg, how):
    rng = np.random.default_rng(cfg.seed)
    idx = d.index
    lo, hi = cfg.window + 2, len(idx) - cfg.window - 2
    if hi <= lo:
        return np.array([])
    out = []
    for _ in range(cfg.n_boot):
        pos = rng.integers(lo, hi, 150)
        r, _ = ratios(d, idx[pos], cfg, how)
        if len(r) > 20:
            out.append(r.mean())
    return np.array(out)


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = Config()
    a3 = anchors(cfg, 3)

    print("=" * 80)
    print("STEP 0: DOES THE MEDIAN VERSION STILL DETECT A REAL EFFECT?")
    print("=" * 80)
    print("A robustness check is useless if it is simply less sensitive. So:")
    print("inject a known effect and confirm the median still sees it.\n")
    rng = np.random.default_rng(0)
    n = 3500
    ix = pd.bdate_range("2012-01-02", periods=n)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    f = pd.DataFrame({"close": px,
                      "high": px * (1 + np.abs(rng.normal(0, .004, n))),
                      "low": px * (1 - np.abs(rng.normal(0, .004, n)))}, index=ix)
    f["range"] = (f["high"] - f["low"]) / f["close"]
    for how in ["mean", "median", "trim"]:
        r0, _ = ratios(f, a3, cfg, how)
        m0, t0, _ = summ(r0)
        g = f.copy()
        for a in a3:
            p = g.index.searchsorted(a)
            if 0 < p < len(g) - 6:
                g.iloc[p + 1:p + 6, g.columns.get_loc("range")] *= 0.80
        r1, _ = ratios(g, a3, cfg, how)
        m1, t1, _ = summ(r1)
        print(f"  {how:>7}: null {m0:+.4f} (t {t0:+5.2f})   "
              f"with -20% injected {m1:+.4f} (t {t1:+6.2f})   "
              f"target {np.log(0.8):+.4f}")

    rows = []
    for name, inst in MARKETS:
        print(f"\n\n{'=' * 80}\n{name}\n{'=' * 80}")
        d = load_daily(inst, cfg)
        if d is None:
            print("  no data")
            continue
        print(f"  {len(d)} daily bars, {d.index.min().date()} to {d.index.max().date()}\n")

        res = {}
        print(f"  {'aggregator':>12} {'log ratio':>11} {'as %':>8} {'t':>7} {'n':>5}")
        print("  " + "-" * 46)
        for how in ["mean", "median", "trim"]:
            r, _ = ratios(d, a3, cfg, how)
            m, t, nn = summ(r)
            res[how] = (m, t)
            print(f"  {how:>12} {m:>11.4f} {np.exp(m) - 1:>7.1%} {t:>7.2f} {nn:>5}")

        # --- how broad is it? ---
        rmed, _ = ratios(d, a3, cfg, "median")
        frac = (rmed < 0).mean() * 100
        print(f"\n  expiries where volatility fell (median basis): {frac:.1f}%")
        print(f"    (50% means the average is driven by outliers,")
        print(f"     well above 50% means the effect is broad)")

        # --- exclude 2020 ---
        r20, _ = ratios(d, a3, cfg, "median", keep=lambda a: a.year != 2020)
        m20, t20, n20 = summ(r20)
        print(f"\n  excluding 2020:      {m20:+.4f}  t {t20:+.2f}  n {n20}")

        # --- split sample ---
        early, _ = ratios(d, a3, cfg, "median", keep=lambda a: a.year < cfg.split_year)
        late, _ = ratios(d, a3, cfg, "median", keep=lambda a: a.year >= cfg.split_year)
        me, te, ne = summ(early)
        ml, tl, nl = summ(late)
        print(f"  {cfg.start.year}-{cfg.split_year - 1}:           "
              f"{me:+.4f}  t {te:+.2f}  n {ne}")
        print(f"  {cfg.split_year}-{cfg.end.year - 1}:           "
              f"{ml:+.4f}  t {tl:+.2f}  n {nl}")
        agree = (np.isfinite(me) and np.isfinite(ml) and me < 0 and ml < 0)
        print(f"  both halves negative: {agree}")

        # --- bootstrap on the median version ---
        nb = bootstrap(d, cfg, "median")
        pct = (nb < res["median"][0]).mean() * 100 if len(nb) > 50 else np.nan
        if len(nb) > 50:
            print(f"\n  bootstrap null (median): mean {nb.mean():+.4f}, "
                  f"5th-95th {np.percentile(nb, 5):+.4f} to {np.percentile(nb, 95):+.4f}")
            print(f"  real expiry at the {pct:.1f}th percentile")

        rows.append(dict(name=name, mean=res["mean"][0], med=res["median"][0],
                         t_med=res["median"][1], trim=res["trim"][0],
                         frac=frac, ex20=m20, early=me, late=ml,
                         agree=agree, pct=pct))

    if not rows:
        return
    print("\n\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(f"  {'market':>12} {'mean':>9} {'median':>9} {'t(med)':>8} {'trim':>9} "
          f"{'ex2020':>9} {'early':>9} {'late':>9} {'%down':>7}")
    print("  " + "-" * 86)
    for x in rows:
        print(f"  {x['name']:>12} {x['mean']:>9.4f} {x['med']:>9.4f} "
              f"{x['t_med']:>8.2f} {x['trim']:>9.4f} {x['ex20']:>9.4f} "
              f"{x['early']:>9.4f} {x['late']:>9.4f} {x['frac']:>6.0f}%")

    surv = sum(1 for x in rows if x["med"] < 0 and x["t_med"] < -2)
    both = sum(1 for x in rows if x["agree"])
    broad = sum(1 for x in rows if x["frac"] > 55)
    boot = sum(1 for x in rows if np.isfinite(x["pct"]) and x["pct"] <= 5)

    print(f"\n  significant on MEDIAN (t < -2)   : {surv} of {len(rows)}")
    print(f"  negative in BOTH sample halves   : {both} of {len(rows)}")
    print(f"  majority of expiries contract    : {broad} of {len(rows)}")
    print(f"  below 5th pct of median bootstrap: {boot} of {len(rows)}")

    print("\n" + "=" * 80)
    if surv >= 3 and both >= 3 and broad >= 3:
        print("  THE FINDING SURVIVES.")
        print("  It is not a handful of CPI prints. Volatility genuinely")
        print("  contracts after monthly expiry, broadly and in both halves of")
        print("  the sample, in the opposite direction to what is marketed.")
        print("  That is a real, documented, publishable result.")
    elif surv >= 2 or both >= 3:
        print("  PARTIAL. Weaker on medians than on means, which means large")
        print("  days are doing some of the work. Real but overstated by the")
        print("  original test. Report it with this caveat attached.")
    else:
        print("  THE FINDING DOES NOT SURVIVE.")
        print("  On medians it largely disappears, so the original result was")
        print("  driven by a small number of high-volatility days sitting in")
        print("  the pre-expiry window. That is the macro calendar, not gamma.")
        print()
        print("  Worth knowing you nearly published the wrong conclusion, and")
        print("  that one ten-minute robustness check caught it.")
    print("=" * 80)


main()

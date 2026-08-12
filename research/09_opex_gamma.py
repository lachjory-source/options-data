"""
TESTING THE GAMMA MECHANISM WITHOUT BUYING ANY OPTIONS DATA
===========================================================

THE CLAIM
    Dealers hold a large gamma position built from open option interest. Long
    gamma means they sell rallies and buy dips, which SUPPRESSES volatility.
    At monthly expiry that position rolls off. With less gamma to absorb
    moves, volatility should EXPAND in the days after.

    SpotGamma calls it "OPEX un-pinning": sideways before, expansive after.
    They present it as fact with no statistic attached, so let's attach one.

WHY THIS IS TESTABLE FOR FREE
    Monthly equity option expiry is the third Friday. That is a calendar you
    can compute exactly. You need no open interest, no implied vol surface,
    no dealer positioning model. Just daily prices and a date.

    If gamma roll-off genuinely moves realized volatility, it must show up
    here. If it does not, the framework is in trouble before you spend a cent.

THE THREE CONTROLS, WHICH ARE THE WHOLE POINT
    1. PLACEBO FRIDAYS. Run the identical test anchored on the 1st, 2nd and
       4th Friday of each month. Only the 3rd is expiry. If the effect shows
       up on every Friday, it is a week-of-month artefact, not gamma.

    2. BOOTSTRAP NULL. Draw 2000 random anchor dates and measure the same
       before/after ratio. That builds the distribution of what this test
       produces from nothing, and tells you where the real number sits in it.

    3. FIVE MARKETS. A single-market result is worth almost nothing, as you
       have now seen twice tonight.

AND A SHARPER PREDICTION
    Triple-witching months (March, June, September, December) expire index
    futures and index options as well, so far more gamma rolls off. The
    effect should be LARGER in those months. A mechanism predicts a gradient.
    A coincidence does not.

HOW TO RUN
    Put  !pip install dukascopy-python -q  as the FIRST LINE of this cell.
    Uses DAILY bars, so the download is quick.
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
    start: datetime = datetime(2012, 1, 1)     # daily bars are cheap, go long
    end: datetime = datetime(2026, 1, 1)
    window: int = 5                            # trading days each side
    n_boot: int = 2000
    seed: int = 7


# =============================================================================
# DATA
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
    # dukascopy returns a tz-aware UTC index, but be defensive: tz_localize(None)
    # raises on a naive index and tz_convert raises on an aware one.
    ix = pd.to_datetime(d.index)
    ix = ix.tz_localize(None) if ix.tz is not None else ix
    d.index = ix.normalize()
    d = d[~d.index.duplicated()].sort_index()
    d["ret"] = d["close"].pct_change()
    d["absret"] = d["ret"].abs()
    d["range"] = (d["high"] - d["low"]) / d["close"]
    return d.dropna()


# =============================================================================
# CALENDAR
# =============================================================================

def nth_friday(y, m, n):
    """nth Friday of a month. n=3 is standard monthly option expiry."""
    d = date(y, m, 1)
    offset = (4 - d.weekday()) % 7          # 4 = Friday
    day = 1 + offset + 7 * (n - 1)
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
# THE MEASUREMENT
# =============================================================================

def ratios(d, anchor_dates, cfg, col="range"):
    """
    For each anchor: mean of `col` over the 5 trading days BEFORE it, and the
    5 trading days AFTER it. The anchor day itself is excluded because expiry
    day has its own mechanics.

    Returns log(post/pre). Logs so the ratios are symmetric and additive.
    """
    idx = d.index
    v = d[col].values
    out = []
    for a in anchor_dates:
        pos = idx.searchsorted(a)
        if pos < cfg.window + 1 or pos + cfg.window + 1 >= len(idx):
            continue
        pre = v[pos - cfg.window:pos]
        post = v[pos + 1:pos + 1 + cfg.window]
        if len(pre) < cfg.window or len(post) < cfg.window:
            continue
        a_, b_ = pre.mean(), post.mean()
        if a_ > 0 and b_ > 0:
            out.append(np.log(b_ / a_))
    return np.array(out)


def summarise(x):
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return x.mean(), t, len(x)


def bootstrap(d, cfg, col="range"):
    """Null distribution: what does this test produce from random dates?"""
    rng = np.random.default_rng(cfg.seed)
    idx = d.index
    lo, hi = cfg.window + 2, len(idx) - cfg.window - 2
    if hi <= lo:
        return np.array([])
    n_anchor = 150
    out = []
    for _ in range(cfg.n_boot):
        pos = rng.integers(lo, hi, n_anchor)
        r = ratios(d, idx[pos], cfg, col)
        if len(r) > 20:
            out.append(r.mean())
    return np.array(out)


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = Config()

    # ---- STEP 0: validation ----
    print("=" * 78)
    print("STEP 0: VALIDATION")
    print("=" * 78)
    print("Synthetic daily data with NO calendar effect. The measured")
    print("post/pre ratio must come out at zero in logs.\n")
    rng = np.random.default_rng(0)
    n = 3500
    idx = pd.bdate_range("2012-01-02", periods=n)
    r = rng.normal(0, 0.01, n)
    px = 100 * np.exp(np.cumsum(r))
    fake = pd.DataFrame({"close": px,
                         "high": px * (1 + np.abs(rng.normal(0, .004, n))),
                         "low": px * (1 - np.abs(rng.normal(0, .004, n)))},
                        index=idx)
    fake["ret"] = fake["close"].pct_change()
    fake["absret"] = fake["ret"].abs()
    fake["range"] = (fake["high"] - fake["low"]) / fake["close"]
    fake = fake.dropna()
    m0, t0, n0 = summarise(ratios(fake, anchors(cfg, 3), cfg, "range"))
    print(f"  no effect: log ratio {m0:+.4f}  t {t0:+.2f}  n {n0}  "
          f"-> {'PASS' if abs(t0) < 2.5 else 'FAIL'}")

    # inject a real effect to confirm the test can see one
    boost = fake.copy()
    for a in anchors(cfg, 3):
        pos = boost.index.searchsorted(a)
        if 0 < pos < len(boost) - 6:
            boost.iloc[pos + 1:pos + 6, boost.columns.get_loc("range")] *= 1.25
    m1, t1, _ = summarise(ratios(boost, anchors(cfg, 3), cfg, "range"))
    print(f"  +25% injected after expiry: log ratio {m1:+.4f}  t {t1:+.2f}  "
          f"-> {'PASS' if t1 > 3 else 'FAIL'}   (log 1.25 = {np.log(1.25):+.4f})")

    # ---- the real thing ----
    rows = []
    for name, inst in MARKETS:
        print(f"\n\n{'=' * 78}\n{name}\n{'=' * 78}")
        d = load_daily(inst, cfg)
        if d is None:
            print("  no data")
            continue
        print(f"  {len(d)} daily bars, {d.index.min().date()} to {d.index.max().date()}")

        for col, lbl in [("range", "daily range"), ("absret", "abs return")]:
            r3 = ratios(d, anchors(cfg, 3), cfg, col)
            m, t, nn = summarise(r3)
            print(f"\n  {lbl.upper()}   post-expiry vs pre-expiry")
            print(f"    log ratio {m:+.4f}  ({np.exp(m) - 1:+.1%})   "
                  f"t {t:+.2f}   n {nn} expiries")

            if col == "range":
                # placebo Fridays
                print(f"    placebo anchors:")
                pl = {}
                for k in (1, 2, 4):
                    rk = ratios(d, anchors(cfg, k), cfg, col)
                    mk, tk, _ = summarise(rk)
                    pl[k] = mk
                    print(f"      {k}{'st' if k == 1 else 'nd' if k == 2 else 'th'} "
                          f"Friday: {mk:+.4f}  t {tk:+.2f}")
                better = sum(1 for v in pl.values() if v > m)
                print(f"    3rd Friday ranks {better + 1} of 4")

                # bootstrap null
                nb = bootstrap(d, cfg, col)
                if len(nb) > 100:
                    pct = (nb < m).mean() * 100
                    print(f"    bootstrap null: mean {nb.mean():+.4f}, "
                          f"5th-95th {np.percentile(nb, 5):+.4f} to "
                          f"{np.percentile(nb, 95):+.4f}")
                    print(f"    real expiry sits at the {pct:.1f}th percentile")
                else:
                    pct = np.nan

                # triple witching gradient
                tw, reg = [], []
                for a in anchors(cfg, 3):
                    rr = ratios(d, [a], cfg, col)
                    if len(rr):
                        (tw if a.month in (3, 6, 9, 12) else reg).append(rr[0])
                mt, tt_, _ = summarise(np.array(tw))
                mr, tr_, _ = summarise(np.array(reg))
                print(f"    triple-witching months: {mt:+.4f} (n={len(tw)})")
                print(f"    other months:           {mr:+.4f} (n={len(reg)})")
                print(f"    gradient {mt - mr:+.4f}   (mechanism predicts positive)")

                rows.append(dict(name=name, m=m, t=t, n=nn, rank=better + 1,
                                 pct=pct, grad=mt - mr))

    if not rows:
        return
    print("\n\n" + "=" * 78)
    print("CROSS-MARKET SUMMARY  (daily range, post vs pre expiry)")
    print("=" * 78)
    print(f"  {'market':>12} {'log ratio':>11} {'as %':>8} {'t':>7} "
          f"{'Fri rank':>9} {'boot pct':>10} {'TW gradient':>12}")
    print("  " + "-" * 74)
    for x in rows:
        print(f"  {x['name']:>12} {x['m']:>11.4f} {np.exp(x['m']) - 1:>7.1%} "
              f"{x['t']:>7.2f} {str(x['rank']) + '/4':>9} {x['pct']:>9.1f}% "
              f"{x['grad']:>12.4f}")

    pos = sum(1 for x in rows if x["m"] > 0)
    sig = sum(1 for x in rows if x["t"] > 2)
    top = sum(1 for x in rows if x["rank"] == 1)
    boot = sum(1 for x in rows if x["pct"] >= 95)
    grad = sum(1 for x in rows if x["grad"] > 0)

    print(f"\n  volatility higher after expiry : {pos} of {len(rows)}")
    print(f"  significant (t > 2)            : {sig} of {len(rows)}")
    print(f"  beats all 3 placebo Fridays    : {top} of {len(rows)}")
    print(f"  above 95th pct of bootstrap    : {boot} of {len(rows)}")
    print(f"  triple-witching gradient +ve   : {grad} of {len(rows)}")

    print("\n" + "=" * 78)
    if pos >= 4 and top >= 3 and boot >= 3:
        print("  THE MECHANISM SURVIVES ITS FIRST REAL TEST.")
        print("  Volatility expands after expiry, specifically at the 3rd Friday")
        print("  and not other Fridays, beyond what random dates produce.")
        print("  That justifies spending money on real open-interest data.")
    elif pos >= 4 and (top >= 3 or boot >= 3):
        print("  PARTIAL. The direction is consistent but at least one control")
        print("  is unconvinced. Worth a second look, not worth a subscription.")
    else:
        print("  THE MECHANISM FAILS ITS CHEAPEST TEST.")
        print("  Gamma roll-off is the one claim you can check with a calendar")
        print("  and free prices. If volatility does not expand after expiry,")
        print("  the walls, flip levels and regime tags rest on an effect that")
        print("  does not show up where it must. Do not buy the data.")
    print("=" * 78)


main()

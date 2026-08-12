"""
REPLICATING A JOURNAL OF FINANCIAL ECONOMICS FINDING
=====================================================

THE PAPER
    Baltussen, Da, Lammers & Martens (2021), "Hedging demand and market
    intraday momentum", Journal of Financial Economics.

    Over 60 futures across equities, bonds, commodities and currencies,
    1974-2020: the return in the LAST 30 MINUTES before the close is
    positively predicted by the return over the REST OF THE DAY.

    Their explanation is mechanical, not behavioural. Option market makers who
    are short gamma must buy as price rises and sell as it falls to stay delta
    neutral. Leveraged ETFs must rebalance in the direction of the day's move,
    at the close. Both are forced to trade regardless of price, and both push
    the last half hour in the same direction the day has already gone.

    They show the effect appears when aggregate dealer gamma is NEGATIVE and
    strengthens as it becomes more negative. It reverts over following days.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE YOU HAVE TESTED
    - The hypothesis is pre-registered by someone else, years ago, in a top
      journal. You cannot accidentally fit it to your data.
    - It has a mechanism with a named, forced participant.
    - Daily observations, so no overlapping-window inflation.
    - It needs PRICE DATA ONLY. No option chains, no paid feeds.

WHAT THIS SCRIPT RUNS
    0. Validation on a random walk. Beta must come out ~0.
    1. The headline regression, per market.
    2. THE CONTROL: run the same regression on every 30-minute window of the
       day. If the rest-of-day return predicts EVERY window equally, this is
       generic autocorrelation, not a closing-hour hedging effect. The last
       window has to stand out.
    3. The conditional: split by trailing volatility. Dealers are more likely
       short gamma in high-vol regimes, so the paper predicts a stronger
       effect there. A gradient is evidence. A flat line is not.
    4. The tradeable version, net of costs. A real statistical relationship
       and a profitable trade are different things.

HOW TO RUN
    Put  !pip install dukascopy-python -q  as the FIRST LINE of this cell.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

import dukascopy_python as dk
from dukascopy_python.instruments import (
    INSTRUMENT_IDX_AMERICA_E_SANDP_500,
    INSTRUMENT_IDX_EUROPE_E_DAAX,
    INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    INSTRUMENT_IDX_ASIA_E_N225JAP,
    INSTRUMENT_IDX_ASIA_E_XJO_ASX,
)

MARKETS = [
    dict(name="S&P 500", inst=INSTRUMENT_IDX_AMERICA_E_SANDP_500,
         tz="America/New_York", open="09:30", close="16:00"),
    dict(name="DAX", inst=INSTRUMENT_IDX_EUROPE_E_DAAX,
         tz="Europe/Berlin", open="09:00", close="17:30"),
    dict(name="FTSE 100", inst=INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
         tz="Europe/London", open="08:00", close="16:30"),
    dict(name="Nikkei 225", inst=INSTRUMENT_IDX_ASIA_E_N225JAP,
         tz="Asia/Tokyo", open="09:00", close="15:00"),
    dict(name="ASX 200", inst=INSTRUMENT_IDX_ASIA_E_XJO_ASX,
         tz="Australia/Sydney", open="10:00", close="16:00"),
]


@dataclass
class Config:
    start: datetime = datetime(2021, 1, 1)
    end: datetime = datetime(2026, 1, 1)
    window_min: int = 30          # the closing window the paper studies
    cost_bps: float = 2.0         # round trip
    min_bars: int = 40            # bars required for a valid session


# =============================================================================
# DATA
# =============================================================================

def load(inst, tz, cfg):
    parts = []
    for y in range(cfg.start.year, cfg.end.year):
        try:
            parts.append(dk.fetch(inst, dk.INTERVAL_MIN_5, dk.OFFER_SIDE_BID,
                                  max(cfg.start, datetime(y, 1, 1)),
                                  min(cfg.end, datetime(y + 1, 1, 1))))
        except Exception:
            pass
    if not parts:
        return None
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    df = df[["open", "high", "low", "close"]].dropna()
    df.index = df.index.tz_convert(tz)
    return df


def sessions(d, mkt, cfg):
    """One row per trading day: the day's return split into 'rest of day' and
    the final window, plus every intermediate window for the control."""
    o = int(mkt["open"][:2]) * 60 + int(mkt["open"][3:])
    c = int(mkt["close"][:2]) * 60 + int(mkt["close"][3:])
    mod = d.index.hour * 60 + d.index.minute
    d = d[(mod >= o) & (mod < c)]
    if len(d) == 0:
        return None
    d = d.copy()
    d["date"] = d.index.normalize()
    d["mod"] = d.index.hour * 60 + d.index.minute

    good = d.groupby("date").size()
    d = d[d["date"].isin(good[good >= cfg.min_bars].index)]
    if len(d) == 0:
        return None

    rows = []
    for day, g in d.groupby("date", sort=True):
        px = g["close"].values
        mm = g["mod"].values
        last = mm[-1]
        cut = last - cfg.window_min
        i = np.searchsorted(mm, cut, side="right") - 1
        if i < 5 or i >= len(px) - 1:
            continue
        rows.append({
            "date": day,
            "r_rest": px[i] / px[0] - 1.0,          # open -> 30 min before close
            "r_last": px[-1] / px[i] - 1.0,          # the final 30 minutes
            "r_first": px[min(5, len(px) - 1)] / px[0] - 1.0,   # first 30 min
            "r_full": px[-1] / px[0] - 1.0,
        })
    t = pd.DataFrame(rows).set_index("date")
    t["vol20"] = t["r_full"].rolling(20).std().shift(1)   # known BEFORE the day
    return t.dropna()


# =============================================================================
# STATS
# =============================================================================

def ols_robust(y, x):
    """Simple regression with White (HC0) standard errors. Daily observations
    are effectively independent here, so no overlap correction is needed, but
    volatility clusters so the errors are heteroskedastic."""
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    S = (X * (e ** 2)[:, None]).T @ X
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    return b[1], (b[1] / se[1] if se[1] > 0 else np.nan), r2


def window_control(d, mkt, cfg):
    """
    THE CONTROL. Regress each 30-minute window of the day on the return from
    the open up to the start of that window. If the closing window is special,
    its beta should be the largest. If every window looks the same, you have
    found ordinary autocorrelation, not a hedging effect.
    """
    o = int(mkt["open"][:2]) * 60 + int(mkt["open"][3:])
    c = int(mkt["close"][:2]) * 60 + int(mkt["close"][3:])
    mod = d.index.hour * 60 + d.index.minute
    dd = d[(mod >= o) & (mod < c)].copy()
    dd["date"] = dd.index.normalize()
    dd["mod"] = dd.index.hour * 60 + dd.index.minute

    edges = list(range(o + cfg.window_min, c + 1, cfg.window_min))
    out = []
    for k, end in enumerate(edges):
        beg = end - cfg.window_min
        if beg <= o:
            continue
        pre, win = [], []
        for _, g in dd.groupby("date", sort=True):
            px, mm = g["close"].values, g["mod"].values
            if len(px) < 10:
                continue
            i0 = np.searchsorted(mm, beg, side="right") - 1
            i1 = np.searchsorted(mm, end, side="right") - 1
            if i0 < 2 or i1 <= i0:
                continue
            pre.append(px[i0] / px[0] - 1.0)
            win.append(px[i1] / px[i0] - 1.0)
        if len(pre) > 200:
            b, t, _ = ols_robust(np.array(win), np.array(pre))
            out.append({"end": end, "beta": b, "t": t, "n": len(pre)})
    res = pd.DataFrame(out)
    if len(res):
        # exactly ONE window is the closing window. An earlier version flagged
        # two, which quietly removed a legitimate comparison from the null set.
        res["is_last"] = res["end"] == res["end"].max()
    return res


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = Config()
    print("=" * 78)
    print("STEP 0: VALIDATION ON A RANDOM WALK")
    print("=" * 78)
    print("No hedging flows exist in a random walk. Beta must be ~0.\n")
    rng = np.random.default_rng(1)
    n = 2000
    fake_rest = rng.normal(0, 0.008, n)
    fake_last = rng.normal(0, 0.003, n)
    b, t, r2 = ols_robust(fake_last, fake_rest)
    print(f"  beta {b:+.4f}   t {t:+.2f}   R2 {r2:.4f}   "
          f"-> {'PASS' if abs(t) < 2.5 else 'FAIL'}")

    results = []
    for mkt in MARKETS:
        print(f"\n\n{'=' * 78}\n{mkt['name']}\n{'=' * 78}")
        d = load(mkt["inst"], mkt["tz"], cfg)
        if d is None or len(d) < 50000:
            print("  insufficient data")
            continue
        t_ = sessions(d, mkt, cfg)
        if t_ is None or len(t_) < 300:
            print("  not enough sessions")
            continue
        print(f"  {len(t_)} sessions, {t_.index.min().date()} to {t_.index.max().date()}")

        # --- 1. the headline regression ---
        b, tt, r2 = ols_robust(t_["r_last"].values, t_["r_rest"].values)
        print(f"\n  HEADLINE  r_last30 = a + b * r_restofday")
        print(f"    beta {b:+.4f}   t {tt:+.2f}   R2 {r2:.4f}")
        print(f"    paper predicts b > 0")

        # first-30-minutes version, the older Gao et al. formulation
        b1, t1, _ = ols_robust(t_["r_last"].values, t_["r_first"].values)
        print(f"    (first 30 min as predictor: beta {b1:+.4f}, t {t1:+.2f})")

        # --- 2. the window control ---
        wc = window_control(d, mkt, cfg)
        if len(wc) > 3:
            last_b = float(wc[wc["is_last"]]["beta"].iloc[-1]) if wc["is_last"].any() \
                else float(wc["beta"].iloc[-1])
            others = wc[~wc["is_last"]]["beta"].values
            rank = 1 + int((others > last_b).sum())
            print(f"\n  CONTROL  same regression on every 30-min window")
            print(f"    closing window beta {last_b:+.4f}")
            print(f"    other windows: mean {others.mean():+.4f}, "
                  f"range {others.min():+.4f} to {others.max():+.4f}")
            print(f"    closing window ranks {rank} of {len(others) + 1}")
        else:
            last_b, rank = np.nan, np.nan

        # --- 3. the volatility conditional ---
        print(f"\n  CONDITIONAL  by trailing 20-day volatility")
        q = pd.qcut(t_["vol20"], 3, labels=["low vol", "mid", "high vol"])
        betas = {}
        for lab in ["low vol", "mid", "high vol"]:
            s = t_[q == lab]
            if len(s) > 100:
                bb, tb, _ = ols_robust(s["r_last"].values, s["r_rest"].values)
                betas[lab] = bb
                print(f"    {lab:>9}: beta {bb:+.4f}   t {tb:+.2f}   n {len(s)}")
        grad = (betas.get("high vol", np.nan) - betas.get("low vol", np.nan))
        print(f"    high minus low: {grad:+.4f}   "
              f"(paper predicts positive)")

        # --- 4. the tradeable version ---
        sig = np.sign(t_["r_rest"].values)
        gross = sig * t_["r_last"].values * 1e4          # in basis points
        net = gross - cfg.cost_bps
        sh = net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std(ddof=1) > 0 else np.nan
        tn = net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))
        print(f"\n  TRADEABLE  hold the last 30 min in the day's direction")
        print(f"    gross {gross.mean():+.2f} bps/day   net {net.mean():+.2f} bps/day")
        print(f"    t {tn:+.2f}   annualised Sharpe {sh:+.2f}   "
              f"costs assumed {cfg.cost_bps} bps")

        results.append(dict(name=mkt["name"], beta=b, t=tt, last_b=last_b,
                            rank=rank, grad=grad, net=net.mean(), sharpe=sh,
                            tn=tn))

    if not results:
        return
    print("\n\n" + "=" * 78)
    print("CROSS-MARKET SUMMARY")
    print("=" * 78)
    print(f"  {'market':>12} {'beta':>8} {'t':>7} {'win rank':>10} "
          f"{'vol grad':>10} {'net bps':>9} {'Sharpe':>8}")
    print("  " + "-" * 68)
    for x in results:
        print(f"  {x['name']:>12} {x['beta']:>8.4f} {x['t']:>7.2f} "
              f"{str(x['rank']):>10} {x['grad']:>10.4f} {x['net']:>9.2f} "
              f"{x['sharpe']:>8.2f}")

    pos = sum(1 for x in results if x["beta"] > 0)
    sig_ = sum(1 for x in results if x["t"] > 2)
    prof = sum(1 for x in results if x["tn"] > 2)
    grad_pos = sum(1 for x in results if x["grad"] > 0)

    print(f"\n  beta positive:            {pos} of {len(results)}")
    print(f"  beta significant (t>2):   {sig_} of {len(results)}")
    print(f"  vol gradient positive:    {grad_pos} of {len(results)}")
    print(f"  net profitable (t>2):     {prof} of {len(results)}")

    print("\n" + "=" * 78)
    print("  HOW TO READ THIS")
    print("=" * 78)
    print("  Beta positive nearly everywhere replicates the paper.")
    print("  The closing window ranking FIRST among all windows supports the")
    print("    hedging story specifically, rather than generic momentum.")
    print("  A positive volatility gradient supports the gamma mechanism.")
    print("  And 'net profitable' is a separate question from all three. A real")
    print("    effect at 1-2 bps a day is a genuine finding and still not a")
    print("    trade once you pay a spread.")
    print("=" * 78)


main()

"""
ORB NOISE FLOOR: HOW MUCH EDGE DOES OPTIMIZATION MANUFACTURE?
=============================================================

Crypto never closes. There is no opening auction, no accumulated overnight
news, no bell. So an "opening range breakout" at 03:00 UTC on Bitcoin has NO
mechanism behind it whatsoever. Any edge you find there is manufactured.

That makes it the perfect measuring instrument.

WHAT THIS DOES
    1. Runs an ORB at all 24 hourly boundaries on BTC, with FIXED default
       parameters. That gives you the raw noise distribution: what a strategy
       with no mechanism looks like before anyone tunes it.

    2. Then runs a parameter sweep at each hour and keeps only the BEST
       result, which is exactly what "I ran parameter optimization" means.

    3. Reports the gap. That gap is how much apparent edge the search itself
       creates out of nothing.

WHY YOU WANT THIS NUMBER
    Next time anyone shows you an optimized ORB backtest, you will know how
    large a result has to be before it beats what optimization produces from
    pure noise. Without that reference point, every equity curve looks good.

THE RULES (deliberately plain)
    - Opening range = first N minutes after the chosen hour
    - Long if price trades above the range high, short if below the range low
    - Entry at the close of the breakout bar
    - Stop at the opposite side of the range
    - Otherwise exit at the end of the session
    - One trade per session, first breakout only
    - Results in R multiples, so a stop-out is -1R

HOW TO RUN
    Colab: New notebook, paste this file into a cell, Shift+Enter.
    Runs on simulated data first (known answer), then real BTC.
"""

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    bar_minutes: int = 5
    months: tuple = tuple(f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13))

    cost_bps: float = 5.0            # round trip, as a share of the stop distance

    # the parameter grid an "optimizer" would search
    grid_range_min: tuple = (15, 30, 60, 120)
    grid_session_h: tuple = (4, 8, 12)

    # the fixed default, used for the un-optimized baseline
    default_range_min: int = 15
    default_session_h: int = 8


BARS_PER_DAY = None      # set from bar_minutes


# =============================================================================
# DATA
# =============================================================================

BINANCE_URL = ("https://data.binance.vision/data/spot/monthly/klines/"
               "{sym}/{iv}/{sym}-{iv}-{month}.zip")
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
              "ignore"]


def load_binance(cfg):
    frames = []
    for m in cfg.months:
        try:
            r = requests.get(BINANCE_URL.format(sym=cfg.symbol, iv=cfg.interval,
                                                month=m), timeout=90)
            if r.status_code != 200:
                print(f"  skip {m}")
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.read(z.namelist()[0]).decode("utf-8")
            hh = raw.lstrip().lower().startswith("open_time")
            frames.append(pd.read_csv(io.StringIO(raw), header=0 if hh else None,
                                      names=None if hh else KLINE_COLS))
            print(f"  {m}: {len(frames[-1])} bars")
        except Exception as e:
            print(f"  {m} failed: {e}")
    if not frames:
        raise RuntimeError("No data.")
    df = pd.concat(frames, ignore_index=True)[
        ["open_time", "high", "low", "close"]].copy()
    for c in ["high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ot = pd.to_numeric(df["open_time"], errors="coerce")
    is_us = ot > 1e15
    t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    if is_us.any():
        t.loc[is_us] = pd.to_datetime(ot[is_us], unit="us", utc=True)
    if (~is_us).any():
        t.loc[~is_us] = pd.to_datetime(ot[~is_us], unit="ms", utc=True)
    df["time"] = t
    return df.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)


def simulate_random(n_days=730, bars_per_day=288, seed=0, vol=0.0016):
    """Pure random walk. No structure at any hour, by construction."""
    rng = np.random.default_rng(seed)
    n = n_days * bars_per_day
    close = 50000 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    wick = np.abs(rng.normal(0, vol * 0.7, n)) * close
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC"),
        "high": close + wick, "low": close - wick, "close": close})


def to_daily_matrix(df, bars_per_day):
    """Reshape into (days, bars_per_day), keeping only complete days, then
    stitch consecutive days together so sessions can cross midnight."""
    d = df.copy()
    d["date"] = d["time"].dt.floor("D")
    counts = d.groupby("date").size()
    good = counts[counts == bars_per_day].index
    d = d[d["date"].isin(good)]
    n_days = len(good)
    H = d["high"].values.reshape(n_days, bars_per_day)
    L = d["low"].values.reshape(n_days, bars_per_day)
    C = d["close"].values.reshape(n_days, bars_per_day)
    # stitch day i with day i+1 so an evening session can run past midnight
    H2 = np.concatenate([H[:-1], H[1:]], axis=1)
    L2 = np.concatenate([L[:-1], L[1:]], axis=1)
    C2 = np.concatenate([C[:-1], C[1:]], axis=1)
    return H2, L2, C2, n_days - 1


# =============================================================================
# THE ORB ITSELF  (fully vectorised across days)
# =============================================================================

def run_orb(H, L, C, hour, range_min, session_h, cfg):
    bpm = 60 // cfg.bar_minutes
    start = hour * bpm
    rb = range_min // cfg.bar_minutes
    sb = session_h * bpm
    if rb >= sb or start + sb > H.shape[1]:
        return np.array([])

    rng_hi = H[:, start:start + rb].max(axis=1)
    rng_lo = L[:, start:start + rb].min(axis=1)

    s0, s1 = start + rb, start + sb
    Hs, Ls, Cs = H[:, s0:s1], L[:, s0:s1], C[:, s0:s1]
    m = Hs.shape[1]
    if m < 2:
        return np.array([])

    long_hit = Hs > rng_hi[:, None]
    short_hit = Ls < rng_lo[:, None]
    i_long = np.where(long_hit.any(axis=1), long_hit.argmax(axis=1), m + 1)
    i_short = np.where(short_hit.any(axis=1), short_hit.argmax(axis=1), m + 1)

    is_long = i_long < i_short
    is_short = i_short < i_long          # ties are dropped: direction ambiguous
    trade = is_long | is_short
    first = np.where(is_long, i_long, i_short)
    first_c = np.clip(first, 0, m - 1)

    entry = np.take_along_axis(Cs, first_c[:, None], axis=1).ravel()
    stop = np.where(is_long, rng_lo, rng_hi)
    risk = np.abs(entry - stop)

    cols = np.arange(m)
    after = cols[None, :] > first_c[:, None]
    min_low = np.where(after, Ls, np.inf).min(axis=1)
    max_high = np.where(after, Hs, -np.inf).max(axis=1)
    stopped = np.where(is_long, min_low <= stop, max_high >= stop)

    exit_px = Cs[:, -1]
    raw = np.where(is_long, exit_px - entry, entry - exit_px)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(stopped, -1.0, raw / np.maximum(risk, 1e-12))
        cost = (entry * cfg.cost_bps / 1e4) / np.maximum(risk, 1e-12)
    r = r - cost

    ok = trade & (risk > 1e-9) & np.isfinite(r)
    return r[ok]


def stats_of(r):
    if len(r) < 30:
        return np.nan, np.nan, len(r)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    return r.mean(), t, len(r)


# =============================================================================
# THE EXPERIMENT
# =============================================================================

def sweep(H, L, C, cfg, label):
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")

    base, best = [], []
    print(f"  {'hour':>5} {'baseline E[R]':>14} {'t':>7} {'n':>6}   "
          f"{'best E[R]':>10} {'t':>7}  {'best params':>16}")
    print("  " + "-" * 72)

    for h in range(24):
        r = run_orb(H, L, C, h, cfg.default_range_min, cfg.default_session_h, cfg)
        e0, t0, n0 = stats_of(r)

        be, bt, bp = -9e9, np.nan, None
        for rm in cfg.grid_range_min:
            for sh in cfg.grid_session_h:
                rr = run_orb(H, L, C, h, rm, sh, cfg)
                e, t, n = stats_of(rr)
                if np.isfinite(e) and e > be:
                    be, bt, bp = e, t, f"{rm}m/{sh}h"
        base.append((e0, t0))
        best.append((be, bt))
        print(f"  {h:>5} {e0:>14.4f} {t0:>7.2f} {n0:>6}   "
              f"{be:>10.4f} {bt:>7.2f}  {str(bp):>16}")

    b_e = np.array([x[0] for x in base])
    b_t = np.array([x[1] for x in base])
    o_e = np.array([x[0] for x in best])
    o_t = np.array([x[1] for x in best])

    print(f"\n  {'':<34}{'baseline':>12}{'optimized':>12}")
    print("  " + "-" * 58)
    print(f"  {'mean expectancy across hours':<34}{np.nanmean(b_e):>12.4f}{np.nanmean(o_e):>12.4f}")
    print(f"  {'best single hour':<34}{np.nanmax(b_e):>12.4f}{np.nanmax(o_e):>12.4f}")
    print(f"  {'highest t-stat':<34}{np.nanmax(b_t):>12.2f}{np.nanmax(o_t):>12.2f}")
    # split the sign, otherwise significantly LOSING hours get counted as hits
    print(f"  {'hours significantly POSITIVE (t>2)':<34}"
          f"{int(np.nansum(b_t > 2)):>12}{int(np.nansum(o_t > 2)):>12}")
    print(f"  {'hours significantly NEGATIVE (t<-2)':<34}"
          f"{int(np.nansum(b_t < -2)):>12}{int(np.nansum(o_t < -2)):>12}")
    print(f"  {'hours showing a profit':<34}"
          f"{int(np.nansum(b_e > 0)):>12}{int(np.nansum(o_e > 0)):>12}")

    lift = np.nanmean(o_e) - np.nanmean(b_e)
    n_combos = len(cfg.grid_range_min) * len(cfg.grid_session_h)
    print(f"\n  OPTIMIZATION LIFT: {lift:+.4f}R per trade, on average, purely")
    print(f"  from picking the best of just {n_combos} parameter combinations.")
    print(f"  A real optimizer searches hundreds. The lift grows with the search.")
    return b_e, o_e, b_t, o_t


def main():
    cfg = Config()
    bpd = 24 * 60 // cfg.bar_minutes

    print("=" * 74)
    print("STEP 0: CALIBRATION ON A PURE RANDOM WALK")
    print("=" * 74)
    print("There is nothing here. Every hour is identical noise. Whatever the")
    print("'optimized' column shows is manufactured entirely by the search.\n")
    H, L, C, n = to_daily_matrix(simulate_random(n_days=730, bars_per_day=bpd), bpd)
    print(f"  {n} sessions per hour")
    sweep(H, L, C, cfg, "RANDOM WALK (no mechanism, by construction)")

    print("\n\n" + "=" * 74)
    print("STEP 1: REAL BITCOIN")
    print("=" * 74)
    print("Crypto never closes, so there is no opening auction at any hour.")
    print("If the numbers here look like the random walk above, that is the")
    print("answer: ORB on a 24/7 market is measuring nothing.\n")
    try:
        df = load_binance(cfg)
    except Exception as e:
        print(f"  Could not load: {e}")
        return
    H, L, C, n = to_daily_matrix(df, bpd)
    print(f"\n  {n} sessions per hour")
    sweep(H, L, C, cfg, f"{cfg.symbol} 5m -- ORB at every hour")

    print("\n" + "=" * 74)
    print("HOW TO USE THIS NUMBER")
    print("=" * 74)
    print("The 'best single hour, optimized' figure is your noise floor. It is")
    print("what a tuned ORB produces on a market with no opening bell.")
    print()
    print("When someone shows you an optimized ORB backtest on SPY, their result")
    print("has to clear that floor before it means anything at all. Most of the")
    print("beautiful curves you will be shown do not.")


main()

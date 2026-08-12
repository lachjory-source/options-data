"""
GATE 1: DOES THE OPENING EFFECT REPLICATE ACROSS MARKETS?
=========================================================

On the S&P 500 the real open ranked 1st of 14 cash-session slots, z = +1.85.
Ranking first out of fourteen happens by chance roughly one time in fourteen.
Suggestive. Not enough to act on.

This runs the IDENTICAL test on five markets with different sessions, different
time zones and different participants. Nothing is tuned, nothing is changed
except the instrument and its cash-session hours.

    Under the null, the real open should rank near the middle in most markets.
    If it ranks at the top in market after market, that is far stronger
    evidence than anything more S&P data could ever give you.

PRE-REGISTERED DECISION RULE, fixed before you run it:
    Fisher combined p < 0.05  ->  proceed to gate 2 (redesign the trade)
    Fisher combined p >= 0.05 ->  STOP. The S&P result was noise.

    Write the result in your ledger either way, including the failure.

AN HONEST CAVEAT ABOUT THE STATISTICS
    Equity indices are correlated. A global risk-off day moves all five. So
    these are NOT five independent experiments, and Fisher's method assumes
    independence. The combined p-value below is therefore OPTIMISTIC.

    Treat p = 0.04 as "interesting, keep going" rather than "established".
    Treat p = 0.30 as decisive in the other direction, because correlation
    cannot rescue a null result, only inflate a positive one.

HOW TO RUN
    Cell 1:  !pip install dukascopy-python
    Cell 2:  this file
    Takes a while. Five markets, five years of 5-minute bars each.
"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import dukascopy_python as dk
from dukascopy_python.instruments import (
    INSTRUMENT_IDX_AMERICA_E_SANDP_500,
    INSTRUMENT_IDX_EUROPE_E_DAAX,
    INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    INSTRUMENT_IDX_ASIA_E_N225JAP,
    INSTRUMENT_IDX_ASIA_E_XJO_ASX,
)


# =============================================================================
# THE FIVE MARKETS
# =============================================================================

MARKETS = [
    dict(name="S&P 500",   inst=INSTRUMENT_IDX_AMERICA_E_SANDP_500,
         tz="America/New_York", open="09:30", close="16:00"),
    dict(name="DAX",       inst=INSTRUMENT_IDX_EUROPE_E_DAAX,
         tz="Europe/Berlin",    open="09:00", close="17:30"),
    dict(name="FTSE 100",  inst=INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
         tz="Europe/London",    open="08:00", close="16:30"),
    dict(name="Nikkei 225", inst=INSTRUMENT_IDX_ASIA_E_N225JAP,
         tz="Asia/Tokyo",       open="09:00", close="15:00"),
    dict(name="ASX 200",   inst=INSTRUMENT_IDX_ASIA_E_XJO_ASX,
         tz="Australia/Sydney", open="10:00", close="16:00"),
]


@dataclass
class Config:
    start: datetime = datetime(2021, 1, 1)
    end: datetime = datetime(2026, 1, 1)
    bar_minutes: int = 5
    cost_bps: float = 2.0
    min_range_pct: float = 0.05
    winsor_R: float = 10.0
    range_min: int = 15
    session_h: int = 8
    min_bars_range: int = 2
    min_bars_scan: int = 12
    min_trades: int = 200


# =============================================================================
# DATA
# =============================================================================

def load(inst, tz, cfg):
    parts = []
    for y in range(cfg.start.year, cfg.end.year):
        a = max(cfg.start, datetime(y, 1, 1))
        b = min(cfg.end, datetime(y + 1, 1, 1))
        try:
            parts.append(dk.fetch(inst, dk.INTERVAL_MIN_5, dk.OFFER_SIDE_BID, a, b))
        except Exception as e:
            print(f"      {y} failed: {type(e).__name__}")
    if not parts:
        return None
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    df = df[["open", "high", "low", "close"]].dropna()
    df = df[df["high"] >= df["low"]]
    df.index = df.index.tz_convert(tz)
    return df


# =============================================================================
# ORB  (identical to the single-market version, nothing tuned)
# =============================================================================

def orb(d, open_minute, cfg):
    idx = d.index
    sess = (idx - pd.Timedelta(minutes=open_minute)).floor("D")
    mins = (idx - (sess + pd.Timedelta(minutes=open_minute))).total_seconds() / 60.0

    in_range = (mins >= 0) & (mins < cfg.range_min)
    in_scan = (mins >= cfg.range_min) & (mins < cfg.session_h * 60)
    if not in_range.any() or not in_scan.any():
        return np.array([])

    hi, lo, cl = d["high"].values, d["low"].values, d["close"].values
    rg = pd.DataFrame({"s": sess[in_range], "h": hi[in_range], "l": lo[in_range]})
    agg = rg.groupby("s").agg(rh=("h", "max"), rl=("l", "min"), nb=("h", "size"))
    agg = agg[agg["nb"] >= cfg.min_bars_range]
    if agg.empty:
        return np.array([])

    sc = pd.DataFrame({"s": sess[in_scan], "h": hi[in_scan],
                       "l": lo[in_scan], "c": cl[in_scan]})
    counts = sc.groupby("s").size()
    sc = sc[sc["s"].isin(counts[counts >= cfg.min_bars_scan].index)]
    sc = sc.join(agg[["rh", "rl"]], on="s", how="inner")
    if sc.empty:
        return np.array([])

    width = (sc["rh"] - sc["rl"]) / sc["rh"] * 100
    sc = sc[width >= cfg.min_range_pct]
    if sc.empty:
        return np.array([])

    sc = sc.reset_index(drop=True)
    sc["pos"] = sc.groupby("s").cumcount()
    big = 10 ** 9
    up = sc.assign(p=np.where(sc["h"] > sc["rh"], sc["pos"], big)).groupby("s")["p"].min()
    dn = sc.assign(p=np.where(sc["l"] < sc["rl"], sc["pos"], big)).groupby("s")["p"].min()

    rev = sc.iloc[::-1]
    sc["mla"] = rev.groupby("s")["l"].cummin().iloc[::-1].groupby(sc["s"]).shift(-1)
    sc["mha"] = rev.groupby("s")["h"].cummax().iloc[::-1].groupby(sc["s"]).shift(-1)
    last_close = sc.groupby("s")["c"].last()

    ent = pd.DataFrame({"up": up, "dn": dn}).join(agg[["rh", "rl"]])
    ent["dir"] = np.where(ent["up"] < ent["dn"], 1, np.where(ent["dn"] < ent["up"], -1, 0))
    ent = ent[(ent["dir"] != 0)]
    ent["bar"] = np.where(ent["dir"] == 1, ent["up"], ent["dn"])
    ent = ent[ent["bar"] < big]
    if ent.empty:
        return np.array([])

    hit = sc.set_index(["s", "pos"]).reindex(
        pd.MultiIndex.from_arrays([ent.index, ent["bar"].astype(int)]))
    entry = hit["c"].values
    stop = np.where(ent["dir"].values == 1, ent["rl"].values, ent["rh"].values)
    risk = np.abs(entry - stop)
    stopped = np.where(ent["dir"].values == 1,
                       np.nan_to_num(hit["mla"].values, nan=np.inf) <= stop,
                       np.nan_to_num(hit["mha"].values, nan=-np.inf) >= stop)
    exit_px = last_close.reindex(ent.index).values
    raw = np.where(ent["dir"].values == 1, exit_px - entry, entry - exit_px)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(stopped, -1.0, raw / np.maximum(risk, 1e-12))
        r = r - (entry * cfg.cost_bps / 1e4) / np.maximum(risk, 1e-12)
    if cfg.winsor_R > 0:
        r = np.clip(r, -cfg.winsor_R, cfg.winsor_R)
    ok = np.isfinite(r) & (risk > 0)
    return r[ok]


# =============================================================================
# STATS
# =============================================================================

def chi2_sf_even(x, df):
    """Exact survival function of chi-square for EVEN degrees of freedom.
    Fisher's statistic always has even df, so no scipy needed."""
    k = df // 2
    s, term = 0.0, 1.0
    for j in range(k):
        if j > 0:
            term *= (x / 2) / j
        s += term
    return math.exp(-x / 2) * s


def test_market(mkt, cfg):
    print(f"\n  {mkt['name']}: loading...")
    d = load(mkt["inst"], mkt["tz"], cfg)
    if d is None or len(d) < 50000:
        print(f"    insufficient data, skipping")
        return None
    print(f"    {len(d)} bars, {d.index.min().date()} to {d.index.max().date()}")

    co = int(mkt["open"][:2]) * 60 + int(mkt["open"][3:])
    cc = int(mkt["close"][:2]) * 60 + int(mkt["close"][3:])
    slots = sorted(set(list(range(0, 24 * 60, 30)) + [co]))
    cash_slots = [s for s in slots if co <= s <= cc]

    vals = {}
    for s in cash_slots:
        r = orb(d, s, cfg)
        if len(r) >= cfg.min_trades:
            vals[s] = float(np.mean(r))
    if co not in vals or len(vals) < 6:
        print(f"    only {len(vals)} usable cash slots, skipping")
        return None

    real = vals[co]
    null = np.array([v for s, v in vals.items() if s != co])
    n_all = len(vals)
    rank = 1 + int((null > real).sum())          # 1 = best
    p = rank / n_all                             # chance of ranking this high
    z = (real - null.mean()) / null.std(ddof=1) if null.std(ddof=1) > 0 else np.nan

    print(f"    real open {real:+.4f}R   null mean {null.mean():+.4f}R   "
          f"rank {rank}/{n_all}   z {z:+.2f}   p {p:.3f}")
    return dict(name=mkt["name"], real=real, null_mean=float(null.mean()),
                rank=rank, n=n_all, z=z, p=p)


def main():
    cfg = Config()
    print("=" * 76)
    print("GATE 1: CROSS-MARKET REPLICATION OF THE OPENING EFFECT")
    print("=" * 76)
    print("Identical test, five markets. Nothing tuned. Decision rule fixed")
    print("before running: Fisher combined p < 0.05 proceeds, otherwise stop.\n")

    out = [x for x in (test_market(m, cfg) for m in MARKETS) if x]
    if len(out) < 3:
        print("\n  Fewer than 3 markets loaded. Cannot run the test.")
        return

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"  {'market':>12} {'real open':>11} {'null mean':>11} {'rank':>8} "
          f"{'z':>7} {'p':>7}")
    print("  " + "-" * 62)
    for x in out:
        print(f"  {x['name']:>12} {x['real']:>11.4f} {x['null_mean']:>11.4f} "
              f"{x['rank']:>4}/{x['n']:<3} {x['z']:>7.2f} {x['p']:>7.3f}")

    stat = -2.0 * sum(math.log(max(x["p"], 1e-12)) for x in out)
    dfree = 2 * len(out)
    combined = chi2_sf_even(stat, dfree)

    n_top = sum(1 for x in out if x["rank"] == 1)
    n_pos = sum(1 for x in out if x["real"] > 0)

    print(f"\n  Markets where the open ranked FIRST: {n_top} of {len(out)}")
    print(f"  Markets where the open was PROFITABLE: {n_pos} of {len(out)}")
    print(f"\n  Fisher combined statistic: {stat:.2f} on {dfree} df")
    print(f"  Combined p-value:          {combined:.4f}")

    print("\n" + "=" * 76)
    if combined < 0.05:
        print("  GATE 1 PASSED. The opening effect replicates across markets.")
        print()
        print("  Now read the 'real open' column. If those numbers are still")
        print("  negative, you have confirmed a REAL effect that this trade")
        print("  structure cannot monetise. Gate 2 is redesigning the trade")
        print("  (wider stops, targets, partial exits), NOT hunting a new signal.")
        print()
        print("  And remember the caveat: these markets are correlated, so this")
        print("  p-value is optimistic. It is a green light to continue, not proof.")
    else:
        print("  GATE 1 FAILED. The opening effect does not replicate.")
        print()
        print("  The S&P result was one market ranking first out of fourteen,")
        print("  which happens by chance about one time in fourteen. It did.")
        print()
        print("  Log it in the ledger and stop here. Everything downstream was")
        print("  conditional on this, and it cost you an evening instead of")
        print("  months of building on a coincidence.")
    print("=" * 76)


main()

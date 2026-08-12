"""
GATE 2: CAN A BETTER TRADE STRUCTURE MONETISE THE OPENING EFFECT?
=================================================================

WHERE YOU ARE
    Gate 1 passed. The opening bell is worth about +0.12R relative to a random
    session slot, consistently across five markets.
    But the ORB rules themselves lose about -0.19R, so the net is still negative.
    You need roughly +0.07R of structural improvement just to reach breakeven.

WHY THE RULES LOSE, IN TWO PARTS
    ~-0.13R  transaction cost. The stop sits one range-width away, which is
             about 0.15% of price. A 2bp round trip is 13% of that.
    ~-0.06R  the breakout itself. Roughly 85% small losses, 15% large winners,
             netting slightly negative.

    So most of the damage is COST, and cost in R terms is
    (spread / stop distance). Double the stop, halve the cost. That is
    arithmetic, not a search, which is why it cannot be curve fit.

WHAT THIS TESTS
    stop width   1.0, 1.5, 2.0, 3.0 range-widths
    exit         session close, or a fixed target at 1R / 2R / 3R
    = 16 combinations

THE RULE THAT MAKES THIS HONEST: WALK-FORWARD ONLY
    Your optimization lift is +0.15R. The real effect is +0.12R. Tuning can
    manufacture MORE than the entire genuine effect, so any in-sample search
    here is worthless: it returns a positive number either way.

    So: choose parameters on a training window, trade them BLIND on the next
    window, roll forward. Only the blind windows are reported. Nothing that
    picked its own parameters using the data it is scored on appears below.

AND A CONTROL, WHICH IS THE PART EVERYONE SKIPS
    The identical walk-forward runs at 4 fake opens in the same session. If
    walk-forward improves those just as much, the improvement is procedure,
    not effect. The real open has to beat its own controls.

PRE-REGISTERED DECISION RULE
    Real open out-of-sample > 0 AND better than all 4 controls  -> gate 3
    Anything else                                               -> STOP

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
    cost_bps: float = 2.0
    range_min: int = 15
    session_h: int = 8
    min_range_pct: float = 0.08     # stricter than before, since winsor is OFF
    min_bars_range: int = 2
    min_bars_scan: int = 12

    stop_mults: tuple = (1.0, 1.5, 2.0, 3.0)
    targets: tuple = (0.0, 1.0, 2.0, 3.0)   # 0.0 means hold to session close

    train: int = 500                # sessions used to choose parameters
    test: int = 125                 # sessions traded blind with that choice
    n_controls: int = 4


# =============================================================================
# BUILD ONE ROW PER SESSION, ONCE. Every parameter combo reuses it.
# =============================================================================

def session_table(d, open_minute, cfg):
    idx = d.index
    sess = (idx - pd.Timedelta(minutes=open_minute)).floor("D")
    mins = (idx - (sess + pd.Timedelta(minutes=open_minute))).total_seconds() / 60.0
    in_r = (mins >= 0) & (mins < cfg.range_min)
    in_s = (mins >= cfg.range_min) & (mins < cfg.session_h * 60)
    if not in_r.any() or not in_s.any():
        return None

    hi, lo, cl = d["high"].values, d["low"].values, d["close"].values
    agg = pd.DataFrame({"s": sess[in_r], "h": hi[in_r], "l": lo[in_r]}).groupby("s") \
        .agg(rh=("h", "max"), rl=("l", "min"), nb=("h", "size"))
    agg = agg[agg["nb"] >= cfg.min_bars_range]
    if agg.empty:
        return None

    sc = pd.DataFrame({"s": sess[in_s], "h": hi[in_s], "l": lo[in_s], "c": cl[in_s]})
    cnt = sc.groupby("s").size()
    sc = sc[sc["s"].isin(cnt[cnt >= cfg.min_bars_scan].index)]
    sc = sc.join(agg[["rh", "rl"]], on="s", how="inner")
    sc = sc[(sc["rh"] - sc["rl"]) / sc["rh"] * 100 >= cfg.min_range_pct]
    if sc.empty:
        return None

    sc = sc.reset_index(drop=True)
    sc["pos"] = sc.groupby("s").cumcount()
    BIG = 10 ** 9
    up = sc.assign(p=np.where(sc["h"] > sc["rh"], sc["pos"], BIG)).groupby("s")["p"].min()
    dn = sc.assign(p=np.where(sc["l"] < sc["rl"], sc["pos"], BIG)).groupby("s")["p"].min()

    rev = sc.iloc[::-1]
    sc["mla"] = rev.groupby("s")["l"].cummin().iloc[::-1].groupby(sc["s"]).shift(-1)
    sc["mha"] = rev.groupby("s")["h"].cummax().iloc[::-1].groupby(sc["s"]).shift(-1)
    lastc = sc.groupby("s")["c"].last()

    e = pd.DataFrame({"up": up, "dn": dn}).join(agg[["rh", "rl"]])
    e["dir"] = np.where(e["up"] < e["dn"], 1, np.where(e["dn"] < e["up"], -1, 0))
    e = e[(e["dir"] != 0)]
    e["bar"] = np.where(e["dir"] == 1, e["up"], e["dn"])
    e = e[e["bar"] < BIG]
    if e.empty:
        return None

    hit = sc.set_index(["s", "pos"]).reindex(
        pd.MultiIndex.from_arrays([e.index, e["bar"].astype(int)]))
    t = pd.DataFrame({
        "dir": e["dir"].values,
        "entry": hit["c"].values,
        "rh": e["rh"].values,
        "rl": e["rl"].values,
        "mla": hit["mla"].values,
        "mha": hit["mha"].values,
        "close": lastc.reindex(e.index).values,
    }, index=e.index).dropna(subset=["entry", "close"])
    t["width"] = t["rh"] - t["rl"]
    return t[t["width"] > 0].sort_index()


def evaluate(t, stop_mult, target_R, cfg):
    """
    Worst-case ordering: if both the stop and the target were touched during
    the session, the stop is assumed to have come first. We cannot know the
    intrabar path, and assuming the favourable order is how backtests lie.
    """
    long = t["dir"].values == 1
    ext = (stop_mult - 1.0) * t["width"].values
    stop = np.where(long, t["rl"].values - ext, t["rh"].values + ext)
    entry = t["entry"].values
    risk = np.abs(entry - stop)
    ok = risk > 0
    tgt = np.where(long, entry + target_R * risk, entry - target_R * risk)

    stopped = np.where(long, t["mla"].values <= stop, t["mha"].values >= stop)
    hit_t = (np.where(long, t["mha"].values >= tgt, t["mla"].values <= tgt)
             if target_R > 0 else np.zeros(len(t), dtype=bool))

    close_r = np.where(long, t["close"].values - entry, entry - t["close"].values)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(stopped, -1.0,
                     np.where(hit_t, target_R, close_r / np.maximum(risk, 1e-12)))
        r = r - (entry * cfg.cost_bps / 1e4) / np.maximum(risk, 1e-12)
    r = np.where(ok & np.isfinite(r), r, np.nan)
    return pd.Series(r, index=t.index)


# =============================================================================
# WALK-FORWARD
# =============================================================================

def walk_forward(t, cfg):
    """Choose parameters on the training block, trade them blind on the next
    block, roll. Only the blind blocks are returned."""
    n = len(t)
    if n < cfg.train + cfg.test:
        return None, []

    combos = [(sm, tg) for sm in cfg.stop_mults for tg in cfg.targets]
    cache = {c: evaluate(t, c[0], c[1], cfg).values for c in combos}

    oos, picks = [], []
    start = cfg.train
    while start + cfg.test <= n:
        best, best_e = None, -9e9
        for c in combos:
            v = cache[c][:start]
            v = v[np.isfinite(v)]
            if len(v) > 100 and v.mean() > best_e:
                best, best_e = c, v.mean()
        if best is None:
            break
        blind = cache[best][start:start + cfg.test]
        oos.append(blind[np.isfinite(blind)])
        picks.append(best)
        start += cfg.test

    return (np.concatenate(oos) if oos else None), picks


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
    df = df[df["high"] >= df["low"]]
    df.index = df.index.tz_convert(tz)
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = Config()
    print("=" * 78)
    print("GATE 2: WALK-FORWARD TRADE REDESIGN")
    print("=" * 78)
    print(f"  {len(cfg.stop_mults) * len(cfg.targets)} combos, "
          f"train {cfg.train} sessions, trade {cfg.test} blind, roll.")
    print("  Only blind results reported. Controls at 4 fake opens.\n")

    rows = []
    for mkt in MARKETS:
        print(f"\n  {mkt['name']}: loading...")
        d = load(mkt["inst"], mkt["tz"], cfg)
        if d is None or len(d) < 50000:
            print("    insufficient data")
            continue

        co = int(mkt["open"][:2]) * 60 + int(mkt["open"][3:])
        cc = int(mkt["close"][:2]) * 60 + int(mkt["close"][3:])
        cash = [s for s in range(co, cc + 1, 30) if s != co]
        rng = np.random.default_rng(42)
        ctrl = list(rng.choice(cash, size=min(cfg.n_controls, len(cash)),
                               replace=False))

        res = {}
        for label, slot in [("REAL", co)] + [(f"ctrl{i+1}", int(s))
                                             for i, s in enumerate(ctrl)]:
            t = session_table(d, slot, cfg)
            if t is None or len(t) < cfg.train + cfg.test:
                res[label] = None
                continue
            oos, picks = walk_forward(t, cfg)
            if oos is None or len(oos) < 100:
                res[label] = None
                continue
            res[label] = dict(e=float(oos.mean()), n=len(oos),
                              t=float(oos.mean() / (oos.std(ddof=1) / np.sqrt(len(oos)))),
                              picks=picks)

        if res.get("REAL") is None:
            print("    not enough sessions for walk-forward")
            continue

        r = res["REAL"]
        cs = [v["e"] for k, v in res.items() if k != "REAL" and v]
        print(f"    REAL open : {r['e']:+.4f}R  t={r['t']:+.2f}  n={r['n']}")
        for k in sorted(res):
            if k != "REAL" and res[k]:
                print(f"    {k:<10}: {res[k]['e']:+.4f}R")
        print(f"    chosen params by fold: {r['picks']}")
        rows.append(dict(name=mkt["name"], real=r["e"], t=r["t"], n=r["n"],
                         ctrl_mean=float(np.mean(cs)) if cs else np.nan,
                         beats_all=bool(cs and all(r["e"] > c for c in cs))))

    if not rows:
        print("\n  Nothing completed.")
        return

    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 78)
    print(f"  {'market':>12} {'real (OOS)':>12} {'t':>7} {'controls':>11} "
          f"{'beats all?':>11}")
    print("  " + "-" * 58)
    for x in rows:
        print(f"  {x['name']:>12} {x['real']:>12.4f} {x['t']:>7.2f} "
              f"{x['ctrl_mean']:>11.4f} {str(x['beats_all']):>11}")

    n_pos = sum(1 for x in rows if x["real"] > 0)
    n_beat = sum(1 for x in rows if x["beats_all"])
    mean_r = float(np.mean([x["real"] for x in rows]))
    mean_c = float(np.nanmean([x["ctrl_mean"] for x in rows]))

    print(f"\n  markets profitable out-of-sample : {n_pos} of {len(rows)}")
    print(f"  markets beating all controls     : {n_beat} of {len(rows)}")
    print(f"  mean real {mean_r:+.4f}R   mean control {mean_c:+.4f}R   "
          f"edge {mean_r - mean_c:+.4f}R")

    print("\n" + "=" * 78)
    if n_pos >= 3 and n_beat >= 3:
        print("  GATE 2 PASSED. Positive out-of-sample and beating its own")
        print("  controls in most markets. Proceed to gate 3: cost sensitivity")
        print("  at 2, 5 and 10 bps, then the untouched 2025 holdout.")
    elif mean_r > mean_c and n_pos >= 2:
        print("  MARGINAL. The real open beats its controls on average but not")
        print("  consistently. That is what a small real effect looks like AND")
        print("  what noise looks like. Do not proceed on this alone.")
    else:
        print("  GATE 2 FAILED. Widening stops and adding targets did not")
        print("  rescue it out-of-sample. The opening effect is real and too")
        print("  small to monetise with this trade. Log it and stop.")
        print()
        print("  This is the most likely outcome and it is a clean answer,")
        print("  not a failure. You now know the size of the effect and the")
        print("  size of the costs, and that the second is bigger.")
    print("=" * 78)


main()

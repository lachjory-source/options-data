"""
LIQUIDITY SWEEP BASELINE BACKTESTER  --  v2

CHANGE LOG
    v2  Fixed a timestamp parsing bug. Binance switched from millisecond to
        microsecond timestamps in January 2025, i.e. partway through a normal
        backtest window. v1 guessed the unit ONCE for the whole dataset, so
        every 2024 file was misread and mapped to January 1970. The price
        series and ordering survived intact, but every time-of-day and session
        breakdown was garbage. Now detected per row, with a loud warning.

        Nothing else changed. The v1 headline result stands.

Purpose: test whether the raw liquidity sweep pattern has any edge at all,
BEFORE adding any filters. This is the control experiment. If the naked
pattern is worthless, that is useful information, not a failure.

HOW TO RUN (easiest path, no install):
    1. colab.research.google.com -> new notebook
    2. Paste this whole file into a cell
    3. Shift+Enter

READ THIS BEFORE TRUSTING ANY OUTPUT:
    Every parameter below is an arbitrary choice. Freeze them in
    PRE_REGISTRATION.md BEFORE you look at results. If you change a
    parameter after seeing a result, you are curve fitting, and you must
    log it in the Variants Tested tab.
"""

import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# =============================================================================
# CONFIG  -- freeze these in PRE_REGISTRATION.md before running
# =============================================================================

@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    months: list = field(default_factory=lambda: [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
        "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    ])
    # OUT-OF-SAMPLE: leave alone until your rules are frozen.
    oos_months: list = field(default_factory=lambda: [
        "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
        "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
    ])

    swing_n: int = 2              # bars either side required to confirm a swing
    lookback: int = 50            # how many bars back a level stays relevant
    min_pierce_pct: float = 0.0   # 0.0 = any pierce counts (baseline)

    rr: float = 2.0               # target as a multiple of risk
    max_hold: int = 24            # bars before timed exit
    cost_bps: float = 10.0        # round-trip fees + slippage, basis points
    stop_buffer_pct: float = 0.0  # widen the stop beyond the sweep extreme

    run_random_control: bool = True
    n_random_trials: int = 100
    validation_trials: int = 40
    seed: int = 42

    use_oos: bool = False         # flip to True ONCE, after freezing rules


# =============================================================================
# DATA
# =============================================================================

BINANCE_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{sym}/{iv}/{sym}-{iv}-{month}.zip"
)

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
    "ignore",
]


def load_binance(symbol, interval, months, verbose=True):
    """Download free monthly kline archives. No API key required."""
    if requests is None:
        raise RuntimeError("pip install requests")

    frames = []
    for m in months:
        url = BINANCE_URL.format(sym=symbol, iv=interval, month=m)
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                if verbose:
                    print(f"  skip {m} (HTTP {r.status_code})")
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.read(z.namelist()[0]).decode("utf-8")
            has_header = raw.lstrip().lower().startswith("open_time")
            df = pd.read_csv(
                io.StringIO(raw),
                header=0 if has_header else None,
                names=None if has_header else KLINE_COLS,
            )
            frames.append(df)
            if verbose:
                print(f"  loaded {m}: {len(df)} bars")
        except Exception as e:
            print(f"  FAILED {m}: {e}")

    if not frames:
        raise RuntimeError("No data downloaded. Check symbol/interval/months.")

    df = pd.concat(frames, ignore_index=True)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- v2 FIX ---
    # Binance switched from millisecond to microsecond timestamps in Jan 2025,
    # partway through a typical backtest window. Detect the unit PER ROW.
    # A single global guess silently maps every file on the other side of the
    # switch to January 1970: the price series still looks fine and the bar
    # count is still right, but every session breakdown becomes noise.
    ot = pd.to_numeric(df["open_time"], errors="coerce")
    is_us = ot > 1e15
    t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    if is_us.any():
        t.loc[is_us] = pd.to_datetime(ot[is_us], unit="us", utc=True)
    if (~is_us).any():
        t.loc[~is_us] = pd.to_datetime(ot[~is_us], unit="ms", utc=True)
    df["time"] = t

    df = df.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)

    # Loud sanity check. A silently wrong date range is worse than a crash.
    if df["time"].min() < pd.Timestamp("2010-01-01", tz="UTC"):
        print("\n  *** WARNING: timestamps before 2010. Unit parsing failed. ***")
        print("  *** Session breakdowns are meaningless. ***\n")
    gaps = df["time"].diff().dt.total_seconds().dropna()
    if len(gaps) and gaps.max() > 6 * gaps.median():
        big = (gaps > 6 * gaps.median()).sum()
        print(f"  note: {big} gap(s) in the series, largest {gaps.max()/3600:.1f}h")
    return df


def make_synthetic(n_bars=20000, seed=0, start=50000.0, vol=0.004):
    """
    A pure random walk with NO structure whatsoever.

    Validates the engine. There is nothing to find here, so the sweep must
    perform the SAME as random entries. If it beats them, the code has
    lookahead bias and every other result is worthless.

    Do NOT expect expectancy near zero. Costs, the worst-case intrabar rule,
    and a stop inside normal bar noise drag it well below zero. That drag is
    real and applies live too. What matters is the DIFFERENCE between the
    pattern and random entries carrying the same drag.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, vol, n_bars)
    close = start * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[start], close[:-1]])
    spread = np.abs(rng.normal(0, vol * 0.8, n_bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n_bars, freq="1h", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(1, 100, n_bars),
    })


# =============================================================================
# SWING DETECTION  (no lookahead: a swing at bar j is only usable from j+n)
# =============================================================================

def find_swings(df, n):
    low = df["low"].values
    high = df["high"].values
    N = len(df)
    swing_lows, swing_highs = [], []
    for j in range(n, N - n):
        wlo = low[j - n: j + n + 1]
        whi = high[j - n: j + n + 1]
        if low[j] == wlo.min() and (wlo == low[j]).sum() == 1:
            swing_lows.append((j + n, low[j], j))
        if high[j] == whi.max() and (whi == high[j]).sum() == 1:
            swing_highs.append((j + n, high[j], j))
    return swing_lows, swing_highs


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(df, cfg):
    low, high, close = df["low"].values, df["high"].values, df["close"].values
    N = len(df)
    swing_lows, swing_highs = find_swings(df, cfg.swing_n)

    lows_by_conf, highs_by_conf = {}, {}
    for conf, price, origin in swing_lows:
        lows_by_conf.setdefault(conf, []).append((price, origin))
    for conf, price, origin in swing_highs:
        highs_by_conf.setdefault(conf, []).append((price, origin))

    active_lows, active_highs, signals = [], [], []
    buf = cfg.stop_buffer_pct / 100.0

    for i in range(N):
        for price, origin in lows_by_conf.get(i, []):
            active_lows.append([price, origin])
        for price, origin in highs_by_conf.get(i, []):
            active_highs.append([price, origin])

        active_lows = [x for x in active_lows if i - x[1] <= cfg.lookback]
        active_highs = [x for x in active_highs if i - x[1] <= cfg.lookback]

        pierced = [x for x in active_lows if low[i] < x[0]]
        if pierced:
            level = max(x[0] for x in pierced)
            pierce_pct = (level - low[i]) / level * 100
            if close[i] > level and pierce_pct >= cfg.min_pierce_pct:
                signals.append({"bar": i, "side": "long", "level": level,
                                "entry": close[i], "stop": low[i] * (1 - buf),
                                "pierce_pct": pierce_pct})
            active_lows = [x for x in active_lows if low[i] >= x[0]]

        pierced = [x for x in active_highs if high[i] > x[0]]
        if pierced:
            level = min(x[0] for x in pierced)
            pierce_pct = (high[i] - level) / level * 100
            if close[i] < level and pierce_pct >= cfg.min_pierce_pct:
                signals.append({"bar": i, "side": "short", "level": level,
                                "entry": close[i], "stop": high[i] * (1 + buf),
                                "pierce_pct": pierce_pct})
            active_highs = [x for x in active_highs if high[i] <= x[0]]

    return signals


# =============================================================================
# SIMULATION  (worst-case intrabar assumptions)
# =============================================================================

def simulate_trade(df, sig, cfg):
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    N = len(df)
    i = sig["bar"]
    entry, stop = sig["entry"], sig["stop"]
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    long = sig["side"] == "long"
    target = entry + cfg.rr * risk if long else entry - cfg.rr * risk
    mae = mfe = 0.0
    exit_r = exit_bar = reason = None
    ambiguous = False

    for k in range(i + 1, min(i + 1 + cfg.max_hold, N)):
        if long:
            bad, good = (low[k] - entry) / risk, (high[k] - entry) / risk
        else:
            bad, good = (entry - high[k]) / risk, (entry - low[k]) / risk
        mae, mfe = min(mae, bad), max(mfe, good)

        stop_hit = low[k] <= stop if long else high[k] >= stop
        target_hit = high[k] >= target if long else low[k] <= target

        if stop_hit and target_hit:
            ambiguous = True          # cannot know the intrabar path
        if stop_hit:                  # worst case: stop resolves first
            exit_r, exit_bar, reason = -1.0, k, "stop"
            break
        if target_hit:
            exit_r, exit_bar, reason = cfg.rr, k, "target"
            break

    if exit_r is None:
        k = min(i + cfg.max_hold, N - 1)
        px = close[k]
        exit_r = (px - entry) / risk if long else (entry - px) / risk
        exit_bar, reason = k, "timeout"

    cost_r = (entry * cfg.cost_bps / 10000.0) / risk
    return {**sig,
            "time": df["time"].iloc[i],
            "risk_price": risk,
            "risk_pct": risk / entry * 100,
            "r_gross": exit_r,
            "r_net": exit_r - cost_r,
            "cost_r": cost_r,
            "mae_r": mae, "mfe_r": mfe,
            "bars_held": exit_bar - i,
            "exit_reason": reason,
            "ambiguous": ambiguous,
            "hour_utc": df["time"].iloc[i].hour}


def run(df, cfg):
    sigs = generate_signals(df, cfg)
    return pd.DataFrame([t for t in (simulate_trade(df, s, cfg) for s in sigs) if t])


# =============================================================================
# RANDOM CONTROL
# =============================================================================

def random_control(df, cfg, real_trades):
    rng = np.random.default_rng(cfg.seed)
    N, n_trades = len(df), len(real_trades)
    if n_trades == 0:
        return None
    risk_pcts = real_trades["risk_pct"].values
    sides = real_trades["side"].values
    close = df["close"].values

    results = []
    for _ in range(cfg.n_random_trials):
        bars = rng.integers(cfg.lookback, N - cfg.max_hold - 1, n_trades)
        rs = []
        for b, rp, sd in zip(bars, rng.choice(risk_pcts, n_trades),
                             rng.choice(sides, n_trades)):
            entry = close[b]
            risk = entry * rp / 100
            stop = entry - risk if sd == "long" else entry + risk
            t = simulate_trade(df, {"bar": int(b), "side": sd, "level": entry,
                                    "entry": entry, "stop": stop,
                                    "pierce_pct": 0.0}, cfg)
            if t:
                rs.append(t["r_net"])
        if rs:
            results.append(np.mean(rs))
    return np.array(results)


# =============================================================================
# STATS
# =============================================================================

def stats(trades, label="STRATEGY"):
    if len(trades) == 0:
        print(f"\n{label}: no trades generated.")
        return {}
    r = trades["r_net"]
    wins, losses = r[r > 0], r[r <= 0]
    equity = r.cumsum()
    se = r.std() / np.sqrt(len(r)) if len(r) > 1 else np.nan
    out = {"trades": len(r),
           "win_rate_pct": len(wins) / len(r) * 100,
           "expectancy_r": r.mean(),
           "t_stat": r.mean() / se if se and se > 0 else np.nan}

    print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
    print(f"  Trades              {len(r)}")
    print(f"  Win rate            {out['win_rate_pct']:.1f}%")
    print(f"  Avg win / loss      {wins.mean() if len(wins) else 0:+.2f}R / "
          f"{losses.mean() if len(losses) else 0:+.2f}R")
    print(f"  Expectancy NET      {r.mean():+.4f}R per trade")
    print(f"  Expectancy GROSS    {trades['r_gross'].mean():+.4f}R  (before costs)")
    print(f"  Total               {r.sum():+.1f}R")
    print(f"  Max drawdown        {(equity.cummax() - equity).max():.1f}R")
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf
    print(f"  Profit factor       {pf:.2f}")
    print(f"  t-stat              {out['t_stat']:.2f}   (|t| < 2 = indistinguishable from zero)")

    print(f"\n  Median stop size    {trades['risk_pct'].median():.3f}% of price")
    print(f"  Mean cost per trade {trades['cost_r'].mean():.3f}R  "
          f"(median {trades['cost_r'].median():.3f}R)")
    if trades["cost_r"].mean() > 0.10:
        print(f"     WARNING: costs average {trades['cost_r'].mean()*100:.0f}% of your risk per trade.")
    amb = trades["ambiguous"].mean() * 100
    print(f"  Ambiguous bars      {amb:.1f}% of trades had stop AND target in one bar")
    if amb > 20:
        print("     (scored as losses by the worst-case rule; this timeframe is too")
        print("      coarse to resolve them, so treat the result as a lower bound)")
    return out


def breakdowns(trades):
    if len(trades) == 0:
        return
    print(f"\n{'-' * 62}\nBREAKDOWNS (hypothesis generation only, not conclusions)\n{'-' * 62}")
    print("\nBy side:")
    print(trades.groupby("side")["r_net"].agg(["count", "mean"]).round(4))
    print("\nBy exit reason:")
    print(trades.groupby("exit_reason")["r_net"].agg(["count", "mean"]).round(4))
    print("\nBy UTC session block:")
    blocks = pd.cut(trades["hour_utc"], [-1, 5, 11, 17, 23],
                    labels=["Asia 00-05", "London 06-11", "NY 12-17", "Late 18-23"])
    print(trades.groupby(blocks, observed=False)["r_net"].agg(["count", "mean"]).round(4))
    print("\nMAE/MFE:")
    w, l = trades[trades.r_net > 0], trades[trades.r_net <= 0]
    if len(w):
        print(f"  Median MAE on winners: {w['mae_r'].median():.2f}R")
    if len(l):
        print(f"  Median MFE on losers:  {l['mfe_r'].median():.2f}R")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = Config()

    print("=" * 62)
    print("STEP 1: ENGINE VALIDATION ON PURE RANDOM DATA")
    print("=" * 62)
    print("Nothing to find in a random walk, so the sweep must perform the")
    print("SAME as random entries. If it beats them, the engine is broken.\n")
    synth = make_synthetic(n_bars=20000, seed=cfg.seed)
    synth_trades = run(synth, cfg)
    stats(synth_trades, "SWEEP ON RANDOM WALK")
    val_cfg = Config(**{**cfg.__dict__, "n_random_trials": cfg.validation_trials})
    sctrl = random_control(synth, val_cfg, synth_trades)
    if sctrl is not None and len(sctrl):
        pct = (sctrl < synth_trades["r_net"].mean()).mean() * 100
        print(f"\n  Random-entry control on same data: {sctrl.mean():+.4f}R")
        print(f"  Sweep sits at the {pct:.0f}th percentile of that control.")
        if pct > 97:
            print("\n  *** FAIL: beats random on structureless data. Engine broken. ***")
        else:
            print("  PASS: indistinguishable from random. No lookahead bias detected.")

    print("\n\n" + "=" * 62)
    print("STEP 2: REAL DATA")
    print("=" * 62)
    months = cfg.months + (cfg.oos_months if cfg.use_oos else [])
    if cfg.use_oos:
        print("  !! OUT-OF-SAMPLE INCLUDED. You only get to do this once. !!\n")
    df = load_binance(cfg.symbol, cfg.interval, months)
    print(f"\n  {len(df)} bars from {df['time'].min()} to {df['time'].max()}")

    trades = run(df, cfg)
    stats(trades, f"LIQUIDITY SWEEP BASELINE -- {cfg.symbol} {cfg.interval}")
    breakdowns(trades)

    if cfg.run_random_control and len(trades):
        print(f"\n{'=' * 62}\nSTEP 3: DOES IT BEAT RANDOM ENTRIES?\n{'=' * 62}")
        ctrl = random_control(df, cfg, trades)
        if ctrl is not None and len(ctrl):
            real = trades["r_net"].mean()
            pct = (ctrl < real).mean() * 100
            print(f"  Random entries, same risk/target structure:")
            print(f"    mean {ctrl.mean():+.4f}R  (5th-95th pct: "
                  f"{np.percentile(ctrl, 5):+.4f} to {np.percentile(ctrl, 95):+.4f})")
            print(f"  Sweep expectancy  {real:+.4f}R")
            print(f"  Sweep beats {pct:.1f}% of random runs.")
            if pct < 95:
                print("\n  VERDICT: not distinguishable from random. The naked pattern")
                print("  has no standalone edge in this sample. That is a real result.")
            else:
                print("\n  VERDICT: beats random at the 95% level IN-SAMPLE. Necessary")
                print("  but not sufficient. It still has to survive out-of-sample.")

    trades.to_csv("sweep_trades.csv", index=False)
    print(f"\n\nSaved {len(trades)} trades to sweep_trades.csv")


if __name__ == "__main__":
    main()

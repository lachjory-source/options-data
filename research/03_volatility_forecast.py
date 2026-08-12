"""
REALIZED VOLATILITY FORECASTER  --  v2

WHAT CHANGED FROM v1, and why

1. THE NAIVE BENCHMARK WAS RIGGED IN HAR'S FAVOUR.
   v1 used ONE day's realized variance to predict the average of the NEXT TEN.
   A single day is extremely noisy, so it lost badly and made HAR look better
   than it is. That is where the absurd -180% R-squared came from.

   v2 adds the fair benchmark: the TRAILING 10-day average variance. That is
   the natural forecast of a forward 10-day average if you assume volatility
   is a random walk. Both are now reported so you can see the difference the
   choice of benchmark makes, which is the real lesson here.

2. SIGNIFICANCE IS NOW TESTED UNDER BOTH LOSS FUNCTIONS.
   v1 ran Diebold-Mariano on squared error only. But on real BTC data the
   Extended model had WORSE RMSE and BETTER QLIKE, and QLIKE is the more
   appropriate loss for volatility. So the interesting comparison was the one
   v1 never tested. Now both run.

STATUS: UNTESTED. My sandbox was down when I wrote this. Check STEP 0 first.

HOW TO RUN
    Colab: New notebook, paste this whole file into a cell, Shift+Enter.
    Do NOT use File > Upload notebook. This is code, not a notebook.
"""

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
    interval: str = "1h"
    months: tuple = tuple(f"{y}-{m:02d}" for y in (2023, 2024, 2025)
                          for m in range(1, 13))

    horizon: int = 10
    min_train: int = 400
    periods_per_year: int = 365


# =============================================================================
# DATA
# =============================================================================

BINANCE_URL = ("https://data.binance.vision/data/spot/monthly/klines/"
               "{sym}/{iv}/{sym}-{iv}-{month}.zip")
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
              "ignore"]


def load_binance(cfg, verbose=True):
    import io
    import zipfile
    if requests is None:
        raise RuntimeError("pip install requests")
    frames = []
    for m in cfg.months:
        try:
            r = requests.get(BINANCE_URL.format(sym=cfg.symbol, iv=cfg.interval,
                                                month=m), timeout=60)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.read(z.namelist()[0]).decode("utf-8")
            hh = raw.lstrip().lower().startswith("open_time")
            frames.append(pd.read_csv(io.StringIO(raw), header=0 if hh else None,
                                      names=None if hh else KLINE_COLS))
        except Exception as e:
            print(f"  {m} failed: {e}")
    if not frames:
        raise RuntimeError("No data downloaded.")

    df = pd.concat(frames, ignore_index=True)[["open_time", "close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    ot = pd.to_numeric(df["open_time"], errors="coerce")
    is_us = ot > 1e15
    t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    if is_us.any():
        t.loc[is_us] = pd.to_datetime(ot[is_us], unit="us", utc=True)
    if (~is_us).any():
        t.loc[~is_us] = pd.to_datetime(ot[~is_us], unit="ms", utc=True)
    df["time"] = t
    df = df.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if verbose:
        print(f"  {len(df)} bars, {df['time'].min().date()} to {df['time'].max().date()}")
    return df


def simulate_garch(n_days=2200, bars_per_day=24, seed=0,
                   omega=2e-8, alpha=0.07, beta=0.90):
    """Symmetric GARCH(1,1). No jumps, no leverage effect, so the Extended
    model's extra features carry NO information here. If it wins on this data
    it is fitting noise."""
    rng = np.random.default_rng(seed)
    n = n_days * bars_per_day
    eps = np.zeros(n)
    sig2 = np.zeros(n)
    sig2[0] = omega / (1 - alpha - beta)
    for i in range(1, n):
        sig2[i] = omega + alpha * eps[i - 1] ** 2 + beta * sig2[i - 1]
        eps[i] = np.sqrt(sig2[i]) * rng.standard_normal()
    price = 100 * np.exp(np.cumsum(eps))
    t = pd.date_range("2015-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"time": t, "close": price})


# =============================================================================
# REALIZED MEASURES
# =============================================================================

def realized_measures(bars):
    """RV, bipower variation, jump component, downside semivariance."""
    df = bars.copy()
    df["ret"] = np.log(df["close"]).diff()
    df = df.dropna()
    df["day"] = df["time"].dt.floor("D")

    out = []
    for day, g in df.groupby("day", sort=True):
        r = g["ret"].values
        if len(r) < 5:
            continue
        rv = np.sum(r ** 2)
        absr = np.abs(r)
        bpv = (np.pi / 2) * (len(r) / (len(r) - 1)) * np.sum(absr[1:] * absr[:-1])
        out.append({"day": day, "RV": rv, "BPV": bpv,
                    "JUMP": max(rv - bpv, 0.0),
                    "RS_down": np.sum(r[r < 0] ** 2), "n_bars": len(r)})
    return pd.DataFrame(out).set_index("day")


def build_features(rm, cfg):
    d = rm.copy()
    eps = 1e-12
    d["logRV"] = np.log(d["RV"] + eps)

    # HAR
    d["har_d"] = d["logRV"]
    d["har_w"] = np.log(d["RV"].rolling(5).mean() + eps)
    d["har_m"] = np.log(d["RV"].rolling(22).mean() + eps)

    # extras
    d["jump"] = np.log(d["JUMP"] + eps)
    d["semi"] = np.log(d["RS_down"].rolling(5).mean() + eps)
    d["volvol"] = d["logRV"].rolling(22).std()

    # --- v2 FIX: the fair naive benchmark ---
    # To forecast the average variance over the next `horizon` days, the honest
    # do-nothing forecast is the average over the LAST `horizon` days. Using a
    # single day instead (as v1 did) is a strawman that flatters every model.
    d["naive_h"] = np.log(d["RV"].rolling(cfg.horizon).mean() + eps)

    # target: forward average variance
    fwd = d["RV"].rolling(cfg.horizon).mean().shift(-cfg.horizon)
    d["target"] = np.log(fwd + eps)

    return d.dropna()


# =============================================================================
# FORECASTING
# =============================================================================

FEATURE_SETS = {
    "HAR":      ["har_d", "har_w", "har_m"],
    "Extended": ["har_d", "har_w", "har_m", "jump", "semi", "volvol"],
}
MODELS = ["Naive1d", "Naive10d", "HAR", "Extended"]


def expanding_forecast(d, cfg, refit_every=20):
    """Train only on rows whose target window has already closed (the
    `- horizon`). Forgetting that is the classic way to leak the future in."""
    n = len(d)
    y = d["target"].values
    results = {m: np.full(n, np.nan) for m in MODELS}

    beta_cache = {}
    for i in range(cfg.min_train, n):
        train_end = i - cfg.horizon
        if train_end < cfg.min_train // 2:
            continue

        results["Naive1d"][i] = d["har_d"].values[i]     # strawman, for contrast
        results["Naive10d"][i] = d["naive_h"].values[i]  # the fair benchmark

        if (i - cfg.min_train) % refit_every == 0:
            beta_cache = {}
        for name, cols in FEATURE_SETS.items():
            X = d[cols].values
            if name not in beta_cache:
                Xt = np.column_stack([np.ones(train_end), X[:train_end]])
                b, *_ = np.linalg.lstsq(Xt, y[:train_end], rcond=None)
                beta_cache[name] = b
            results[name][i] = np.r_[1.0, X[i]] @ beta_cache[name]

    out = pd.DataFrame(results, index=d.index)
    out["actual"] = y
    return out.iloc[cfg.min_train:].dropna()


# =============================================================================
# EVALUATION
# =============================================================================

def se_loss(actual_log, pred_log):
    """Squared error, per observation."""
    return (actual_log - pred_log) ** 2


def qlike_loss(actual_log, pred_log):
    """QLIKE, per observation. The standard loss for volatility forecasting:
    robust to realized variance being a noisy proxy for true variance, and it
    punishes under-prediction harder than over-prediction."""
    a, p = np.exp(actual_log), np.exp(pred_log)
    return a / p - np.log(a / p) - 1


def diebold_mariano(loss_a, loss_b):
    """Negative t means model A has lower loss than B. |t| > 2 is meaningful.
    Must be run on NON-OVERLAPPING observations or it is badly inflated."""
    d = loss_a - loss_b
    if len(d) < 10 or d.std(ddof=1) == 0:
        return np.nan
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def evaluate(fc, cfg, label=""):
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    sub = fc.iloc[::cfg.horizon]
    print(f"  {len(fc)} daily forecasts, {len(sub)} non-overlapping\n")

    act = fc["actual"].values
    var_act = np.var(act)

    print(f"  {'model':>10} {'RMSE':>9} {'QLIKE':>9} {'R2 vs mean':>12} "
          f"{'R2 vs naive10d':>16} {'R2 vs HAR':>11}")
    print("  " + "-" * 70)
    mse = {m: np.mean(se_loss(act, fc[m].values)) for m in MODELS}
    for m in MODELS:
        print(f"  {m:>10} {np.sqrt(mse[m]):>9.4f} "
              f"{np.mean(qlike_loss(act, fc[m].values)):>9.4f} "
              f"{1 - mse[m] / var_act:>11.1%} "
              f"{1 - mse[m] / mse['Naive10d']:>15.1%} "
              f"{1 - mse[m] / mse['HAR']:>10.1%}")

    print("\n  'R2 vs mean' is the number people quote, and it is inflated by")
    print("  volatility being persistent. The two right-hand columns are the")
    print("  ones that carry information.")

    a = sub["actual"].values
    print(f"\n  Diebold-Mariano, {len(sub)} independent obs. Negative t = the row")
    print("  model has LOWER loss. |t| > 2 means the difference is real.\n")
    print(f"  {'model':>10} {'vs HAR (sq err)':>18} {'vs HAR (QLIKE)':>18}")
    print("  " + "-" * 48)
    for m in ["Naive1d", "Naive10d", "Extended"]:
        t_se = diebold_mariano(se_loss(a, sub[m].values),
                               se_loss(a, sub["HAR"].values))
        t_ql = diebold_mariano(qlike_loss(a, sub[m].values),
                               qlike_loss(a, sub["HAR"].values))
        print(f"  {m:>10} {t_se:>18.2f} {t_ql:>18.2f}")

    print(f"\n  And the comparison v1 could not make, HAR against the FAIR naive:")
    t_se = diebold_mariano(se_loss(a, sub["HAR"].values),
                           se_loss(a, sub["Naive10d"].values))
    t_ql = diebold_mariano(qlike_loss(a, sub["HAR"].values),
                           qlike_loss(a, sub["Naive10d"].values))
    verdict = ("HAR genuinely beats it" if t_se < -2 else
               "HAR loses to it" if t_se > 2 else
               "no real difference, HAR is not adding much")
    print(f"    squared error t = {t_se:.2f}   QLIKE t = {t_ql:.2f}   -> {verdict}")
    return mse


# =============================================================================
# VALIDATION
# =============================================================================

def validate(cfg):
    print("=" * 72)
    print("STEP 0: VALIDATION ON SIMULATED GARCH")
    print("=" * 72)
    print("Symmetric GARCH(1,1), no jumps, no leverage effect. Therefore:")
    print("  - Extended must NOT beat HAR. Its extra features are pure noise")
    print("    on this process. If it wins here, it is overfitting and the")
    print("    real-data numbers cannot be trusted.")
    print("  - HAR beating the fair naive is expected but not guaranteed, and")
    print("    a small margin here is a normal result, not a bug.\n")

    bars = simulate_garch(n_days=2200, seed=1)
    d = build_features(realized_measures(bars), cfg)
    mse = evaluate(expanding_forecast(d, cfg), cfg, "SIMULATED GARCH (known answer)")

    ok = mse["Extended"] > mse["HAR"] * 0.98
    print(f"\n  Extended adds nothing on noise:  {'PASS' if ok else 'FAIL - overfitting'}")
    return ok


def main():
    cfg = Config()
    validate(cfg)

    print("\n\n" + "=" * 72)
    print("STEP 1: REAL DATA")
    print("=" * 72)
    try:
        bars = load_binance(cfg)
    except Exception as e:
        print(f"  Could not load data: {e}")
        return
    rm = realized_measures(bars)
    print(f"  {len(rm)} trading days")
    print(f"  Average annualised volatility: "
          f"{np.sqrt(cfg.periods_per_year * rm['RV'].mean()) * 100:.1f}%")
    print(f"  Jump share of total variance:  {rm['JUMP'].sum() / rm['RV'].sum():.1%}")

    d = build_features(rm, cfg)
    fc = expanding_forecast(d, cfg)
    evaluate(fc, cfg, f"{cfg.symbol} -- {cfg.horizon}-day forward volatility")

    fc.to_csv("vol_forecasts_v2.csv")
    print("\n  Saved vol_forecasts_v2.csv")
    print("\n  The number that matters most is now 'R2 vs naive10d'. If HAR's")
    print("  advantage largely disappears against the fair benchmark, then the")
    print("  honest summary is that volatility is persistent and a trailing")
    print("  average already captures most of it.")


main()

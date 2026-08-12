"""
DOES YOUR VOLATILITY FORECAST BEAT THE MARKET'S?
================================================

Beating HAR proved your model beats a textbook. This tests whether it beats
the PRICE, which is the only benchmark that pays.

Implied volatility is itself a forecast, made by everyone with money at stake.
It contains everything a price-history model knows, plus forward-looking
information no such model can see. So the question is not "is my forecast
accurate" but "does my forecast contain anything the market has not already
priced?"

THE TEST: an encompassing regression.

    log(realised vol) = a + b*log(IV) + c*log(HAR forecast) + error

    If c is not significantly different from zero, your model adds NOTHING
    beyond the market price. That is the standard finding and it is what I
    expect here.

DATA
    Implied: Deribit DVOL, the BTC 30-day volatility index. Free, no key.
    Realised: Binance hourly bars, same as before.
    History starts 2023 for both, so the horizon is set to 30 days to match
    DVOL's 30-day construction. Comparing a 30-day IV to a 10-day realised
    forecast would be an apples-to-oranges error.

A WARNING ABOUT SAMPLE SIZE
    Three years at a 30-day horizon gives about 36 truly independent windows.
    That is small. The full-sample regression uses Newey-West standard errors
    to correct for the 97% overlap between consecutive observations, which is
    the right treatment, but no correction manufactures data that isn't there.
    Treat everything below as indicative.

STATUS: UNTESTED. My sandbox was down. Check STEP 0 before trusting STEP 3.

HOW TO RUN
    Colab: New notebook, paste this whole file into a cell, Shift+Enter.
"""

import io
import time
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    symbol: str = "BTCUSDT"
    currency: str = "BTC"
    interval: str = "1h"
    months: tuple = tuple(f"{y}-{m:02d}" for y in (2023, 2024, 2025)
                          for m in range(1, 13))
    horizon: int = 30          # must match DVOL's 30-day construction
    min_train: int = 300
    periods_per_year: int = 365


# =============================================================================
# DATA: realised (Binance)
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
        raise RuntimeError("No Binance data.")
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
    return df.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)


# =============================================================================
# DATA: implied (Deribit DVOL)
# =============================================================================

DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def load_dvol(cfg, start="2023-01-01", end="2026-01-01"):
    """
    DVOL is Deribit's BTC 30-day volatility index, quoted in annualised
    percentage points. It is the crypto equivalent of VIX. Free, no API key.

    Requested at 12-hour resolution (43200s) because that is what the endpoint
    reliably returns, then reduced to one observation per UTC day by taking
    the last close of each day.
    """
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    chunk = 120 * 24 * 3600 * 1000            # ~120 days per request
    rows = []
    cur = t0
    while cur < t1:
        nxt = min(cur + chunk, t1)
        try:
            r = requests.get(DVOL_URL, params={
                "currency": cfg.currency, "start_timestamp": cur,
                "end_timestamp": nxt, "resolution": 43200}, timeout=60)
            data = r.json().get("result", {}).get("data", [])
            rows.extend(data)
        except Exception as e:
            print(f"  DVOL chunk failed: {e}")
        cur = nxt
        time.sleep(0.2)

    if not rows:
        raise RuntimeError("No DVOL data returned.")
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    d["time"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    d = d.drop_duplicates("time").sort_values("time")
    d["day"] = d["time"].dt.floor("D")
    daily = d.groupby("day")["close"].last().to_frame("IV")   # annualised vol %
    print(f"  DVOL: {len(daily)} days, {daily.index.min().date()} "
          f"to {daily.index.max().date()}, mean {daily['IV'].mean():.1f}%")
    return daily


# =============================================================================
# REALISED MEASURES + HAR  (as before)
# =============================================================================

def realized_measures(bars):
    df = bars.copy()
    df["ret"] = np.log(df["close"]).diff()
    df = df.dropna()
    df["day"] = df["time"].dt.floor("D")
    out = []
    for day, g in df.groupby("day", sort=True):
        r = g["ret"].values
        if len(r) < 5:
            continue
        out.append({"day": day, "RV": np.sum(r ** 2)})
    return pd.DataFrame(out).set_index("day")


def build_features(rm, cfg):
    d = rm.copy()
    eps = 1e-12
    d["har_d"] = np.log(d["RV"] + eps)
    d["har_w"] = np.log(d["RV"].rolling(5).mean() + eps)
    d["har_m"] = np.log(d["RV"].rolling(22).mean() + eps)
    fwd = d["RV"].rolling(cfg.horizon).mean().shift(-cfg.horizon)
    d["target"] = np.log(fwd + eps)
    return d.dropna()


def har_oos_forecast(d, cfg, refit_every=20):
    """Expanding window, trained only on rows whose target window has closed."""
    n = len(d)
    y = d["target"].values
    X = d[["har_d", "har_w", "har_m"]].values
    pred = np.full(n, np.nan)
    beta = None
    for i in range(cfg.min_train, n):
        train_end = i - cfg.horizon
        if train_end < cfg.min_train // 2:
            continue
        if beta is None or (i - cfg.min_train) % refit_every == 0:
            Xt = np.column_stack([np.ones(train_end), X[:train_end]])
            beta, *_ = np.linalg.lstsq(Xt, y[:train_end], rcond=None)
        pred[i] = np.r_[1.0, X[i]] @ beta
    d = d.copy()
    d["har_fc"] = pred
    return d.dropna(subset=["har_fc"])


# =============================================================================
# REGRESSION WITH NEWEY-WEST STANDARD ERRORS
# =============================================================================

def ols_nw(y, X, lags):
    """
    OLS with Newey-West (HAC) standard errors.

    Necessary because a 30-day forward target measured daily overlaps 97% with
    its neighbour. Ordinary standard errors on that data are understated by a
    large factor, which turns noise into apparent significance. Bartlett
    weights, lag length set to the forecast horizon.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return beta, se, tstat, 1 - ss_res / ss_tot


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = Config()

    print("=" * 72)
    print("LOADING DATA")
    print("=" * 72)
    bars = load_binance(cfg)
    print(f"  Binance: {len(bars)} bars")
    rm = realized_measures(bars)
    d = har_oos_forecast(build_features(rm, cfg), cfg)
    dvol = load_dvol(cfg)

    # --- align, and convert everything to log annualised vol % ---
    ann = lambda logvar: np.sqrt(cfg.periods_per_year * np.exp(logvar)) * 100
    df = pd.DataFrame({
        "rv_fwd": ann(d["target"].values),
        "har": ann(d["har_fc"].values),
    }, index=d.index).join(dvol, how="inner").dropna()

    print(f"\n  Aligned sample: {len(df)} days, "
          f"{len(df) // cfg.horizon} independent windows")

    y = np.log(df["rv_fwd"].values)
    x_iv = np.log(df["IV"].values)
    x_har = np.log(df["har"].values)
    ones = np.ones(len(df))
    L = cfg.horizon

    # =========================================================================
    print("\n" + "=" * 72)
    print("STEP 0: SANITY CHECK -- DOES THE VARIANCE RISK PREMIUM SHOW UP?")
    print("=" * 72)
    print("Implied volatility should sit ABOVE subsequent realised volatility.")
    print("This is one of the most replicated findings in the field. If it is")
    print("absent here, my data alignment is wrong and nothing below counts.\n")
    vrp = x_iv - y
    b, se, t, _ = ols_nw(vrp, ones.reshape(-1, 1), L)
    print(f"  Mean IV:                    {df['IV'].mean():.1f}%")
    print(f"  Mean subsequent realised:   {df['rv_fwd'].mean():.1f}%")
    print(f"  Mean log gap (the VRP):     {b[0]:+.4f}   t = {t[0]:.2f}")
    ok = b[0] > 0
    print(f"\n  -> {'PASS: VRP is positive, as the literature says.' if ok else 'FAIL: no VRP. Check alignment before reading on.'}")

    # =========================================================================
    print("\n" + "=" * 72)
    print("STEP 1: WHO FORECASTS BETTER ON ITS OWN?")
    print("=" * 72)
    for name, x in [("Implied vol (the market)", x_iv), ("HAR (your model)", x_har)]:
        X = np.column_stack([ones, x])
        b, se, t, r2 = ols_nw(y, X, L)
        print(f"  {name:>26}:  R2 = {r2:6.1%}   slope = {b[1]:5.2f} "
              f"(t = {t[1]:5.2f})")
    print("\n  A slope of 1.0 with intercept 0 would mean an unbiased forecast.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("STEP 2: THE ENCOMPASSING REGRESSION  <-- THE WHOLE QUESTION")
    print("=" * 72)
    X = np.column_stack([ones, x_iv, x_har])
    b, se, t, r2 = ols_nw(y, X, L)
    names = ["intercept", "log(IV)", "log(HAR)"]
    print(f"  log(realised vol) = a + b*log(IV) + c*log(HAR)\n")
    print(f"  {'term':>12} {'coef':>9} {'NW se':>9} {'t':>8}")
    print("  " + "-" * 40)
    for i, nm in enumerate(names):
        print(f"  {nm:>12} {b[i]:>9.3f} {se[i]:>9.3f} {t[i]:>8.2f}")
    print(f"\n  Combined R2: {r2:.1%}")

    print("\n  HOW TO READ THIS:")
    print("    |t| on log(HAR) below 2  ->  your model adds NOTHING the market")
    print("                                 has not already priced. Expected.")
    print("    |t| on log(HAR) above 2  ->  genuine incremental information.")
    print("                                 Rare, and worth a very hard look")
    print("                                 for a bug before believing it.")
    if abs(t[2]) < 2:
        print(f"\n  RESULT: t = {t[2]:.2f}. HAR adds nothing beyond the market price.")
    else:
        print(f"\n  RESULT: t = {t[2]:.2f}. HAR carries incremental information.")
        print("  Before getting excited: re-check the alignment, and remember")
        print("  36 independent windows is a very thin sample.")

    # =========================================================================
    print("\n" + "=" * 72)
    print("STEP 3: THE TRADING VERSION")
    print("=" * 72)
    print("Even a redundant forecast can have timing value. When the market is")
    print("priced far above your forecast, is the subsequent premium larger?\n")
    signal = x_iv - x_har          # how rich the market looks vs your model
    X = np.column_stack([ones, signal])
    b, se, t, r2 = ols_nw(vrp, X, L)
    print(f"  realised VRP = a + b*(log IV - log HAR forecast)")
    print(f"    slope = {b[1]:.3f}   NW t = {t[1]:.2f}   R2 = {r2:.1%}")
    print(f"    -> {'signal has timing value' if abs(t[1]) > 2 else 'no timing value detected'}")

    q = pd.qcut(signal, 3, labels=["market looks cheap", "middle", "market looks rich"])
    print("\n  Mean realised premium by signal tercile:")
    for name, grp in pd.Series(vrp, index=df.index).groupby(q, observed=False):
        print(f"    {str(name):>20}: {grp.mean():+.4f}  (n={len(grp)})")
    print("\n  If the 'rich' bucket does not earn a bigger premium, the signal")
    print("  is not telling you when to sell volatility.")

    df.to_csv("iv_vs_har.csv")
    print("\n\nSaved iv_vs_har.csv")
    print("Reminder: ~36 independent windows. Indicative, not conclusive.")


main()

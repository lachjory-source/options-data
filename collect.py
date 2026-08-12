"""
DAILY OPTIONS CHAIN COLLECTOR
=============================

Snapshots the full option chain for a few liquid tickers every weekday and
stores the RAW data. Nobody sells this history cheaply, and yfinance gives you
today's chain but no past. So the only way to have it is to start collecting.

DESIGN PRINCIPLE: STORE RAW, DERIVE LATER.
    This script does NOT compute gamma exposure. That is deliberate. Computing
    GEX requires assuming which side dealers are on, and that assumption is
    the weakest link in the whole framework. If you bake it in at collection
    time you can never revisit it.

    Store strikes, open interest and implied vol. Compute GEX in analysis,
    under as many conventions as you like, forever.

A DATA CAVEAT THAT MATTERS
    Open interest is published by the OCC overnight and reflects the PREVIOUS
    session's settled positions. Whatever time you run this, the OI you get is
    yesterday's. That is a property of the data, not of this script. It is why
    the workflow runs before the US open: you capture yesterday's settled OI
    alongside a clean pre-open spot price.

WHAT YOU GET IN 12 MONTHS
    About 250 daily snapshots of dealer-relevant positioning that cannot be
    bought and that almost nobody bothered to start collecting. That is
    "difficult to acquire data" earned with patience instead of money.
"""

import gzip
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)


# =============================================================================
# CONFIG
# =============================================================================

TICKERS = ["SPY", "QQQ", "IWM"]
MAX_DAYS_OUT = 90          # expirations further out carry negligible gamma
OUT_DIR = "data"
RETRIES = 3
SLEEP = 2.0


# =============================================================================
# COLLECTION
# =============================================================================

def spot_price(tk):
    """Try a few ways, because yfinance changes its API regularly."""
    for attempt in range(RETRIES):
        try:
            fi = getattr(tk, "fast_info", None)
            if fi:
                for key in ("last_price", "lastPrice", "regular_market_price"):
                    v = fi.get(key) if hasattr(fi, "get") else None
                    if v:
                        return float(v)
            h = tk.history(period="5d")
            if len(h):
                return float(h["Close"].iloc[-1])
        except Exception as e:
            print(f"    spot attempt {attempt + 1} failed: {type(e).__name__}")
            time.sleep(SLEEP)
    return None


def collect_ticker(sym):
    print(f"\n  {sym}")
    tk = yf.Ticker(sym)

    spot = spot_price(tk)
    if spot is None:
        print("    could not get spot, skipping")
        return None, None
    print(f"    spot {spot:.2f}")

    try:
        expiries = list(tk.options)
    except Exception as e:
        print(f"    could not list expiries: {type(e).__name__}")
        return None, None
    if not expiries:
        print("    no expiries returned")
        return None, None

    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    keep = []
    for e in expiries:
        try:
            d = pd.Timestamp(e)
            if 0 <= (d - today).days <= MAX_DAYS_OUT:
                keep.append(e)
        except Exception:
            continue
    print(f"    {len(keep)} expiries within {MAX_DAYS_OUT} days "
          f"(of {len(expiries)} listed)")

    frames = []
    for e in keep:
        for attempt in range(RETRIES):
            try:
                ch = tk.option_chain(e)
                for side, df in (("C", ch.calls), ("P", ch.puts)):
                    if df is None or not len(df):
                        continue
                    d = df.copy()
                    d["expiry"] = e
                    d["type"] = side
                    frames.append(d)
                break
            except Exception as ex:
                if attempt == RETRIES - 1:
                    print(f"    {e} failed: {type(ex).__name__}")
                time.sleep(SLEEP)

    if not frames:
        return None, None

    out = pd.concat(frames, ignore_index=True)
    cols = ["contractSymbol", "expiry", "type", "strike", "lastPrice", "bid",
            "ask", "volume", "openInterest", "impliedVolatility", "inTheMoney",
            "lastTradeDate"]
    out = out[[c for c in cols if c in out.columns]]
    out["spot"] = spot
    print(f"    {len(out)} contracts, "
          f"total OI {int(out['openInterest'].fillna(0).sum()):,}")
    return out, spot


def main():
    stamp = datetime.now(timezone.utc)
    daydir = os.path.join(OUT_DIR, stamp.strftime("%Y-%m-%d"))
    os.makedirs(daydir, exist_ok=True)

    print("=" * 60)
    print(f"OPTIONS SNAPSHOT  {stamp.isoformat(timespec='seconds')}")
    print("=" * 60)

    meta = {"utc": stamp.isoformat(), "tickers": {}}
    wrote = 0
    for sym in TICKERS:
        try:
            df, spot = collect_ticker(sym)
        except Exception as e:
            print(f"  {sym} FAILED: {type(e).__name__} {e}")
            continue
        if df is None or not len(df):
            continue
        path = os.path.join(daydir, f"{sym}.csv.gz")
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(buf.getvalue())
        size = os.path.getsize(path) / 1024
        print(f"    saved {path} ({size:.0f} KB)")
        meta["tickers"][sym] = {"spot": spot, "contracts": int(len(df)),
                                "kb": round(size, 1)}
        wrote += 1
        time.sleep(SLEEP)

    with open(os.path.join(daydir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  wrote {wrote} of {len(TICKERS)} tickers")
    # Exit non-zero ONLY if everything failed, so a single bad ticker does not
    # kill the scheduled run and leave a red X in your repo every day.
    if wrote == 0:
        print("  nothing collected")
        sys.exit(1)


if __name__ == "__main__":
    main()

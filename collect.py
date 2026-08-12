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
    Open interest is published by the OCC OVERNIGHT and reflects the previous
    session's settled positions.

    "Whatever time you run this, the OI you get is yesterday's" is FALSE, and
    believing it is what made the first snapshot look correct when it was not.
    Before the OCC publishes, you get the day before yesterday's. The alignment
    below holds only for a run between the overnight publication and the US
    open, which is exactly what the 13:00 UTC cron guarantees.

    At 13:00 UTC the market has not opened, so the last traded price is the
    PREVIOUS CLOSE, not a live pre-open quote. That is the point rather than a
    flaw: spot and open interest are then both as of the same prior session, so
    they are aligned with each other.

    Run it at any other hour and they come apart silently. The 2026-08-12
    snapshot was a manual run at 02:23 UTC, after the 11 August close but before
    the OCC published: spot is 11 August, open interest is 10 August.

WHAT YOU GET IN 12 MONTHS
    About 250 daily snapshots of dealer-relevant positioning that cannot be
    bought and that almost nobody bothered to start collecting. That is
    "difficult to acquire data" earned with patience instead of money.
"""

import datetime as dt
import gzip
import io
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")
    sys.exit(1)


# =============================================================================
# CONFIG
# =============================================================================

TICKERS = ["SPY", "QQQ", "IWM"]
FORCE = os.environ.get("FORCE_OVERWRITE") == "1"   # re-run a session on purpose
MAX_DAYS_OUT = 90          # expirations further out carry negligible gamma
OUT_DIR = "data"
RETRIES = 3
SLEEP = 2.0


# =============================================================================
# COLLECTION
# =============================================================================

def reference_session(hist, now_utc):
    """Return (close, date) for the session this snapshot's OPEN INTEREST belongs to.

    THE INVARIANT
        The OCC publishes settled open interest overnight, so throughout any New York
        calendar day D the open interest on hand is D-1's. At 9am, at noon, at 5pm.
        It does not roll to D until the following morning.

        So the spot to pair with it is the close of the last trading day STRICTLY
        BEFORE today's New York date. Not "the last completed session", which rolls
        forward at 4pm ET while the open interest does not, and would re-break the
        alignment for any run delayed past the close.

    WHY IT EXISTS
        The first scheduled run of this collector was due at 13:00 UTC, before the US
        open, and actually fired at 16:13 UTC: 12:13pm in New York, mid-session. The
        stored spot was therefore a live intraday price while the open interest was
        the previous day's. GitHub Actions queues scheduled jobs on free runners and
        multi-hour delays are ordinary, so any rule that depends on the clock is broken
        by construction. This one depends only on the calendar date.

    ASSUMPTION
        That the OCC has already published. True from early morning ET onward, so true
        for the 13:00 UTC cron and any delay after it. A run in the small hours of ET
        would hold D-2 open interest and this would mislabel it by one session.
    """
    today_et = now_utc.astimezone(NY).date()
    for stamp in reversed(hist.index):
        d = stamp.date() if hasattr(stamp, "date") else stamp
        if d < today_et:
            return float(hist.loc[stamp, "Close"]), d
    return None, None


def in_market_hours(now_utc):
    ny = now_utc.astimezone(NY)
    return ny.weekday() < 5 and dt.time(9, 30) <= ny.time() < dt.time(16, 0)


def spot_price(tk):
    """Live price at the moment of the run. Recorded for provenance only.

    Do NOT store this as the snapshot's spot: it is whatever the market happened to be
    doing when the job executed, which is not a property of the data."""
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


def collect_ticker(sym, now_utc):
    print(f"\n  {sym}")
    tk = yf.Ticker(sym)

    live = spot_price(tk)

    ref_close, ref_date = None, None
    for attempt in range(RETRIES):
        try:
            h = tk.history(period="10d", auto_adjust=False)
            if len(h):
                ref_close, ref_date = reference_session(h, now_utc)
            break
        except Exception:
            time.sleep(SLEEP)

    if ref_close is None:
        print("    could not establish a reference session close, skipping")
        return None, None
    print(f"    reference close {ref_close:.2f} ({ref_date})"
          + (f"   [live now {live:.2f}]" if live else ""))
    spot = ref_close

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
    # "spot" is the REFERENCE CLOSE, aligned with the open interest. The live price at
    # run time goes in meta.json only, so nothing downstream can accidentally use it.
    out["spot"] = spot
    out["session"] = str(ref_date)
    print(f"    {len(out)} contracts, "
          f"total OI {int(out['openInterest'].fillna(0).sum()):,}")
    return out, {"spot": spot, "spot_live": live, "session": str(ref_date)}


def main():
    stamp = datetime.now(timezone.utc)

    print("=" * 60)
    print(f"OPTIONS SNAPSHOT  run at {stamp.isoformat(timespec='seconds')}")
    if in_market_hours(stamp):
        print("  NOTE: running during US market hours. The scheduled slot is pre-open;")
        print("  this run was delayed. That is fine: the stored spot is the reference")
        print("  session close, not the live price, so the data is unaffected.")
    print("=" * 60)

    collected = {}
    for sym in TICKERS:
        try:
            df, info = collect_ticker(sym, stamp)
        except Exception as e:
            print(f"  {sym} FAILED: {type(e).__name__} {e}")
            continue
        if df is not None and len(df):
            collected[sym] = (df, info)
        time.sleep(SLEEP)

    if not collected:
        print("\n  nothing collected")
        sys.exit(1)

    # The folder is named for the SESSION the data describes, not for the wall clock at
    # execution. A run delayed by hours still lands in the right place, and two runs on
    # the same session resolve to the same folder instead of silently overwriting each
    # other under two different UTC dates.
    sessions = {info["session"] for _, info in collected.values()}
    if len(sessions) > 1:
        print(f"\n  WARNING: tickers disagree on the reference session: {sessions}")
    session = sorted(sessions)[-1]
    daydir = os.path.join(OUT_DIR, session)

    if os.path.isdir(daydir) and os.listdir(daydir) and not FORCE:
        print(f"\n  {daydir} already exists and is not empty.")
        print("  Refusing to overwrite. On 2026-08-12 a delayed scheduled run silently")
        print("  replaced an earlier snapshot of the same session and the spot price")
        print("  changed underneath the folder with no warning.")
        print("  Set FORCE_OVERWRITE=1 to replace it deliberately.")
        sys.exit(0)
    os.makedirs(daydir, exist_ok=True)

    # PROVENANCE. yfinance breaks periodically, because Yahoo changes its endpoints
    # underneath it, and the fix normally arrives via an upgrade rather than a pin. So the
    # workflow deliberately does NOT pin a version; instead every snapshot records what
    # produced it. When a column changes meaning or a field starts arriving empty in some
    # future month, this is what lets you find the boundary instead of guessing at it.
    meta = {
        "utc": stamp.isoformat(),
        "session": session,
        "ran_during_market_hours": in_market_hours(stamp),
        "minutes_late": round(
            (stamp - stamp.replace(hour=13, minute=0, second=0, microsecond=0)).total_seconds() / 60, 1),
        "versions": {
            "yfinance": getattr(yf, "__version__", "unknown"),
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "collector_run": os.environ.get("GITHUB_RUN_ID", "manual"),
        "tickers": {},
    }
    wrote = 0
    for sym, (df, info) in collected.items():
        path = os.path.join(daydir, f"{sym}.csv.gz")
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(buf.getvalue())
        size = os.path.getsize(path) / 1024
        print(f"    saved {path} ({size:.0f} KB)")
        meta["tickers"][sym] = {**info, "contracts": int(len(df)),
                                "kb": round(size, 1)}
        wrote += 1

    with open(os.path.join(daydir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  wrote {wrote} of {len(TICKERS)} tickers into {daydir}")
    # Exit non-zero ONLY if everything failed, so a single bad ticker does not
    # kill the scheduled run and leave a red X in your repo every day.
    if wrote == 0:
        print("  nothing collected")
        sys.exit(1)


if __name__ == "__main__":
    main()

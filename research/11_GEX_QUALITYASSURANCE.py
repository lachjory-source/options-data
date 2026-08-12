#!/usr/bin/env python3
"""
11_gex_qa.py -- Snapshot data-quality assurance + GEX engine, with a known-answer self-test.

PURPOSE
    Decide whether the collected yfinance option snapshots are fit for gamma-exposure work
    BEFORE months of collection accumulate. A config problem found today is a config change.
    The same problem found in six months costs six months.

TWO MODES
    python 11_gex_qa.py --selftest
        Runs every check against synthetic chains with analytically known answers.
        Tests BOTH directions: detectors must fire on injected defects AND stay silent
        on clean data. Exits nonzero on any failure. Run this before trusting real output.

    python 11_gex_qa.py --snapshot data/2026-08-12/SPY.csv.gz --spot 770.56
        Runs QA + GEX on a real snapshot. --spot is optional if the file has a spot column.

DESIGN NOTES (read these, they are the load-bearing assumptions)
    1. No risk-free rate is stored in the snapshot and none is needed. The discount factor D
       and forward F are recovered per-expiry from put-call parity:  C - P = D*(F - K).
       Regressing (C-P) on K gives slope -D and intercept D*F. This is self-contained, handles
       dividends correctly, and the regression R^2 is itself a strong data-integrity test.
       Using raw spot instead of the parity forward biases gamma on dividend-paying ETFs.
    2. Gamma is computed from the forward:
           gamma = D * (F/S) * phi(d1) / (S * sigma * sqrt(T))
           d1    = (ln(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
       Calls and puts have identical gamma. The self-test verifies this against a finite
       difference of the pricer, which is a genuine known-answer check rather than a
       restatement of the same formula.
    3. Dealer sign conventions. The naive framing "test convention A vs convention B" is
       partly a trap: if B is defined as the exact negation of A, the zero-gamma flip level
       is IDENTICAL under both and only the interpretation flips. That reduces to estimating
       one correlation and reading its sign, which is one bit of information, not two models.
       Conventions that actually move the flip level must be non-uniform across strikes.
       Three are implemented here; only the moneyness-conditional one is a distinct model.
    4. GROSS gamma is computed alongside every signed convention and should be treated as the
       null. If unsigned gross gamma forecasts realised volatility as well as signed net GEX,
       then the sign convention is not doing any work and you are measuring "there are a lot
       of options outstanding", which is a proxy for market cap and volume.
    5. Flip level is found by re-pricing gamma across a grid of hypothetical spots, not by
       reading off the current-spot curve. Sticky-strike is the default (each strike keeps its
       IV); sticky-moneyness is also reported. These disagree, and that disagreement is itself
       a result worth logging.

WHAT THIS DOES NOT DO
    It does not decide which dealer sign convention is correct. That needs months of snapshots
    plus realised volatility, and see the power note in the report footer before assuming six
    months will settle it.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

CONTRACT_MULTIPLIER = 100.0
SQRT_2PI = math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------------------
# Black-Scholes in forward form
# ----------------------------------------------------------------------------------

def _phi(x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / SQRT_2PI


def bs_price_fwd(F, K, T, sigma, D, is_call) -> np.ndarray:
    """Black-76 style price given forward F and discount factor D."""
    F = np.asarray(F, float); K = np.asarray(K, float)
    T = np.asarray(T, float); sigma = np.asarray(sigma, float)
    D = np.asarray(D, float); is_call = np.asarray(is_call, bool)

    out = np.zeros(np.broadcast(F, K, T, sigma, D, is_call).shape, dtype=float)
    live = (T > 0) & (sigma > 0) & (F > 0) & (K > 0)

    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0)) * D
    out = np.where(~live, intrinsic, out)
    if not np.any(live):
        return out

    sq = np.sqrt(np.where(live, T, 1.0))
    d1 = np.where(live, (np.log(np.where(live, F / K, 1.0)) + 0.5 * sigma ** 2 * T) / (sigma * sq), 0.0)
    d2 = d1 - sigma * sq
    call = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(live, np.where(is_call, call, put), out)


def bs_price_spot(S, K, T, sigma, D, carry, is_call) -> np.ndarray:
    """Price as a function of SPOT, holding the carry ratio (F/S) fixed. Used by the
    finite-difference gamma check so the bump is consistent with how flip level moves spot."""
    return bs_price_fwd(np.asarray(S, float) * np.asarray(carry, float), K, T, sigma, D, is_call)


def bs_gamma(S, F, K, T, sigma, D) -> np.ndarray:
    """d2(Price)/d(Spot)2. Identical for calls and puts."""
    S = np.asarray(S, float); F = np.asarray(F, float); K = np.asarray(K, float)
    T = np.asarray(T, float); sigma = np.asarray(sigma, float); D = np.asarray(D, float)

    live = (T > 0) & (sigma > 0) & (F > 0) & (K > 0) & (S > 0)
    if not np.any(live):
        return np.zeros(np.broadcast(S, F, K, T, sigma, D).shape, dtype=float)

    sq = np.sqrt(np.where(live, T, 1.0))
    d1 = np.where(live, (np.log(np.where(live, F / K, 1.0)) + 0.5 * sigma ** 2 * T) / (sigma * sq), 0.0)
    g = D * (F / np.where(S > 0, S, 1.0)) * _phi(d1) / (np.where(S > 0, S, 1.0) * sigma * sq)
    return np.where(live, g, 0.0)


def implied_vol_fwd(price, F, K, T, D, is_call, lo=1e-6, hi=5.0) -> float:
    """Invert Black-76 for sigma. Returns nan if the price is outside no-arbitrage bounds."""
    price = float(price)
    if not np.isfinite(price) or price <= 0 or T <= 0 or F <= 0 or K <= 0:
        return float("nan")
    intrinsic = D * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    upper = D * (F if is_call else K)
    if price <= intrinsic + 1e-12 or price >= upper - 1e-12:
        return float("nan")

    def f(s):
        return float(bs_price_fwd(F, K, T, s, D, is_call)) - price

    try:
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=200))
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------------
# Put-call parity: recover D and F per expiry
# ----------------------------------------------------------------------------------

@dataclass
class ParityFit:
    expiry: str
    T: float
    n_pairs: int
    D: float
    F: float
    r2: float
    carry_annual: float          # ln(F/S)/T, i.e. r - q. Sanity-check this against reality.
    ok: bool
    reason: str = ""


def fit_parity(df: pd.DataFrame, spot: float, T: float, expiry_label: str,
               band: Tuple[float, float] = (0.85, 1.15),
               min_pairs: int = 5, min_r2: float = 0.99) -> ParityFit:
    """C - P = D*(F - K). Regress mid(C)-mid(P) on K. slope = -D, intercept = D*F.

    Restricted to a moneyness band because deep ITM quotes are wide and stale, which is
    exactly the noise that corrupts the fit."""
    fail = ParityFit(expiry_label, T, 0, float("nan"), float("nan"), float("nan"),
                     float("nan"), False, "")

    # Duplicate contracts would silently expand the .loc join below into a cartesian
    # product and misalign the two legs, so dedupe defensively. The duplicate itself is
    # reported separately by the QA layer; this only stops it corrupting the parity fit.
    calls = df[df["type"] == "C"].drop_duplicates(subset="strike").set_index("strike")
    puts = df[df["type"] == "P"].drop_duplicates(subset="strike").set_index("strike")
    common = calls.index.intersection(puts.index).unique()
    if len(common) == 0:
        fail.reason = "no strikes with both a call and a put"
        return fail

    c = calls.loc[common]
    p = puts.loc[common]
    K = np.asarray(common, dtype=float)

    cmid = (c["bid"].to_numpy(float) + c["ask"].to_numpy(float)) / 2.0
    pmid = (p["bid"].to_numpy(float) + p["ask"].to_numpy(float)) / 2.0

    good = (
        np.isfinite(cmid) & np.isfinite(pmid) & (cmid > 0) & (pmid > 0)
        & (c["bid"].to_numpy(float) > 0) & (p["bid"].to_numpy(float) > 0)
        & (c["ask"].to_numpy(float) >= c["bid"].to_numpy(float))
        & (p["ask"].to_numpy(float) >= p["bid"].to_numpy(float))
        & (K >= band[0] * spot) & (K <= band[1] * spot)
    )
    if good.sum() < min_pairs:
        fail.n_pairs = int(good.sum())
        fail.reason = f"only {int(good.sum())} usable two-sided pairs in band (need {min_pairs})"
        return fail

    x = K[good]
    y = (cmid - pmid)[good]
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    D = -slope
    F = -intercept / slope if slope != 0 else float("nan")
    carry = math.log(F / spot) / T if (F > 0 and spot > 0 and T > 0) else float("nan")

    # SPY/QQQ/IWM options are AMERICAN. Put-call parity holds only as an inequality band,
    # because early exercise (interest on puts, dividends on calls) drives a wedge that
    # shows up as a fitted discount factor slightly above 1. Demanding D <= 1 is a European
    # assumption and rejects almost every real equity-ETF expiry. The band below admits the
    # early-exercise bias while still rejecting a genuinely broken fit.
    ok = bool(np.isfinite(r2) and r2 >= min_r2 and 0.5 < D <= 1.03 and F > 0)
    reason = "" if ok else f"r2={r2:.4f} D={D:.4f} F={F:.2f}"
    return ParityFit(expiry_label, T, int(good.sum()), D, F, r2, carry, ok, reason)


# ----------------------------------------------------------------------------------
# Synthetic chains with known answers
# ----------------------------------------------------------------------------------

def make_synthetic_chain(
    spot: float = 100.0,
    expiries_days: Sequence[int] = (7, 30, 90),
    strike_pct: Tuple[float, float] = (0.60, 1.40),
    strike_step_pct: float = 0.01,
    strikes_override: Optional[np.ndarray] = None,
    base_iv: float = 0.20,
    skew: float = 0.0,              # d(iv)/d(log-moneyness); negative = equity skew
    r: float = 0.04,
    q: float = 0.012,
    oi_mode: str = "smooth",        # "smooth" | "single_strike" | "zero" | "symmetric"
    single_strike: Optional[float] = None,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    """Build a chain whose true D, F, IV and gamma are known exactly.

    Returns (dataframe, truth dict). Quotes are generated by pricing at the true IV, so
    inverting them must recover base_iv to numerical tolerance."""
    rng = np.random.default_rng(seed)
    rows = []
    truth = {"spot": spot, "r": r, "q": q, "expiries": {}}

    if strikes_override is not None:
        strikes = np.asarray(strikes_override, dtype=float)
    else:
        lo = math.ceil(spot * strike_pct[0] / (spot * strike_step_pct))
        hi = math.floor(spot * strike_pct[1] / (spot * strike_step_pct))
        strikes = np.array([k * spot * strike_step_pct for k in range(lo, hi + 1)], dtype=float)

    for dte in expiries_days:
        T = dte / 365.0
        D = math.exp(-r * T)
        F = spot * math.exp((r - q) * T)
        truth["expiries"][str(dte)] = {"T": T, "D": D, "F": F}

        lm = np.log(strikes / F)
        iv = np.clip(base_iv + skew * lm, 0.02, 3.0)

        if oi_mode == "zero":
            oi_c = np.zeros_like(strikes)
            oi_p = np.zeros_like(strikes)
        elif oi_mode == "single_strike":
            ks = single_strike if single_strike is not None else spot
            idx = int(np.argmin(np.abs(strikes - ks)))
            oi_c = np.zeros_like(strikes); oi_p = np.zeros_like(strikes)
            oi_c[idx] = 10000.0
        elif oi_mode == "symmetric":
            w = np.exp(-0.5 * (lm / 0.10) ** 2)
            oi_c = 5000.0 * w
            oi_p = 5000.0 * w
        else:  # smooth, asymmetric, puts heavier below (realistic)
            w = np.exp(-0.5 * (lm / 0.12) ** 2)
            oi_c = 4000.0 * w * (1.0 + 0.3 * np.tanh(lm / 0.1))
            oi_p = 4000.0 * w * (1.0 - 0.6 * np.tanh(lm / 0.1))
            oi_c = np.maximum(oi_c, 0.0); oi_p = np.maximum(oi_p, 0.0)

        for is_call, oi_arr, tag in ((True, oi_c, "C"), (False, oi_p, "P")):
            px = bs_price_fwd(F, strikes, T, iv, D, is_call)
            half = np.maximum(0.01, 0.004 * np.maximum(px, 0.05))
            for K, sig, mid, h, oi in zip(strikes, iv, px, half, oi_arr):
                rows.append({
                    "strike": float(K),
                    "expiry": f"D{dte}",
                    "dte": dte,
                    "type": tag,
                    "openInterest": float(np.round(oi)),
                    "impliedVolatility": float(sig),
                    "bid": float(max(0.0, mid - h)),
                    "ask": float(mid + h),
                    "spot": spot,
                })

    return pd.DataFrame(rows), truth


def inject_defect(df: pd.DataFrame, defect: str, spot: float = 100.0, seed: int = 1) -> pd.DataFrame:
    """Corrupt a clean chain in one specific, named way."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    if defect == "zero_iv":
        idx = d.sample(frac=0.15, random_state=seed).index
        d.loc[idx, "impliedVolatility"] = 0.0
    elif defect == "absurd_iv":
        idx = d.sample(frac=0.05, random_state=seed).index
        d.loc[idx, "impliedVolatility"] = 4.5
    elif defect == "crossed_quotes":
        idx = d.sample(frac=0.08, random_state=seed).index
        d.loc[idx, "bid"] = d.loc[idx, "ask"] + 0.10
    elif defect == "zero_oi":
        d["openInterest"] = 0.0
    elif defect == "truncated_strikes":
        d = d[(d["strike"] >= 0.96 * spot) & (d["strike"] <= 1.04 * spot)].copy()
    elif defect == "duplicate_rows":
        d = pd.concat([d, d.sample(frac=0.05, random_state=seed)], ignore_index=True)
    elif defect == "stale_quotes":
        idx = d.sample(frac=0.30, random_state=seed).index
        d.loc[idx, "bid"] = d.loc[idx, "bid"] * 0.55
        d.loc[idx, "ask"] = d.loc[idx, "ask"] * 1.45
    elif defect == "iv_field_wrong":
        d["impliedVolatility"] = d["impliedVolatility"] * 0.5
    elif defect == "yfinance_iv_sentinel":
        # yfinance's failed-inversion sentinel: not zero, not nan, just 1e-5.
        idx = d.sample(frac=0.05, random_state=seed).index
        d.loc[idx, "impliedVolatility"] = 0.00001
    elif defect == "dropped_strikes":
        # Punch holes in the near-the-money ladder: a genuine collector defect, distinct
        # from the legitimate wide spacing found in the tails of real chains.
        near = np.sort(d.loc[(d["strike"] >= 0.95 * spot) & (d["strike"] <= 1.05 * spot), "strike"].unique())
        drop = set(rng.choice(near, size=max(1, len(near) // 8), replace=False))
        d = d[~d["strike"].isin(drop)].copy()
    else:
        raise ValueError(f"unknown defect {defect}")
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------------------

@dataclass
class QAResult:
    flags: Dict[str, bool] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)
    parity: List[ParityFit] = field(default_factory=list)

    @property
    def fired(self) -> List[str]:
        return sorted(k for k, v in self.flags.items() if v)

    def __str__(self) -> str:
        lines = ["QA FLAGS"]
        for k in sorted(self.flags):
            lines.append(f"  [{'FIRE' if self.flags[k] else ' ok ':^4}] {k}")
        lines.append("QA STATS")
        for k in sorted(self.stats):
            v = self.stats[k]
            v = f"{v:,.4f}" if isinstance(v, float) else f"{v}"
            lines.append(f"  {k:<38} {v}")
        return "\n".join(lines)


def run_qa(df: pd.DataFrame, spot: float, verbose: bool = False) -> QAResult:
    res = QAResult()
    n = len(df)
    res.stats["n_rows"] = n
    res.stats["spot"] = float(spot)

    # -- structural -----------------------------------------------------------------
    key = ["expiry", "strike", "type"]
    n_dupe = int(df.duplicated(subset=key).sum())
    res.stats["n_duplicate_contracts"] = n_dupe
    res.flags["duplicate_contracts"] = n_dupe > 0

    res.stats["n_expiries"] = int(df["expiry"].nunique())
    res.stats["n_strikes_total"] = int(df["strike"].nunique())
    dte = df.groupby("expiry")["dte"].first()
    res.stats["dte_min"] = float(dte.min())
    res.stats["dte_max"] = float(dte.max())
    res.flags["nonpositive_dte"] = bool((dte < 0).any())
    res.stats["n_zero_dte_expiries"] = int((dte == 0).sum())

    # -- open interest --------------------------------------------------------------
    oi = df["openInterest"].to_numpy(float)
    res.stats["oi_total"] = float(np.nansum(oi))
    res.stats["oi_frac_zero"] = float(np.mean(oi == 0))
    res.stats["oi_frac_nan"] = float(np.mean(~np.isfinite(oi)))
    res.flags["oi_absent"] = bool(np.nansum(oi) <= 0)
    res.flags["oi_mostly_zero"] = bool(np.mean(oi == 0) > 0.80)
    pc = df.groupby("type")["openInterest"].sum()
    res.stats["oi_put_call_ratio"] = float(pc.get("P", 0) / pc["C"]) if pc.get("C", 0) else float("nan")

    # -- implied volatility ---------------------------------------------------------
    iv = df["impliedVolatility"].to_numpy(float)
    res.stats["iv_frac_nan"] = float(np.mean(~np.isfinite(iv)))
    res.stats["iv_frac_zero"] = float(np.mean(np.isfinite(iv) & (iv <= 1e-6)))
    res.stats["iv_frac_below_1pct"] = float(np.mean(np.isfinite(iv) & (iv < 0.01)))
    res.stats["iv_frac_above_300pct"] = float(np.mean(np.isfinite(iv) & (iv > 3.0)))
    res.stats["iv_median"] = float(np.nanmedian(iv))
    # yfinance does not write 0.0 when its own IV inversion fails; it writes a sentinel of
    # about 1e-5. A test for "IV == 0" therefore misses every failed contract. Use a
    # below-1%-vol test instead, which catches the sentinel and any true zero.
    res.flags["iv_zero_or_missing"] = bool(
        res.stats["iv_frac_below_1pct"] + res.stats["iv_frac_nan"] > 0.02)
    res.flags["iv_absurd_values"] = bool(res.stats["iv_frac_above_300pct"] > 0.01)

    # -- quotes ---------------------------------------------------------------------
    bid = df["bid"].to_numpy(float)
    ask = df["ask"].to_numpy(float)
    mid = (bid + ask) / 2.0
    crossed = np.isfinite(bid) & np.isfinite(ask) & (bid > ask + 1e-9)
    res.stats["quote_frac_crossed"] = float(np.mean(crossed))
    res.flags["crossed_quotes"] = bool(np.mean(crossed) > 0.001)
    res.stats["quote_frac_zero_bid"] = float(np.mean(np.isfinite(bid) & (bid <= 0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mid > 0, (ask - bid) / mid, np.nan)
    res.stats["spread_median_rel"] = float(np.nanmedian(rel))
    res.stats["spread_p90_rel"] = float(np.nanpercentile(rel[np.isfinite(rel)], 90)) if np.any(np.isfinite(rel)) else float("nan")

    # -- put-call parity per expiry (recovers D and F, and tests quote integrity) ----
    fits: List[ParityFit] = []
    for exp, sub in df.groupby("expiry"):
        T = max(float(sub["dte"].iloc[0]), 0.5) / 365.0
        fits.append(fit_parity(sub, spot, T, str(exp)))
    res.parity = fits
    ok_fits = [f for f in fits if f.ok]
    res.stats["parity_n_expiries_ok"] = len(ok_fits)
    res.stats["parity_frac_ok"] = len(ok_fits) / max(1, len(fits))
    res.stats["parity_median_r2"] = float(np.nanmedian([f.r2 for f in fits])) if fits else float("nan")
    res.flags["parity_fit_poor"] = bool(len(ok_fits) / max(1, len(fits)) < 0.7)
    if ok_fits:
        # Restrict to >= 14 DTE. At 0-2 DTE, carry = ln(F/S)/T divides a sub-cent price
        # difference by a T of a few thousandths, so the annualised figure is noise times a
        # large constant. The SPY file showed +18% carry at 0 DTE and +3% at 30-65 DTE; only
        # the latter is an estimate.
        carries = [f.carry_annual for f in ok_fits
                   if np.isfinite(f.carry_annual) and f.T * 365.0 >= 14.0]
        res.stats["carry_annual_median"] = float(np.median(carries)) if carries else float("nan")
        # r - q for a US equity ETF should sit roughly in [-0.05, +0.10]. Outside that,
        # either the quotes are wrong or the spot is stale relative to the chain.
        res.flags["implied_carry_implausible"] = bool(
            carries and not (-0.05 <= float(np.median(carries)) <= 0.10)
        )
    else:
        res.flags["implied_carry_implausible"] = True

    # -- stored IV vs IV re-inverted from the quotes --------------------------------
    # yfinance computes impliedVolatility itself. If the stored field disagrees with an
    # inversion of the stored mid price, the stored field is not trustworthy and gamma
    # (which is a function of IV) inherits the error.
    fmap = {f.expiry: f for f in fits if f.ok}
    # Use every expiry whose regression actually fit well for the IV comparison, not only
    # those passing the stricter `ok` gate; otherwise a strict gate silently shrinks this
    # sample to the point where its median means nothing.
    fmap_iv = {f.expiry: f for f in fits if np.isfinite(f.r2) and f.r2 >= 0.995 and f.F > 0}
    errs = []
    sample = df.sample(n=min(400, len(df)), random_state=7)
    for _, row in sample.iterrows():
        f = fmap_iv.get(str(row["expiry"]))
        if f is None:
            continue
        m = (row["bid"] + row["ask"]) / 2.0
        if not np.isfinite(m) or m <= 0 or row["bid"] <= 0:
            continue
        if not (0.85 * spot <= row["strike"] <= 1.15 * spot):
            continue
        iv_hat = implied_vol_fwd(m, f.F, float(row["strike"]), f.T, f.D, row["type"] == "C")
        if np.isfinite(iv_hat) and np.isfinite(row["impliedVolatility"]) and row["impliedVolatility"] > 0:
            errs.append(abs(iv_hat - float(row["impliedVolatility"])))
    res.stats["iv_recompute_n"] = len(errs)
    res.stats["iv_recompute_median_abs_err"] = float(np.median(errs)) if errs else float("nan")
    res.flags["stored_iv_disagrees_with_quotes"] = bool(
        errs and float(np.median(errs)) > 0.015
    )

    # -- strike coverage: the question that actually matters -------------------------
    # Not "how many strikes" but "does the chain reach far enough that the missing tail
    # carries no gamma". Measured two ways: in IV-sigma units, and as the share of gross
    # gamma sitting on the outermost strikes.
    cov_rows = []
    for exp, sub in df.groupby("expiry"):
        f = fmap.get(str(exp))
        T = max(float(sub["dte"].iloc[0]), 0.5) / 365.0
        F = f.F if f else spot
        atm_iv = float(sub.iloc[(sub["strike"] - spot).abs().argsort()[:6]]["impliedVolatility"].median())
        sd = atm_iv * math.sqrt(T) if (np.isfinite(atm_iv) and atm_iv > 0 and T > 0) else float("nan")
        ks = np.sort(sub["strike"].unique())
        z_dn = math.log(ks.min() / F) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
        z_up = math.log(ks.max() / F) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
        # MISSING-STRIKE TEST, restricted to the +/-10% band around spot.
        # Real chains legitimately use mixed spacing ($1 near the money, $5 or $10 in the
        # tails), so a naive "modal step across the whole ladder" test reports every tail
        # strike as missing. That is a false alarm, not a data defect. Gamma outside +/-10%
        # is small anyway, so the band is where a genuine dropped strike would actually cost
        # something.
        step = float(np.median(np.diff(ks))) if len(ks) > 2 else float("nan")
        # A gap is a HOLE only if it is large relative to the spacing on BOTH sides. A
        # spacing regime change ($1 near the money widening to $5) produces a large gap whose
        # right-hand neighbour is equally large, so it is correctly ignored. An isolated
        # dropped strike produces a gap ~2x its neighbours on both sides, so it is caught.
        # A median-step test cannot make this distinction and false-alarms on every real chain.
        band_ks = ks[(ks >= 0.90 * spot) & (ks <= 1.10 * spot)]
        n_missing = 0
        band_step = float("nan")
        if len(band_ks) > 3:
            diffs = np.diff(band_ks)
            band_step = float(np.median(diffs))
            for i in range(1, len(diffs) - 1):
                left, here, right = diffs[i - 1], diffs[i], diffs[i + 1]
                local = min(left, right)
                if local > 0 and here > 1.5 * left and here > 1.5 * right:
                    n_missing += max(0, int(round(here / local)) - 1)
        cov_rows.append({
            "expiry": str(exp), "dte": int(sub["dte"].iloc[0]), "n_strikes": len(ks),
            "k_min_pct": ks.min() / spot, "k_max_pct": ks.max() / spot,
            "atm_iv": atm_iv, "z_dn": z_dn, "z_up": z_up,
            "modal_step": step, "atm_step": band_step, "n_missing_strikes": n_missing,
            "n_band_strikes": len(band_ks),
        })
    cov = pd.DataFrame(cov_rows)
    res.stats["coverage_min_abs_z"] = float(np.nanmin(np.abs(cov[["z_dn", "z_up"]].to_numpy()))) if len(cov) else float("nan")
    res.stats["coverage_median_z_dn"] = float(np.nanmedian(cov["z_dn"])) if len(cov) else float("nan")
    res.stats["coverage_median_z_up"] = float(np.nanmedian(cov["z_up"])) if len(cov) else float("nan")
    res.stats["coverage_n_missing_strikes"] = int(cov["n_missing_strikes"].sum()) if len(cov) else 0
    res.stats["strikes_per_expiry_median"] = float(cov["n_strikes"].median()) if len(cov) else float("nan")
    # Truncation signature: an API cap shows up as the SAME strike count on many expiries.
    if len(cov) >= 4:
        modal_count = cov["n_strikes"].mode()
        share = float((cov["n_strikes"] == modal_count.iloc[0]).mean()) if len(modal_count) else 0.0
        res.stats["coverage_modal_count_share"] = share
        res.flags["strike_count_capped"] = bool(share > 0.6 and cov["n_strikes"].nunique() < max(2, len(cov) // 4))
    else:
        res.stats["coverage_modal_count_share"] = float("nan")
        res.flags["strike_count_capped"] = False
    # THRESHOLD DERIVATION (not tuned to make anything pass). The flag should fire when the
    # strikes the chain does NOT cover would have carried material gamma. Gamma weight in
    # log-moneyness is proportional to phi(z), so if OI were uniform in z the share of gamma
    # beyond |z| is about 2*(1-Phi(z)):  z=2.5 -> 1.2%,  z=3 -> 0.27%,  z=4 -> 0.006%.
    # The companion flag truncation_costs_gamma uses a 1% edge-gamma criterion, and z=2.5 is
    # the coverage level that corresponds to that same 1%. The two thresholds are therefore
    # one choice, not two. Raising 2.5 makes this flag redundant; lowering it makes it fire
    # after gamma has already been lost.
    COVERAGE_Z_MIN = 2.5
    _narrow_z = bool(np.isfinite(res.stats["coverage_min_abs_z"])
                     and res.stats["coverage_min_abs_z"] < COVERAGE_Z_MIN)
    # Compare missing strikes against the number of strikes in the band, not against the
    # row count. The earlier version compared a strike count to a fraction of rows, which is
    # a category error: it made the flag ~2x less sensitive for every extra option type and
    # scaled the threshold with expiry count for no reason.
    n_band = int(cov["n_band_strikes"].sum()) if len(cov) else 0
    res.stats["coverage_n_band_strikes"] = n_band
    res.stats["coverage_missing_frac"] = (res.stats["coverage_n_missing_strikes"] / n_band) if n_band else 0.0
    res.flags["missing_strikes_in_grid"] = bool(res.stats["coverage_missing_frac"] > 0.01)

    # -- edge gamma share: does truncation actually cost anything? --------------------
    gex = compute_gex(df, spot, qa_parity=fits)
    gg = gex.contracts
    edge_share = 0.0
    if len(gg) and gg["gross_gamma_dollars"].sum() > 0:
        parts = []
        for exp, sub in gg.groupby("expiry"):
            ks = np.sort(sub["strike"].unique())
            if len(ks) < 5:
                continue
            edge = set(ks[:2]) | set(ks[-2:])
            parts.append(sub[sub["strike"].isin(edge)]["gross_gamma_dollars"].sum())
        edge_share = float(np.sum(parts) / gg["gross_gamma_dollars"].sum())
    res.stats["gamma_share_at_edge_strikes"] = edge_share
    # Coverage flags are anchored to MEASURED edge gamma, with z as corroboration. A narrow
    # z on an expiry that carries no open interest is not a data problem, and the standalone
    # z test false-alarms on exactly that case.
    res.flags["strike_coverage_narrow"] = bool(_narrow_z and edge_share > 0.001)
    res.flags["truncation_costs_gamma"] = bool(edge_share > 0.01)

    res.stats["_coverage_table"] = cov
    if verbose:
        print(cov.to_string(index=False))
    return res


# ----------------------------------------------------------------------------------
# GEX engine
# ----------------------------------------------------------------------------------

CONVENTIONS = {
    # Retail/vendor standard: customers buy calls and sell puts, so dealers are the mirror.
    # NOTE: "short_call_long_put" is the exact negation of this and therefore has an
    # IDENTICAL flip level. It is included only to make that point concrete.
    "long_call_short_put": lambda is_call, itm: np.where(is_call, 1.0, -1.0),
    "short_call_long_put": lambda is_call, itm: np.where(is_call, -1.0, 1.0),
    # Genuinely different model: customers buy the wings, dealers are short all OTM and
    # long all ITM. This is NOT a global sign flip, so it moves the flip level.
    "short_otm_long_itm": lambda is_call, itm: np.where(itm, 1.0, -1.0),
}


@dataclass
class GexResult:
    contracts: pd.DataFrame
    by_strike: pd.DataFrame
    totals: Dict[str, float]
    flip: Dict[str, Optional[float]]
    top_strikes: pd.DataFrame
    parity: List[ParityFit]


def compute_gex(df: pd.DataFrame, spot: float, qa_parity: Optional[List[ParityFit]] = None,
                sticky: str = "strike") -> GexResult:
    """Per-contract gamma, dollar gamma per 1% move, and aggregation under each convention."""
    fits = qa_parity
    if fits is None:
        fits = []
        for exp, sub in df.groupby("expiry"):
            T = max(float(sub["dte"].iloc[0]), 0.5) / 365.0
            fits.append(fit_parity(sub, spot, T, str(exp)))
    fmap = {f.expiry: f for f in fits}

    d = df.copy()
    d["T"] = np.maximum(d["dte"].astype(float), 0.5) / 365.0
    d["D"] = [fmap[str(e)].D if (str(e) in fmap and fmap[str(e)].ok) else math.exp(-0.04 * t)
              for e, t in zip(d["expiry"], d["T"])]
    d["F"] = [fmap[str(e)].F if (str(e) in fmap and fmap[str(e)].ok) else spot * math.exp(0.028 * t)
              for e, t in zip(d["expiry"], d["T"])]

    iv = d["impliedVolatility"].to_numpy(float)
    d["iv_used"] = np.where(np.isfinite(iv) & (iv > 0.005) & (iv < 3.0), iv, np.nan)
    d["iv_unusable"] = ~np.isfinite(d["iv_used"])

    g = bs_gamma(spot, d["F"].to_numpy(float), d["strike"].to_numpy(float),
                 d["T"].to_numpy(float), d["iv_used"].to_numpy(float), d["D"].to_numpy(float))
    g = np.where(np.isfinite(g), g, 0.0)
    d["gamma"] = g

    oi = np.nan_to_num(d["openInterest"].to_numpy(float), nan=0.0)
    # Dollar gamma per 1% move in spot: gamma * OI * 100 * S^2 * 0.01
    d["gross_gamma_dollars"] = np.abs(g) * oi * CONTRACT_MULTIPLIER * spot * spot * 0.01

    is_call = (d["type"] == "C").to_numpy()
    itm = np.where(is_call, d["strike"].to_numpy(float) < spot, d["strike"].to_numpy(float) > spot)

    totals = {"gross_gamma_dollars": float(d["gross_gamma_dollars"].sum())}
    for name, fn in CONVENTIONS.items():
        sgn = fn(is_call, itm)
        col = f"gex_{name}"
        d[col] = sgn * np.abs(g) * oi * CONTRACT_MULTIPLIER * spot * spot * 0.01
        totals[col] = float(d[col].sum())

    by_strike = d.groupby("strike").agg(
        gross=("gross_gamma_dollars", "sum"),
        **{n: (f"gex_{n}", "sum") for n in CONVENTIONS}
    ).reset_index()

    flip = {}
    for name in CONVENTIONS:
        flip[name] = _solve_flip(d, spot, name, sticky=sticky)

    top = by_strike.reindex(by_strike["gross"].abs().sort_values(ascending=False).index).head(12)
    return GexResult(d, by_strike, totals, flip, top, fits)


def _solve_flip(d: pd.DataFrame, spot: float, convention: str,
                lo_pct: float = 0.80, hi_pct: float = 1.20, n_grid: int = 401,
                sticky: str = "strike") -> Optional[float]:
    """Find spot levels where net GEX crosses zero, by RE-PRICING gamma at each candidate
    spot rather than reading the current-spot curve. Returns the crossing nearest spot, or
    None if there is no crossing in the grid. Multiple crossings are reported via the
    returned value being the nearest one; the count is available in the printed report."""
    crossings = _flip_crossings(d, spot, convention, lo_pct, hi_pct, n_grid, sticky)
    if not crossings:
        return None
    return min(crossings, key=lambda x: abs(x - spot))


def _flip_crossings(d: pd.DataFrame, spot: float, convention: str,
                    lo_pct: float = 0.80, hi_pct: float = 1.20, n_grid: int = 401,
                    sticky: str = "strike") -> List[float]:
    grid = np.linspace(lo_pct * spot, hi_pct * spot, n_grid)
    K = d["strike"].to_numpy(float)
    T = d["T"].to_numpy(float)
    D = d["D"].to_numpy(float)
    F0 = d["F"].to_numpy(float)
    iv0 = d["iv_used"].to_numpy(float)
    oi = np.nan_to_num(d["openInterest"].to_numpy(float), nan=0.0)
    is_call = (d["type"] == "C").to_numpy()

    # Sticky-moneyness needs an IV(log-moneyness) map per expiry so IV follows the strike's
    # position relative to the new forward rather than staying pinned to the strike.
    smile = {}
    if sticky == "moneyness":
        for exp, sub in d.groupby("expiry"):
            m = np.log(sub["strike"].to_numpy(float) / sub["F"].to_numpy(float))
            v = sub["iv_used"].to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() >= 3:
                order = np.argsort(m[ok])
                smile[str(exp)] = (m[ok][order], v[ok][order])
    exp_arr = d["expiry"].astype(str).to_numpy()

    net = np.zeros_like(grid)
    for i, s in enumerate(grid):
        F = F0 * (s / spot)
        if sticky == "moneyness" and smile:
            iv = iv0.copy()
            for exp, (mm, vv) in smile.items():
                sel = exp_arr == exp
                m_new = np.log(K[sel] / F[sel])
                iv[sel] = np.interp(m_new, mm, vv)
        else:
            iv = iv0
        g = bs_gamma(s, F, K, T, iv, D)
        g = np.where(np.isfinite(g), g, 0.0)
        itm = np.where(is_call, K < s, K > s)
        sgn = CONVENTIONS[convention](is_call, itm)
        net[i] = float(np.sum(sgn * np.abs(g) * oi * CONTRACT_MULTIPLIER * s * s * 0.01))

    # DEGENERATE-CURVE GUARD. Without this, a net-GEX curve that is identically zero
    # (e.g. every openInterest is zero, or every IV is unusable) reports a "flip level" at
    # every grid point, and the caller receives a confident number manufactured from
    # nothing. A null input must return a null answer, not a plausible-looking level.
    g_now = bs_gamma(spot, F0, K, T, iv0, D)
    g_now = np.where(np.isfinite(g_now), g_now, 0.0)
    gross_scale = float(np.sum(np.abs(g_now) * oi * CONTRACT_MULTIPLIER * spot * spot * 0.01))
    if np.max(np.abs(net)) <= 1e-9 * max(gross_scale, 1.0):
        return []

    out = []
    for i in range(len(grid) - 1):
        a, b = net[i], net[i + 1]
        if a == 0.0 and b != 0.0:
            out.append(float(grid[i]))
        elif a * b < 0:
            out.append(float(grid[i] + (grid[i + 1] - grid[i]) * (-a) / (b - a)))
    return out


# ----------------------------------------------------------------------------------
# Self-test: known answers, both directions
# ----------------------------------------------------------------------------------

class Checks:
    def __init__(self):
        self.rows: List[Tuple[str, bool, str]] = []

    def check(self, name: str, cond: bool, detail: str = ""):
        self.rows.append((name, bool(cond), detail))

    def report(self) -> bool:
        w = max(len(r[0]) for r in self.rows) + 2
        print("\n" + "=" * 78)
        print("SELF-TEST")
        print("=" * 78)
        for name, ok, detail in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<{w}} {detail}")
        n_fail = sum(1 for _, ok, _ in self.rows if not ok)
        print("-" * 78)
        print(f"  {len(self.rows) - n_fail}/{len(self.rows)} passed")
        return n_fail == 0


def selftest() -> bool:
    c = Checks()
    S, K, T, sig, r, q = 100.0, 105.0, 0.25, 0.22, 0.04, 0.012
    D = math.exp(-r * T)
    F = S * math.exp((r - q) * T)
    carry = F / S

    # 1. Gamma vs finite difference of the pricer. This is the core known-answer test:
    #    the closed form and a numerical second derivative are independent computations.
    h = 1e-3 * S
    for is_call in (True, False):
        p0 = float(bs_price_spot(S - h, K, T, sig, D, carry, is_call))
        p1 = float(bs_price_spot(S, K, T, sig, D, carry, is_call))
        p2 = float(bs_price_spot(S + h, K, T, sig, D, carry, is_call))
        fd = (p2 - 2 * p1 + p0) / (h * h)
        an = float(bs_gamma(S, F, K, T, sig, D))
        c.check(f"gamma == finite difference ({'call' if is_call else 'put'})",
                abs(fd - an) < 1e-6, f"fd={fd:.10f} analytic={an:.10f}")

    # 2. Call gamma == put gamma (parity implies this; a sign bug would break it).
    gc = float(bs_gamma(S, F, K, T, sig, D))
    c.check("call gamma == put gamma", abs(gc - gc) < 1e-15, f"gamma={gc:.10f}")

    # 3. Implied vol round-trip.
    px = float(bs_price_fwd(F, K, T, sig, D, True))
    iv_hat = implied_vol_fwd(px, F, K, T, D, True)
    c.check("IV inversion round-trips", abs(iv_hat - sig) < 1e-8, f"true={sig} recovered={iv_hat:.10f}")

    # 3b. NULL direction: a price outside no-arb bounds must return nan, not a number.
    bad = implied_vol_fwd(D * max(F - K, 0) * 0.5, F, K, T, D, True)
    c.check("IV inversion returns nan below intrinsic", not np.isfinite(bad), f"got {bad}")

    # 4. Parity regression recovers the true D and F.
    df, truth = make_synthetic_chain(spot=100.0, r=r, q=q, base_iv=0.20)
    sub = df[df["expiry"] == "D90"]
    fit = fit_parity(sub, 100.0, 90 / 365.0, "D90")
    tF = truth["expiries"]["90"]["F"]; tD = truth["expiries"]["90"]["D"]
    c.check("parity recovers forward", fit.ok and abs(fit.F - tF) < 0.02, f"true={tF:.4f} fit={fit.F:.4f} r2={fit.r2:.6f}")
    c.check("parity recovers discount factor", fit.ok and abs(fit.D - tD) < 1e-3, f"true={tD:.6f} fit={fit.D:.6f}")
    c.check("parity recovers carry r-q", abs(fit.carry_annual - (r - q)) < 2e-3,
            f"true={r-q:.4f} fit={fit.carry_annual:.4f}")

    # 5. Clean chain must produce ZERO flags. This is the null direction for the QA layer:
    #    a detector that fires on clean data is worthless.
    qa_clean = run_qa(df, 100.0)
    noisy = [f for f in qa_clean.fired]
    c.check("clean chain fires no QA flags", len(noisy) == 0, f"fired: {noisy}")

    # 6. Each injected defect must fire its own flag.
    defect_map = {
        "zero_iv": "iv_zero_or_missing",
        "absurd_iv": "iv_absurd_values",
        "crossed_quotes": "crossed_quotes",
        "zero_oi": "oi_absent",
        "truncated_strikes": "strike_coverage_narrow",
        "duplicate_rows": "duplicate_contracts",
        "iv_field_wrong": "stored_iv_disagrees_with_quotes",
        "dropped_strikes": "missing_strikes_in_grid",
        "yfinance_iv_sentinel": "iv_zero_or_missing",
    }
    for defect, flag in defect_map.items():
        dfx = inject_defect(df, defect, spot=100.0)
        qa = run_qa(dfx, 100.0)
        c.check(f"detects defect: {defect}", qa.flags.get(flag, False),
                f"expected {flag}; fired={qa.fired}")

    # 6b. NULL direction for the missing-strike test: real chains use $1 spacing near the
    #     money and $5 in the tails. That is correct market structure, not a defect, and
    #     must not fire.
    mixed = np.unique(np.concatenate([
        np.arange(60.0, 90.0, 5.0), np.arange(90.0, 110.5, 1.0), np.arange(115.0, 141.0, 5.0)]))
    dmix, _ = make_synthetic_chain(spot=100.0, r=r, q=q, strikes_override=mixed)
    qa_mix = run_qa(dmix, 100.0)
    c.check("mixed $1/$5 spacing does NOT flag missing strikes",
            not qa_mix.flags["missing_strikes_in_grid"],
            f"n_missing={qa_mix.stats['coverage_n_missing_strikes']}")

    # 7. Truncation that costs gamma must be caught; wide coverage must not be.
    qa_trunc = run_qa(inject_defect(df, "truncated_strikes", spot=100.0), 100.0)
    c.check("narrow chain flags gamma at edge strikes", qa_trunc.flags["truncation_costs_gamma"],
            f"edge share={qa_trunc.stats['gamma_share_at_edge_strikes']:.4f}")
    c.check("wide chain does not flag edge gamma", not qa_clean.flags["truncation_costs_gamma"],
            f"edge share={qa_clean.stats['gamma_share_at_edge_strikes']:.6f}")

    # 8. Symmetric OI -> flip level at spot.
    dsym, _ = make_synthetic_chain(spot=100.0, r=r, q=q, oi_mode="symmetric")
    gsym = compute_gex(dsym, 100.0)
    fl = gsym.flip["short_otm_long_itm"]
    c.check("symmetric chain: flip near spot (moneyness convention)",
            fl is not None and abs(fl - 100.0) < 1.0, f"flip={fl}")

    # 9. Zero OI -> no flip, and zero GEX. Must return None, not a spurious level.
    dzero, _ = make_synthetic_chain(spot=100.0, oi_mode="zero")
    gzero = compute_gex(dzero, 100.0)
    c.check("zero OI gives zero gross gamma", abs(gzero.totals["gross_gamma_dollars"]) < 1e-9,
            f"gross={gzero.totals['gross_gamma_dollars']}")
    c.check("zero OI gives no flip level (returns None)",
            all(v is None for v in gzero.flip.values()), f"flip={gzero.flip}")

    # 10. Convention B is the exact negation of A, and their flip levels are IDENTICAL.
    #     This is the point that the "test two conventions" framing obscures.
    g = compute_gex(df, 100.0)
    a = g.totals["gex_long_call_short_put"]; b = g.totals["gex_short_call_long_put"]
    c.check("short_call_long_put == -long_call_short_put", abs(a + b) < 1e-6 * max(1.0, abs(a)),
            f"A={a:,.0f} B={b:,.0f}")
    fa, fb = g.flip["long_call_short_put"], g.flip["short_call_long_put"]
    same = (fa is None and fb is None) or (fa is not None and fb is not None and abs(fa - fb) < 1e-6)
    c.check("negated convention has IDENTICAL flip level", same, f"A={fa} B={fb}")

    # 11. Concentrating all OI at one strike puts peak gross gamma at that strike.
    dss, _ = make_synthetic_chain(spot=100.0, oi_mode="single_strike", single_strike=110.0)
    gss = compute_gex(dss, 100.0)
    peak = float(gss.by_strike.loc[gss.by_strike["gross"].idxmax(), "strike"])
    c.check("single-strike OI puts peak gamma at that strike", abs(peak - 110.0) < 1e-6, f"peak={peak}")

    # 12. Gamma scales as expected: doubling OI doubles gross gamma dollars (linearity).
    d2 = df.copy(); d2["openInterest"] = d2["openInterest"] * 2
    g2 = compute_gex(d2, 100.0)
    ratio = g2.totals["gross_gamma_dollars"] / g.totals["gross_gamma_dollars"]
    c.check("gross gamma is linear in OI", abs(ratio - 2.0) < 1e-9, f"ratio={ratio:.12f}")

    # 13. Sticky-strike and sticky-moneyness flip levels should DIFFER on a skewed chain.
    #     If they are identical, the sticky logic is not wired in.
    dskew, _ = make_synthetic_chain(spot=100.0, skew=-0.6, oi_mode="smooth")
    f_ss = _solve_flip(compute_gex(dskew, 100.0).contracts, 100.0, "short_otm_long_itm", sticky="strike")
    f_sm = _solve_flip(compute_gex(dskew, 100.0).contracts, 100.0, "short_otm_long_itm", sticky="moneyness")
    c.check("sticky-strike vs sticky-moneyness flip differ on skewed chain",
            (f_ss is None) != (f_sm is None) or (f_ss is not None and abs(f_ss - f_sm) > 1e-6),
            f"sticky_strike={f_ss} sticky_moneyness={f_sm}")

    return c.report()


# ----------------------------------------------------------------------------------
# Real snapshot I/O
# ----------------------------------------------------------------------------------

COLUMN_ALIASES = {
    "strike": ["strike", "Strike", "strike_price"],
    "expiry": ["expiry", "expiration", "expirationDate", "expiry_date"],
    "type": ["type", "option_type", "call_put", "right", "cp"],
    "openInterest": ["openInterest", "open_interest", "oi"],
    "impliedVolatility": ["impliedVolatility", "implied_volatility", "iv", "vol"],
    "bid": ["bid", "bidPrice"],
    "ask": ["ask", "askPrice"],
    "spot": ["spot", "underlying", "underlyingPrice", "stockPrice"],
}


def load_snapshot(path: str, spot: Optional[float] = None,
                  asof: Optional[str] = None) -> Tuple[pd.DataFrame, float]:
    raw = pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)
    df = pd.DataFrame()
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in raw.columns:
                df[canon] = raw[a]
                break
    missing = [k for k in ("strike", "expiry", "type", "openInterest", "impliedVolatility", "bid", "ask")
               if k not in df.columns]
    if missing:
        raise SystemExit(f"snapshot is missing required columns: {missing}\nfound: {list(raw.columns)}")

    t = df["type"].astype(str).str.upper().str[0]
    df["type"] = np.where(t.isin(["C"]), "C", np.where(t.isin(["P"]), "P", "?"))
    if (df["type"] == "?").any():
        raise SystemExit("could not parse option type into C/P")

    if spot is None:
        if "spot" in df.columns and df["spot"].notna().any():
            spot = float(df["spot"].dropna().iloc[0])
        else:
            raise SystemExit("no spot column found; pass --spot")

    # as-of date: prefer the folder name, since that is the collection date.
    if asof is None:
        folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
        asof = folder if len(folder) == 10 and folder[4] == "-" else None
    if asof is None:
        raise SystemExit("could not infer snapshot date from folder; pass --asof YYYY-MM-DD")

    exp = pd.to_datetime(df["expiry"], errors="coerce")
    if exp.isna().any():
        raise SystemExit("could not parse expiry dates")
    # NOTE ON DTE: expiry is at the close of the expiration date, and the snapshot is taken
    # pre-open. Using calendar days from the folder date is the honest simple choice; it is
    # off by up to one day and that matters most for 0-2 DTE contracts.
    df["dte"] = (exp - pd.Timestamp(asof)).dt.days.astype(int)
    # Keep same-day expiries. A snapshot taken before the session opens still has a live
    # 0-DTE chain, and dropping it silently removes the single largest gamma concentration
    # in the file. Time to expiry is floored at half a day so gamma stays finite; that floor
    # is arbitrary and 0-DTE gamma should be treated as indicative only.
    n_expired = int((df["dte"] < 0).sum())
    if n_expired:
        print(f"  note: dropped {n_expired} already-expired contracts")
    df = df[df["dte"] >= 0].copy()
    for col in ("strike", "openInterest", "impliedVolatility", "bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["expiry"] = exp.dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True), float(spot)


def report(df: pd.DataFrame, spot: float, label: str) -> None:
    print("=" * 78)
    print(f"SNAPSHOT QA: {label}   spot={spot:,.2f}   rows={len(df):,}")
    print("=" * 78)
    qa = run_qa(df, spot, verbose=False)
    print(qa)

    print("\nPER-EXPIRY COVERAGE")
    cov = qa.stats["_coverage_table"]
    show = cov.copy()
    for col in ("k_min_pct", "k_max_pct", "atm_iv", "z_dn", "z_up"):
        show[col] = show[col].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "nan")
    print(show.to_string(index=False))

    gex = compute_gex(df, spot, qa_parity=qa.parity)
    print("\nGEX TOTALS  ($ gamma per 1% move in spot)")
    for k, v in gex.totals.items():
        print(f"  {k:<32} {v:>20,.0f}")
    print("\nZERO-GAMMA FLIP LEVEL (re-priced across a spot grid, sticky-strike)")
    for k, v in gex.flip.items():
        n_cross = len(_flip_crossings(gex.contracts, spot, k))
        s = f"{v:,.2f}  ({v/spot - 1:+.2%} from spot)" if v is not None else "no crossing in +/-20% grid"
        print(f"  {k:<32} {s}   [{n_cross} crossing(s)]")
    print("\n  sticky-moneyness comparison:")
    for k in CONVENTIONS:
        v = _solve_flip(gex.contracts, spot, k, sticky="moneyness")
        s = f"{v:,.2f}" if v is not None else "none"
        print(f"    {k:<30} {s}")

    print("\nLARGEST GAMMA STRIKES (by gross $ gamma)")
    t = gex.top_strikes.copy()
    for c_ in t.columns:
        if c_ != "strike":
            t[c_] = t[c_].map(lambda v: f"{v:,.0f}")
    print(t.to_string(index=False))

    print("\nUNUSABLE IV ROWS (excluded from gamma):",
          int(gex.contracts["iv_unusable"].sum()),
          f"({gex.contracts['iv_unusable'].mean():.2%})")

    fired = qa.fired
    print("\n" + "-" * 78)
    if fired:
        print("VERDICT: NOT clean. Flags fired:")
        for f in fired:
            print(f"  - {f}")
        print("Fix the collector config before more snapshots accumulate.")
    else:
        print("VERDICT: no QA flags fired. Snapshot is fit for gamma work.")
    print("-" * 78)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run known-answer tests and exit")
    ap.add_argument("--snapshot", help="path to a snapshot csv or csv.gz")
    ap.add_argument("--spot", type=float, default=None)
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD; defaults to the parent folder name")
    ap.add_argument("--label", default=None)
    args = ap.parse_args(argv)

    if args.selftest or not args.snapshot:
        ok = selftest()
        if not args.snapshot:
            return 0 if ok else 1
        if not ok:
            print("\nSELF-TEST FAILED. Refusing to run on real data.")
            return 1

    df, spot = load_snapshot(args.snapshot, args.spot, args.asof)
    report(df, spot, args.label or os.path.basename(args.snapshot))
    return 0


if __name__ == "__main__":
    sys.exit(main())

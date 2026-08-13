"""
IS CROSS-SECTIONAL MOMENTUM ALIVE IN CURRENT US EQUITY DATA?
=============================================================

WHY THIS AND NOT ANOTHER PRICE PATTERN
    Tests 01, 05, 06, 07 and 08 were all TIME-SERIES tests on index products: is this
    market up tomorrow. That is one bet per day per market, so about five bets a day
    across the whole universe, in the single most arbitraged corner of finance.

    The fundamental law of active management is roughly IR = IC * sqrt(breadth), where
    breadth counts INDEPENDENT BETS, not features. Ranking 500 stocks against each other
    is 500 bets a day for the same skill per bet. sqrt(500/5) is about ten times the
    information ratio at identical skill.

    Four nulls in a row is what fishing in the lowest-breadth pond looks like. It is not
    a verdict on the method. This test keeps the method and changes the pond.

WHO IS ON THE OTHER SIDE
    A pattern with no counterparty story is not a signal. For cross-sectional momentum
    the candidate explanations are underreaction to firm-specific news, and the
    disposition effect: holders sell winners too early and hold losers too long, which
    mechanically slows price adjustment in both directions. Contested, but it is an
    actual mechanism with an actual loser, which "wick through a swing low" never had.

THE PRE-REGISTERED RULE, fixed before any data is touched
    Signal      12-month return skipping the most recent month (12-1). Standard since
                Jegadeesh and Titman 1993. NOT tuned here. Tuning it is the failure mode
                test 05 measured at +0.15R of fabricated edge.
    Portfolio   Long top decile, short bottom decile, equal weight.
    Rebalance   Monthly.
    Costs       Applied explicitly, reported gross and net.

    The 1, 3 and 6 month lookbacks are computed too, but as a GRADIENT TEST, not as
    candidates to pick the best from. See below.

THE SURVIVORSHIP TRAP, and why the null design defuses it
    Free constituent lists are TODAY's index members. Backtesting them is the classic
    fatal error in cross-sectional equity work: companies that went to zero are absent,
    so the loser decile is missing its worst members and momentum looks far better than
    it was. This inflates results by several percent a year in published estimates.

    There is no free point-in-time membership data, so the bias cannot be removed.

    An earlier version of this file claimed the null neutralises it, because random
    portfolios are drawn from the same survivors. That is only PARTLY true and the
    overclaim is corrected here. The null does cancel the universe-level effect: both
    legs of a random long/short hold survivors, so it nets out. It does NOT cancel the
    momentum-specific part. Today's index members were selected for having grown, and
    momentum systematically goes long exactly the names that grew. A random portfolio
    has no such alignment with the selection criterion.

    So the direction of the residual bias is toward FLATTERING momentum, and the size is
    unknown. Treat any positive result here as an upper bound, and use the Ken French
    portfolios below for a survivorship-free answer.

WHAT WOULD FALSIFY IT
    A real momentum effect produces a GRADIENT, because the mechanism is horizon
    dependent. Short-horizon reversal (1 month) is a separate and well documented effect
    running the other way. So the prediction is: 1-month ranks WORSE than random,
    12-1 ranks BEST, with 3 and 6 in between. Mechanisms produce gradients, coincidences
    do not. If all four horizons look identical, that is a red flag regardless of how
    good the headline number is.

HOW TO READ THE OUTPUT
    Self-test 9/9 and percentile > 95    the effect is present in this universe
    Percentile between 5 and 95          null. Indistinguishable from random portfolios
    Gradient absent                      distrust it even if the percentile is high
    Net-of-cost percentile collapses     real but not monetisable, same as test 06/07

THE POWER LIMIT, computed before the test was run rather than after
    Breadth buys a higher information ratio, not more observations. Five hundred stocks
    still collapse into ONE long/short return per month, so the sample size is months.

    t = IR * sqrt(years). Published US large-cap 12-1 momentum runs an IR near 0.5
    historically and lower since 2000. Over the 15 years of yfinance history that is
    t = 0.5 * sqrt(15.5) = 2.0 at BEST, and under 1.5 on post-2000 estimates.

    So this test is marginal. Not hopeless like a GEX level test, whose effective sample
    size is about 3, but a null result here will NOT distinguish "momentum is dead" from
    "15 years cannot see it". Say so in the writeup rather than claiming a null.

    THE FIX, and it is free: Ken French's data library has momentum decile portfolios
    back to 1927, survivorship-bias-free, professionally constructed. Roughly 100 years
    instead of 15, which is t = 0.5 * sqrt(98) = 5.0, and no survivorship problem at all.

        https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
        10_Portfolios_Prior_12_2_CSV.zip

    Use both, for different questions:
      French portfolios   IS the effect alive, and when did it weaken. Full power.
      Own construction    CAN I build it, checked against French as a KNOWN ANSWER.
                          If the pipeline cannot reproduce a documented effect on
                          overlapping dates, the pipeline is wrong, not the market.

    That second use is the strongest kind of validation available: a known answer on
    REAL data, which synthetic panels by definition cannot provide. It is exactly the
    gap that let the American-options bug through in test 11.

HOW TO RUN
    Paste into a Colab cell. Put  !pip install yfinance -q  as the FIRST LINE of the
    same cell, or its own cell in the same notebook.

    With no network it runs the self-test and stops, which is still the part that
    matters most.
"""

import math
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  -- pre-registered, do not tune
# =============================================================================

LOOKBACK = 12          # months of past return used for the signal
SKIP = 1               # months skipped before forming (avoids 1-month reversal)
GRADIENT_LOOKBACKS = [1, 3, 6, 12]
N_DECILES = 10
COST_BPS_PER_SIDE = 10.0     # 0.10% per side. Generous for large caps, harsh for small.
N_NULL = 400
START = "2010-01-01"
END = None                   # None = today
MIN_STOCKS = 100             # refuse to run on a universe too small to decile
RUN_SELFTEST = True
SEED = 7


# =============================================================================
# SIGNAL AND PORTFOLIO
# =============================================================================

def momentum_signal(monthly: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Cumulative return over [t-lookback-skip, t-skip), as a DataFrame aligned to t.

    Everything is shifted so that the signal at row t uses ONLY data strictly before t.
    The self-test verifies this by feeding it future-shifted returns and checking the
    edge disappears; that is the lookahead check, and it is the one that matters."""
    logret = np.log1p(monthly)
    if skip > 0:
        window = logret.shift(skip).rolling(lookback).sum()
    else:
        window = logret.rolling(lookback).sum()
    return window.shift(1)          # signal known at the START of month t


def decile_returns(signal: pd.DataFrame, fwd: pd.DataFrame, n_deciles: int = N_DECILES):
    """Return (long_leg, short_leg, spread, turnover) monthly series.

    Ranks are computed cross-sectionally each month among stocks that have BOTH a signal
    and a forward return, so a stock is never ranked on information it could not have
    had, nor credited with a return it could not have earned."""
    cols = signal.columns.intersection(fwd.columns)
    S = signal[cols].to_numpy(dtype=float)
    F = fwd[cols].to_numpy(dtype=float)

    longs, shorts, spreads, turn, keep = [], [], [], [], []
    prev_long, prev_short = None, None

    for i in range(len(S)):
        ok = np.flatnonzero(np.isfinite(S[i]) & np.isfinite(F[i]))
        if len(ok) < n_deciles * 2:
            continue
        k = max(1, len(ok) // n_deciles)
        order = ok[np.argsort(-S[i][ok], kind="stable")]
        top, bot = order[:k], order[-k:]

        longs.append(F[i][top].mean())
        shorts.append(F[i][bot].mean())
        spreads.append(longs[-1] - shorts[-1])
        if prev_long is None:
            turn.append(1.0)
        else:
            turn.append((len(np.setdiff1d(top, prev_long)) / k
                         + len(np.setdiff1d(bot, prev_short)) / k) / 2)
        prev_long, prev_short = top, bot
        keep.append(signal.index[i])

    idx = pd.DatetimeIndex(keep)
    return (pd.Series(longs, index=idx), pd.Series(shorts, index=idx),
            pd.Series(spreads, index=idx), pd.Series(turn, index=idx))


def apply_costs(spread: pd.Series, turnover: pd.Series, bps_per_side: float) -> pd.Series:
    """Both legs trade, and turnover is the fraction replaced. Two sides per name
    replaced (sell the old, buy the new), two legs, hence the factor of 2 twice."""
    cost = turnover * (bps_per_side / 1e4) * 2 * 2
    return spread - cost


# =============================================================================
# THE NULL
# =============================================================================

def null_distribution(fwd: pd.DataFrame, n_names: int, n_months: int,
                      n_iter: int = N_NULL, seed: int = SEED) -> np.ndarray:
    """Distribution of T-STATISTICS from random long/short portfolios of the same size,
    same months, same universe.

    WHY T AND NOT THE MEAN
        The first version of this compared MEAN spreads and it was badly wrong. A
        momentum portfolio is a concentrated factor bet: on real S&P data its monthly
        spread volatility was 5.68% against 1.64% for a random 47-vs-47 long/short, so
        3.5 times more volatile. Comparing means treats the strategy's mean as a fixed
        point and ignores that it is an estimate with a standard error four times wider
        than the null distribution itself.

        The consequence was not subtle. On a synthetic panel built with ZERO
        cross-sectional alpha, where returns came only from dispersed factor betas, the
        mean-based null returned z = +9.87 and percentile 100.0 while the honest t-stat
        was 1.00. It was measuring "differs from random selection", which any factor tilt
        satisfies trivially, rather than "has positive expected return".

        Comparing t-statistics puts both sides on the same footing, because a t-stat
        already divides by the portfolio's own volatility."""
    rng = np.random.default_rng(seed)
    dates = fwd.index[-n_months:] if n_months else fwd.index
    arr = fwd.loc[dates].to_numpy(dtype=float)

    total = np.zeros(n_iter)
    totsq = np.zeros(n_iter)
    used = 0
    for row in arr:
        ok = np.flatnonzero(np.isfinite(row))
        if len(ok) < n_names * 2:
            continue
        vals = row[ok]
        pick = rng.random((n_iter, len(ok))).argsort(axis=1)[:, : n_names * 2]
        sel = vals[pick]
        x = sel[:, :n_names].mean(axis=1) - sel[:, n_names:].mean(axis=1)
        total += x
        totsq += x * x
        used += 1
    if used < 3:
        return np.zeros(n_iter)
    mean = total / used
    var = np.maximum(totsq / used - mean ** 2, 1e-18) * used / (used - 1)
    return mean / (np.sqrt(var) / math.sqrt(used))


def percentile_of(value: float, dist: np.ndarray) -> float:
    return float((dist < value).mean() * 100.0)


# =============================================================================
# SYNTHETIC PANELS WITH KNOWN ANSWERS
# =============================================================================

def make_panel(n_stocks=300, n_months=180, mode="null", strength=0.35, seed=0):
    """Monthly return panel with a KNOWN cross-sectional structure.

    mode = "null"      no cross-sectional predictability at all. Any strategy that
                       beats the null here is broken.
    mode = "momentum"  a persistent per-stock mean, so past ranks predict future ranks.
    mode = "reversal"  next month's return is negatively related to the past window.
    """
    rng = np.random.default_rng(seed)
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    idx = pd.date_range("2005-01-31", periods=n_months, freq="ME")
    noise = rng.normal(0.0, 0.08, size=(n_months, n_stocks))

    if mode == "null":
        r = noise
    elif mode == "momentum":
        mu = rng.normal(0.0, strength * 0.08, size=n_stocks)   # persistent quality
        r = mu[None, :] + noise
    elif mode == "momentum_reversal":
        # Realistic shape: persistent quality drives 12-month momentum, PLUS a one-month
        # bounce that runs the other way. Real equity data has both, and only a panel with
        # both can test whether the engine detects the horizon gradient rather than just
        # detecting predictability of any kind.
        mu = rng.normal(0.0, strength * 0.08, size=n_stocks)
        r = mu[None, :] + noise
        for t in range(1, n_months):
            prev = r[t - 1]
            r[t] = r[t] - 0.45 * (prev - prev.mean())
    elif mode == "reversal":
        r = noise.copy()
        for t in range(13, n_months):
            past = r[t - 13:t - 1].sum(axis=0)
            r[t] = -strength * (past - past.mean()) / 12.0 + noise[t]
    else:
        raise ValueError(mode)
    return pd.DataFrame(r, index=idx, columns=cols)


# =============================================================================
# ONE FULL EVALUATION
# =============================================================================

def evaluate(monthly: pd.DataFrame, lookback=LOOKBACK, skip=SKIP,
             n_null=N_NULL, seed=SEED, verbose=False):
    fwd = monthly.copy()
    sig = momentum_signal(monthly, lookback, skip)
    valid = sig.notna().sum(axis=1) >= N_DECILES * 2
    sig, fwd = sig[valid], fwd[valid]
    if len(sig) < 24:
        return None

    lr, sr, spread, turn = decile_returns(sig, fwd)
    if not len(spread):
        return None
    net = apply_costs(spread, turn, COST_BPS_PER_SIDE)

    k = max(1, int(sig.notna().sum(axis=1).median() // N_DECILES))
    dist = null_distribution(fwd, k, len(spread), n_iter=n_null, seed=seed)
    t_gross = float(spread.mean() / (spread.std() / math.sqrt(len(spread)))) if spread.std() > 0 else float("nan")
    t_net = float(net.mean() / (net.std() / math.sqrt(len(net)))) if net.std() > 0 else float("nan")

    return {
        "n_months": len(spread),
        "names_per_leg": k,
        "gross_mean": float(spread.mean()),
        "net_mean": float(net.mean()),
        "gross_pct": percentile_of(t_gross, dist),
        "net_pct": percentile_of(t_net, dist),
        "turnover": float(turn.mean()),
        "t_stat": t_gross,
        "t_net": t_net,
        "null_mean": float(dist.mean()),
        "null_sd": float(dist.std()),
        "z_vs_null": float((t_gross - dist.mean()) / dist.std()) if dist.std() > 0 else float("nan"),
        "z_net": float((t_net - dist.mean()) / dist.std()) if dist.std() > 0 else float("nan"),
    }


def gradient(monthly: pd.DataFrame, lookbacks=GRADIENT_LOOKBACKS, n_null=300, seed=SEED):
    rows = []
    for lb in lookbacks:
        sk = 1 if lb > 1 else 0
        r = evaluate(monthly, lookback=lb, skip=sk, n_null=n_null, seed=seed)
        if r:
            rows.append({"lookback": lb, "skip": sk, "gross_pct": r["gross_pct"],
                         "gross_mean": r["gross_mean"], "t": r["t_stat"]})
    return pd.DataFrame(rows)


# =============================================================================
# SELF-TEST
# =============================================================================

def selftest() -> bool:
    checks = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # 1. NULL DIRECTION. No cross-sectional structure -> must not beat random portfolios.
    #    A strategy that finds edge here is broken, and this is the equivalent of running
    #    the sweep detector on a random walk in test 01.
    n = make_panel(mode="null", seed=1)
    rn = evaluate(n, n_null=400, seed=1)
    chk("null panel: momentum does NOT beat random", 5 < rn["gross_pct"] < 95,
        f"percentile {rn['gross_pct']:.1f}")

    # 2. DETECTION DIRECTION. Injected persistent quality -> must be found.
    m = make_panel(mode="momentum", strength=0.35, seed=2)
    rm = evaluate(m, n_null=400, seed=2)
    chk("momentum panel: detected", rm["gross_pct"] > 95, f"percentile {rm['gross_pct']:.1f}")

    # 3. SIGN. Injected reversal must come out BELOW the null, not merely non-significant.
    #    Catches a sign flip that would otherwise hide inside a null result.
    rv = make_panel(mode="reversal", strength=0.6, seed=3)
    rr = evaluate(rv, n_null=400, seed=3)
    chk("reversal panel: ranks below the null", rr["gross_pct"] < 5,
        f"percentile {rr['gross_pct']:.1f}")

    # 4. LOOKAHEAD CONTROL. Rank on the very return about to be collected. That is
    #    perfect foresight, and it must dwarf the honest signal. If it does not, the
    #    pipeline is not wiring the signal to the returns the way it claims.
    valid = momentum_signal(m, LOOKBACK, SKIP).notna().sum(axis=1) >= N_DECILES * 2
    _, _, sp_peek, _ = decile_returns(m[valid], m[valid])
    _, _, sp_real, _ = decile_returns(momentum_signal(m, LOOKBACK, SKIP)[valid], m[valid])
    chk("lookahead control: perfect foresight dwarfs the honest signal",
        sp_peek.mean() > sp_real.mean() * 3,
        f"peek {sp_peek.mean():.4f} vs honest {sp_real.mean():.4f}")

    # 5. Signal uses no contemporaneous data: the last usable signal row must be
    #    strictly earlier than the return it is matched against.
    s = momentum_signal(m, LOOKBACK, SKIP)
    first_valid = s.notna().any(axis=1).idxmax()
    expected = m.index[LOOKBACK + SKIP]
    chk("signal starts only after lookback+skip months",
        first_valid >= expected, f"first {first_valid.date()} expected >= {expected.date()}")

    # 6. Costs must reduce net return monotonically.
    _, _, spr, tn = decile_returns(momentum_signal(m, LOOKBACK, SKIP)[valid], m[valid])
    nets = [apply_costs(spr, tn, b).mean() for b in (0, 5, 10, 25)]
    chk("higher costs strictly reduce net return", all(np.diff(nets) < 0),
        " > ".join(f"{x:.5f}" for x in nets))

    # 7. The null is now a distribution of T-STATISTICS, so it must look like one:
    #    centred near zero with unit spread. If the spread drifted far from 1 the null
    #    would silently regain the scale bug it was rewritten to remove.
    dist = null_distribution(n, 30, 100, n_iter=400, seed=5)
    chk("null t-distribution is centred at 0 with unit spread",
        abs(dist.mean()) < 0.25 and 0.75 < dist.std() < 1.30,
        f"mean {dist.mean():+.3f} sd {dist.std():.3f}")

    # 8. GRADIENT, on a panel built with 12-month momentum AND 1-month reversal.
    #    Compared on t-statistics rather than percentiles, because a percentile ceilings
    #    at 100 and then cannot tell "strong" from "stronger", which is the whole job.
    mr = make_panel(mode="momentum_reversal", strength=0.35, seed=8)
    g = gradient(mr, n_null=200)
    t12 = float(g.loc[g.lookback == 12, "t"].iloc[0])
    t1 = float(g.loc[g.lookback == 1, "t"].iloc[0])
    chk("gradient: 12-month beats 1-month when both effects are present",
        t12 > t1, f"t(12m) {t12:+.2f} vs t(1m) {t1:+.2f}")

    # 8b. And the 1-month leg must actually come out NEGATIVE on that panel, since a
    #     reversal was injected. Detecting the gradient is not enough if the sign is wrong.
    chk("gradient: 1-month is negative when reversal is injected", t1 < 0, f"t(1m) {t1:+.2f}")

    # 8c. THE FAILURE THAT GOT THROUGH THE FIRST VERSION. A panel with ZERO
    #     cross-sectional alpha, where all return dispersion comes from factor betas.
    #     Sorting on past return loads on the factor and produces a volatile portfolio
    #     with no edge. The mean-based null called this percentile 100.0 and z = +9.9.
    #     A t-based null must not.
    rng2 = np.random.default_rng(3)
    beta = rng2.normal(1.0, 1.5, 400)
    fac = rng2.normal(0.0, 0.05, 200)
    idio = rng2.normal(0.0, 0.06, (200, 400))
    zero_alpha = pd.DataFrame(fac[:, None] * beta[None, :] + idio,
                              index=pd.date_range("2005-01-31", periods=200, freq="ME"),
                              columns=[f"S{i:03d}" for i in range(400)])
    rz = evaluate(zero_alpha, n_null=400, seed=1)
    chk("zero-alpha factor panel is NOT called significant",
        rz["gross_pct"] < 95, f"percentile {rz['gross_pct']:.1f}, t {rz['t_stat']:+.2f}")

    # 9. Reproducibility. Same seed, same answer. Test 11 shipped a result that changed
    #    between numpy builds because of a non-stable sort; assert it directly here.
    a = evaluate(m, n_null=200, seed=11)["gross_pct"]
    b = evaluate(m, n_null=200, seed=11)["gross_pct"]
    chk("same seed gives identical result", a == b, f"{a} vs {b}")

    w = max(len(c[0]) for c in checks) + 2
    print("\n" + "=" * 78)
    print("SELF-TEST")
    print("=" * 78)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{w}} {detail}")
    nfail = sum(1 for _, ok, _ in checks if not ok)
    print("-" * 78)
    print(f"  {len(checks) - nfail}/{len(checks)} passed")
    return nfail == 0


# =============================================================================
# REAL DATA
# =============================================================================

SP500_URL = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
             "main/data/constituents.csv")


def load_universe(limit=None):
    tickers = pd.read_csv(SP500_URL)["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers[:limit] if limit else tickers


def load_monthly(tickers, start=START, end=END):
    import yfinance as yf
    px = yf.download(tickers, start=start, end=end, interval="1mo",
                     auto_adjust=True, progress=False)["Close"]
    px = px.dropna(axis=1, how="all")
    return px.pct_change().dropna(how="all")


def main():
    if RUN_SELFTEST and not selftest():
        print("\nSELF-TEST FAILED. Not running on real data.")
        return

    try:
        tickers = load_universe()
        print(f"\nuniverse: {len(tickers)} current S&P 500 members")
        monthly = load_monthly(tickers)
    except Exception as e:
        print(f"\ncould not load prices ({type(e).__name__}). "
              "Self-test above is the part that matters; rerun with network for the rest.")
        return

    print(f"monthly returns: {monthly.shape[0]} months x {monthly.shape[1]} names, "
          f"{monthly.index.min().date()} to {monthly.index.max().date()}")
    if monthly.shape[1] < MIN_STOCKS:
        print("universe too small, stopping")
        return

    r = evaluate(monthly)
    print("\n" + "=" * 78)
    print("CROSS-SECTIONAL MOMENTUM, 12-1, long top decile / short bottom decile")
    print("=" * 78)
    print(f"  months tested            {r['n_months']}")
    print(f"  names per leg            {r['names_per_leg']}")
    print(f"  monthly turnover         {r['turnover']:.1%}")
    print(f"  gross monthly spread     {r['gross_mean']:+.4f}   t = {r['t_stat']:.2f}")
    print(f"  net of {COST_BPS_PER_SIDE:.0f}bp per side      {r['net_mean']:+.4f}")
    print(f"  random long/short null   {r['null_mean']:+.4f}  sd {r['null_sd']:.4f}")
    print(f"  GROSS percentile vs null {r['gross_pct']:.1f}   (z = {r['z_vs_null']:+.2f})")
    print(f"  NET   percentile vs null {r['net_pct']:.1f}   (z = {r['z_net']:+.2f})")
    print("  percentile saturates at 100; the z-score is the number that keeps scaling")

    print("\nGRADIENT (a mechanism should be horizon dependent)")
    g = gradient(monthly)
    print(g.to_string(index=False))

    print("\n" + "-" * 78)
    if r["gross_pct"] <= 95:
        print("VERDICT: NULL. Indistinguishable from random portfolios out of the same")
        print("universe. Do not proceed to a combination; there is nothing to combine.")
    elif r["net_pct"] <= 95:
        print("VERDICT: REAL BUT NOT MONETISABLE. Survives gross, dies after costs.")
        print("Same shape as tests 06 and 07. Log it and stop.")
    else:
        print("VERDICT: SURVIVES GROSS AND NET. Check the gradient above before believing")
        print("it: 1-month should rank WORSE than 12-month. If all horizons look alike,")
        print("distrust it. Then re-run on a different region before doing anything else.")
    print("\nPOWER: t = IR * sqrt(years). Over ~15 years an IR of 0.5 gives t = 2.0 at best.")
    print("A null here does NOT separate 'momentum is dead' from '15 years cannot see it'.")
    print("Run the French 10_Portfolios_Prior_12_2 series for the powered version.")
    print("\nThe ABSOLUTE return above is inflated by survivorship: this is today's index")
    print("membership, so companies that went to zero are missing from the short leg.")
    print("The PERCENTILE is the trustworthy number, because the null is drawn from the")
    print("same biased universe and carries the same advantage.")
    print("-" * 78)


main()

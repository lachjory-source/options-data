"""
PROP FIRM EVALUATION SIMULATOR
==============================

The question this answers:

    Given a strategy with a known expectancy, what is my ACTUAL probability
    of hitting a 10% profit target before breaching a 5% daily loss limit
    or a 10% maximum drawdown?

Nobody computes this. Everyone argues about entries instead. But the
evaluation is a race between two barriers, and which barrier you hit first
depends at least as much on position size as on whether your strategy works.

WHAT YOU WILL PROBABLY FIND
    1. With zero edge, your pass probability is nowhere near 50%. The daily
       loss limit is a one-sided barrier: it can end your account but it can
       never end it in your favour.
    2. Position size matters more than expectancy across most of the range.
    3. There is an optimum. Too small and you time out or grind; too large
       and the daily limit gets you. Most people sit far to the right of it.
    4. Fat tails from gaps and slippage move the answer a lot, because the
       daily limit is triggered by your worst days, not your average day.

HOW TO RUN
    Colab: paste into a cell, Shift+Enter. Pure simulation, no downloads.
    Takes about a minute.

ASSUMPTIONS, stated plainly (change them, they are all arguable)
    - Risk per trade is a fixed % of the STARTING balance, not compounding.
    - Drawdown limits are measured against the starting balance (static) or
      the equity peak (trailing). Both are offered.
    - The daily loss limit resets at the start of each trading day.
    - You must trade at least `min_days` before a payout/pass counts.
    - Trades within a day are independent. Real trading is not: losses cluster
      because volatility clusters and because you tilt. This simulator is
      therefore OPTIMISTIC. Reality has more bad days than this model.
"""

from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# RULES AND MODEL
# =============================================================================

@dataclass
class EvalRules:
    """Typical two-step-ish evaluation. Check your actual firm's numbers."""
    profit_target_pct: float = 10.0
    daily_loss_pct: float = 5.0
    max_dd_pct: float = 10.0
    trailing_dd: bool = False      # True = drawdown trails your equity peak
    min_days: int = 5
    max_days: int = 60
    trades_per_day: int = 3


@dataclass
class TraderModel:
    risk_pct: float = 0.5          # % of starting balance risked per trade
    expectancy_r: float = 0.0      # target mean R per trade
    r_dist: np.ndarray = None      # shape of the R distribution (mean gets shifted)


# -----------------------------------------------------------------------------
# R distributions
# -----------------------------------------------------------------------------

def dist_clean(win_rate=0.40, rr=2.0, n=20000, seed=0):
    """Textbook: every loss is exactly -1R. No gaps, no slippage, no fat tail.
    This is the distribution most people implicitly assume they have."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < win_rate, rr, -1.0)


def dist_with_gaps(win_rate=0.40, rr=2.0, gap_rate=0.15, n=20000, seed=0):
    """
    Realistic. Some losses blow through the stop.

    Calibrated to what your SPY backtest actually produced: about 15% of
    trades gapped through the stop, averaging -2.3R against a planned -1R.
    """
    rng = np.random.default_rng(seed)
    r = np.where(rng.random(n) < win_rate, rr, -1.0)
    losers = np.where(r < 0)[0]
    n_gap = int(len(losers) * gap_rate / (1 - win_rate)) if win_rate < 1 else 0
    n_gap = min(n_gap, len(losers))
    hit = rng.choice(losers, n_gap, replace=False)
    # gap losses: mean about -2.3R, occasionally far worse
    r[hit] = -1.0 - np.abs(rng.gamma(shape=1.6, scale=0.85, size=n_gap))
    return r


def dist_from_csv(path="sweep_trades.csv"):
    """Use your own backtested trades. The realistic option if you have them."""
    import pandas as pd
    return pd.read_csv(path)["r_net"].values


def shift_to(r_dist, target_expectancy):
    """Move the distribution's mean to a target while keeping its SHAPE.
    Lets us ask 'what if this exact pattern of wins and losses had an edge?'"""
    return r_dist - r_dist.mean() + target_expectancy


# =============================================================================
# THE SIMULATION  (vectorised: every run advances together)
# =============================================================================

def simulate(rules, model, n_sims=20000, seed=1):
    rng = np.random.default_rng(seed)
    r_pool = shift_to(model.r_dist, model.expectancy_r)

    START = 100.0
    risk = model.risk_pct                      # equity units, since START = 100
    target_level = START + rules.profit_target_pct
    static_floor = START - rules.max_dd_pct

    equity = np.full(n_sims, START)
    peak = np.full(n_sims, START)
    day_start = np.full(n_sims, START)
    active = np.ones(n_sims, dtype=bool)
    outcome = np.zeros(n_sims, dtype=np.int8)  # 0 running 1 pass 2 daily 3 dd 4 timeout
    days_done = np.zeros(n_sims, dtype=np.int32)
    end_day = np.zeros(n_sims, dtype=np.int32)

    for day in range(rules.max_days):
        day_locked = np.zeros(n_sims, dtype=bool)   # hit daily limit, done for today

        for _ in range(rules.trades_per_day):
            live = active & ~day_locked
            if not live.any():
                break
            r = rng.choice(r_pool, live.sum())
            equity[live] += r * risk
            peak = np.maximum(peak, equity)

            # --- losing conditions checked FIRST (worst case) ---
            floor = (peak - rules.max_dd_pct) if rules.trailing_dd else static_floor
            bust_dd = live & (equity <= floor)
            outcome[bust_dd] = 3
            end_day[bust_dd] = day + 1
            active[bust_dd] = False

            live = active & ~day_locked
            bust_day = live & (equity <= day_start - rules.daily_loss_pct)
            outcome[bust_day] = 2
            end_day[bust_day] = day + 1
            active[bust_day] = False

            # daily limit reached but not breached -> stop trading for today
            live = active & ~day_locked
            day_locked |= live & (equity <= day_start - rules.daily_loss_pct * 0.999)

            # --- pass condition ---
            live = active & ~day_locked
            won = live & (equity >= target_level) & (days_done >= rules.min_days - 1)
            outcome[won] = 1
            end_day[won] = day + 1
            active[won] = False

        days_done[active] += 1
        day_start[active] = equity[active]

    outcome[active] = 4
    end_day[active] = rules.max_days

    n = n_sims
    return {
        "pass": (outcome == 1).mean(),
        "fail_daily": (outcome == 2).mean(),
        "fail_dd": (outcome == 3).mean(),
        "timeout": (outcome == 4).mean(),
        "median_days": float(np.median(end_day[outcome == 1])) if (outcome == 1).any() else float("nan"),
        "n": n,
    }


# =============================================================================
# VALIDATION  -- the simulator has to get a known answer right
# =============================================================================

def validate():
    print("=" * 66)
    print("STEP 0: DOES THE SIMULATOR ITSELF WORK?")
    print("=" * 66)
    print("With zero edge, tiny position size, no daily limit and no time limit,")
    print("reaching +10% before -10% is a coin flip. Anything far from 50% means")
    print("the account mechanics are wrong and every number below is garbage.\n")

    # Step size must be big enough to actually reach a barrier in the time
    # allowed, or everything just times out and the test measures nothing.
    # At 1% risk and +-1R outcomes, the barriers are 10 steps away each.
    rules = EvalRules(profit_target_pct=10, daily_loss_pct=999, max_dd_pct=10,
                      min_days=0, max_days=3000, trades_per_day=1)
    model = TraderModel(risk_pct=1.0, expectancy_r=0.0,
                        r_dist=dist_clean(win_rate=0.5, rr=1.0))
    res = simulate(rules, model, n_sims=20000, seed=7)
    print(f"  pass {res['pass']:.1%}   dd {res['fail_dd']:.1%}   "
          f"timeout {res['timeout']:.1%}")
    ok = abs(res["pass"] - 0.5) < 0.03 and res["timeout"] < 0.02
    print(f"  -> {'PASS: mechanics are sound.' if ok else 'FAIL: mechanics are wrong.'}\n")
    return ok


# =============================================================================
# REPORTS
# =============================================================================

def report_baseline(rules):
    print("=" * 66)
    print("STEP 1: YOUR BASELINE  (zero edge, realistic fat tails)")
    print("=" * 66)
    print("This is the honest starting point. Not 50%.\n")
    print(f"  {'risk/trade':>11} {'pass':>8} {'daily limit':>12} {'max DD':>9} {'timeout':>9}")
    print("  " + "-" * 52)
    for rp in [0.25, 0.5, 1.0, 2.0, 3.0]:
        m = TraderModel(risk_pct=rp, expectancy_r=0.0, r_dist=dist_with_gaps())
        r = simulate(rules, m, n_sims=20000, seed=11)
        print(f"  {rp:>10.2f}% {r['pass']:>8.1%} {r['fail_daily']:>12.1%} "
              f"{r['fail_dd']:>9.1%} {r['timeout']:>9.1%}")
    print("\n  Note which column does the killing.")


def report_grid(rules):
    print("\n" + "=" * 66)
    print("STEP 2: HOW MUCH EDGE DO YOU ACTUALLY NEED?")
    print("=" * 66)
    print("Pass probability. Rows = expectancy in R per trade, cols = risk per trade.\n")
    risks = [0.25, 0.5, 1.0, 2.0]
    exps = [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    print(f"  {'expectancy':>11} " + "".join(f"{r:>9.2f}%" for r in risks))
    print("  " + "-" * 51)
    best = {}
    for e in exps:
        row = []
        for rp in risks:
            m = TraderModel(risk_pct=rp, expectancy_r=e, r_dist=dist_with_gaps())
            p = simulate(rules, m, n_sims=8000, seed=13)["pass"]
            row.append(p)
        best[e] = (risks[int(np.argmax(row))], max(row))
        print(f"  {e:>+10.2f}R " + "".join(f"{p:>9.1%}" for p in row))

    print("\n  Best risk size at each expectancy:")
    for e, (rp, p) in best.items():
        print(f"    {e:>+6.2f}R  ->  risk {rp:.2f}%  gives {p:.1%}")

    need = [e for e, (rp, p) in best.items() if p >= 0.5]
    print()
    if need:
        print(f"  Minimum expectancy tested that reaches a 50% pass rate: {min(need):+.2f}R")
    else:
        print("  NONE of the tested expectancies reach a 50% pass rate.")


def report_tails(rules):
    print("\n" + "=" * 66)
    print("STEP 3: WHAT DO THE FAT TAILS COST YOU?")
    print("=" * 66)
    print("Same win rate, same expectancy. The only difference is whether some")
    print("losses blow through the stop, as 15% of yours did on SPY.\n")
    print(f"  {'expectancy':>11} {'clean -1R losses':>18} {'with gap tail':>15} {'cost':>8}")
    print("  " + "-" * 55)
    for e in [0.0, 0.10, 0.20]:
        a = simulate(rules, TraderModel(1.0, e, dist_clean()), n_sims=12000, seed=17)["pass"]
        b = simulate(rules, TraderModel(1.0, e, dist_with_gaps()), n_sims=12000, seed=17)["pass"]
        print(f"  {e:>+10.2f}R {a:>18.1%} {b:>15.1%} {a - b:>+8.1%}")
    print("\n  Your planned risk is a floor on what you can lose, not a ceiling.")


def report_your_strategy(rules):
    """Uses your real backtested trades if sweep_trades.csv is sitting alongside."""
    try:
        r = dist_from_csv()
    except Exception:
        return
    print("\n" + "=" * 66)
    print("STEP 4: YOUR ACTUAL BACKTESTED TRADES")
    print("=" * 66)
    print(f"  Loaded {len(r)} trades, expectancy {r.mean():+.4f}R, "
          f"worst {r.min():.2f}R\n")
    for rp in [0.5, 1.0, 2.0]:
        m = TraderModel(risk_pct=rp, expectancy_r=r.mean(), r_dist=r)
        res = simulate(rules, m, n_sims=20000, seed=23)
        print(f"  risk {rp:.2f}%  ->  pass {res['pass']:.1%}   "
              f"daily-limit {res['fail_daily']:.1%}   maxDD {res['fail_dd']:.1%}")


def main():
    if not validate():
        print("Stopping: fix the mechanics before reading anything else.")
        return
    rules = EvalRules()
    print(f"RULES: {rules.profit_target_pct:.0f}% target, "
          f"{rules.daily_loss_pct:.0f}% daily loss limit, "
          f"{rules.max_dd_pct:.0f}% max drawdown,")
    print(f"       {rules.trades_per_day} trades/day, "
          f"{rules.max_days} day limit, min {rules.min_days} days.\n")
    report_baseline(rules)
    report_grid(rules)
    report_tails(rules)
    report_your_strategy(rules)
    print("\n" + "=" * 66)
    print("Reminder: this model assumes trades are independent. Real losses")
    print("cluster, because volatility clusters and because people tilt after")
    print("a bad day. Every number above is therefore optimistic.")
    print("=" * 66)


if __name__ == "__main__":
    main()

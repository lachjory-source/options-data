## 13 — Do the popular chart indicators work? Smart Money Concepts

The most-trending script on TradingView, tested with a matched random-entry null.

**Method.** The LuxAlgo SMC indicator converted to a Pine `strategy()`, stripped to the
structure-break signal only. Enter on internal CHoCH, stop at the opposing pivot, target
2R, risk 1% of equity per trade, 0.05% commission and 2 ticks slippage. SOLUSD 1h,
Binance, 173 trades.

**The null is the point.** TradingView benchmarks against buy-and-hold, which this repo
rejects. So the control was built into the script: same entry bars, same stop, same
target, direction chosen by a seeded coin flip. That isolates the only thing SMC actually
claims, which is that a CHoCH tells you which way.

| | net | R |
|---|---|---|
| **Signal** | **-1,960** | **-1.96** |
| random seed 7 | +13,325 | +13.32 |
| random seed 4 | +4,119 | +4.12 |
| random seed 3 | +2,477 | +2.48 |
| random seed 6 | +2,365 | +2.37 |
| random seed 5 | -3,460 | -3.46 |
| random seed 2 | -5,716 | -5.72 |
| random seed 8 | -10,148 | -10.15 |
| random seed 10 | -14,243 | -14.24 |
| random seed 1 | -17,525 | -17.52 |
| random seed 9 | -27,753 | -27.75 |

**Signal ranks 5th of 11. Four of ten random runs beat it. z = +0.31.**

**NULL.** A CHoCH carries no information about direction.

### The number worth keeping

Seed 7 returned **+13.3% on a 100,000 account from coin flips**. Same bars, same stops,
same targets, random direction. Seed 9 lost 27.8% doing the identical thing. The spread of
pure chance across 173 trades is **41R**.

That is the second reusable measurement in this repo, alongside the +0.15R optimisation
lift from test 05. Both quantify the same hazard from different angles: 05 measures what
parameter tuning invents, 13 measures what a coin flip invents. A backtest has to clear
both bars before it means anything, and a 13% equity curve clears neither.

### What this does not say

Not that SMC is dead. 173 trades cannot resolve an edge below about 0.14R per trade, and
real edges live well below that. It says there is no LARGE effect on this market.

The trade count was capped by TradingView's free-plan history, not by the question. The
same test in Python on Binance archives would give 10 to 50 times the sample.

It also tests the bare structure break, not SMC with order-block and premium/discount
confluence. That objection is fair, but confluence filters cut the trade count further
while adding tunable choices, so a confluence version needs MORE data to say anything, not
less. Pre-register one rule and rerun the eleven-way comparison; do not try several and
keep the best.

### Bugs found in the source indicator

Reviewing the LuxAlgo script before converting it:

1. **Lookahead bias in the fair value gap function.** `request.security(...)` is called with
   `high[0]`, `low[0]` and `lookahead_on`. On a higher timeframe that returns the completed
   value of a bar still forming, so it reads the future. Any backtest including FVGs is
   invalid. The MTF levels function gets this right by using `[1]` offsets; the FVG one
   does not.
2. **Array elements removed during iteration**, in both `deleteOrderBlocks()` and
   `deleteFairValueGaps()`. Removing index i shifts everything down, so the loop skips the
   next element. Mitigated blocks are missed and their alerts never fire.
3. **Misplaced parenthesis in the confluence filter.** `math.min(close, open - low)` should
   be `math.min(close, open) - low`. Returns the wrong lower wick on bearish bars. Off by
   default.
4. **Order block boxes can invert**, because high and low are deliberately swapped on
   volatile bars and a swapped bar can still be selected.

None of these are why the strategy is null. It is null because the signal has no edge.

### The hindsight illusion, measured

A pivot at bar N cannot be confirmed until bar N+size, so swing structure at the default
setting is 50 bars late and internal structure 5 bars late. The indicator draws the label
back at bar N, so scrolling through history shows structure that appears to have been
knowable at the time. It was not.

The strategy correctly refuses to see it early. The gap between how the indicator looks on
a chart and how the strategy performs IS that illusion, finally quantified.

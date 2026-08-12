# options-data
Daily archive of US equity index option chains. Started August 2026 to build a positioning dataset that isn't available historically.
## Data notes

Daily snapshots of SPY, QQQ and IWM option chains, all expirations within
90 days. Raw data only — gamma exposure is deliberately not computed at
collection time, so the dealer sign convention can be tested later rather
than assumed.

**Date offset:** open interest in a folder dated `YYYY-MM-DD` reflects the
**previous** trading session's settled positions, because the OCC publishes
OI overnight. The spot price in the same folder is from that morning,
pre-open. Account for this when aligning snapshots to returns.

Collected automatically at 13:00 UTC each weekday via GitHub Actions.
The first snapshot (2026-08-12) was a manual run at a different time of day.

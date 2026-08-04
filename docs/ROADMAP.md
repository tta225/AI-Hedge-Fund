# Roadmap

Honest status. Nothing below is described as finished unless it is finished and tested.

---

## Built and working

| Area | Status |
|---|---|
| Domain model — instruments, bars, sides, orders, tick arithmetic | ✅ tested |
| Validated OHLCV series with hard integrity invariants | ✅ tested |
| Data provenance with launder-proof taint propagation | ✅ tested |
| ICT session clock, DST-correct, all killzones + Silver Bullet + macros | ✅ tested |
| Provider abstraction; CSV, yfinance, synthetic adapters; OpenBB quarantined | ✅ |
| **ICT engine** — swings, BOS/CHoCH/MSS, FVG + inversion, order blocks + breakers, liquidity pools + sweeps, dealing range/OTE, SMT | ✅ tested |
| Anti-lookahead discipline (`origin_index` / `confirmed_index`) | ✅ tested directly |
| Two reference ICT strategies | ✅ |
| Risk-first sizing and hard limits incl. kill switch | ✅ tested |
| Simulated venue with pessimistic fill model | ✅ tested |
| Order router as single choke point, live-venue refusal | ✅ tested |
| Portfolio accounting incl. reversal-through-flat | ✅ tested |
| Event-driven backtester with OCO brackets and signal funnel | ✅ tested |
| Performance metrics with daily-resampled Sharpe/Sortino | ✅ tested |
| Five-agent pipeline, LLM-optional, no execute stage | ✅ |
| Operator terminal | ✅ |
| CLI: `demo`, `analyse`, `terminal`, `backtest`, `pipeline`, `config` | ✅ |

**130 tests passing.**

---

## Known limitations

These are deliberate deferrals, documented so they are not mistaken for oversights.

1. **Cash ledger is float64.** Negligible for research and paper trading relative to
   slippage assumptions; a live ledger should use fixed-point. *(`core/types.py`,
   `portfolio/positions.py`)*

2. **Realised risk can exceed planned risk.** Sizing uses the signal's intended entry, but
   fills occur at the next bar's open plus slippage. When the open gaps away from the
   intended entry, actual risk exceeds the budget — observed at roughly 1.3–1.4× planned
   risk on synthetic data. A production system should re-size at fill time, or reject fills
   beyond a maximum entry deviation. **This is the highest-priority correctness item.**

3. **Single-instrument backtests.** The portfolio supports multiple positions; the
   backtester loops one series. Portfolio-level backtesting is not built.

4. **No walk-forward or parameter-stability tooling.** Everything needed to overfit is
   present; the tooling to detect it is not. See below.

5. **No live venue adapter.** Deliberate — the interface exists, the implementation does
   not, and the router refuses to reach one.

6. **Confluence weights are priors, not estimates.** Stated in the code and in
   `ICT_METHODOLOGY.md`. They should be replaced by measured hit rates.

7. **The synthetic generator is not a market simulator.** It produces structure for testing.
   It has no microstructure, no fat tails, no regime persistence beyond an AR(1) drift, and
   no news. Never treat its output as a distribution over real outcomes.

---

## Next, in priority order

### 1. Real data validation — *the only thing that matters right now*

Everything is built. Nothing is validated. Until real ES/NQ/SPY history has been run through
the ICT engine and the strategies, there is **no evidence any of this has an edge**.

* Wire a real feed (Databento for futures, Polygon or Alpaca for equities)
* Measure the base rates the methodology assumes: how often does a swept pool actually
  reverse? how often is an unmitigated FVG revisited? what is the hit rate of an order block
  by quality tier?
* Replace confluence priors with those measured rates

### 2. Fix the risk-slippage gap

Limitation #2 above. Re-size at fill or enforce a maximum entry deviation.

### 3. Overfitting defences

* Walk-forward analysis with strict in-sample/out-of-sample separation
* Parameter-stability surfaces — an edge that survives only at one threshold is not an edge
* Monte Carlo trade-order reshuffling for drawdown distribution
* Multiple-testing correction, since every threshold tried is a hypothesis tested

### 4. Portfolio-level backtesting

Multi-instrument, correlation-aware exposure limits, cross-instrument risk budgeting.

### 5. Live path (paper first, and for a long time)

* Alpaca paper adapter — fastest route to real fills
* IBKR adapter for futures/FX/equities under one account
* Reconciliation between internal portfolio state and broker state
* Heartbeat monitoring and automatic kill-switch triggers on disconnect or drift

### 6. Terminal depth

* Live-updating layout (`rich.Live`)
* Command line with a Bloomberg-style mnemonic grammar
* Candle rendering with ICT zones overlaid
* Multi-symbol workspace

### 7. ICT coverage

* Optimal trade entry automation across timeframes
* Power of Three (accumulation / manipulation / distribution) daily profiling
* IPDA data ranges (20/40/60-day lookbacks)
* Weekly and daily bias templates
* Standard-deviation projections wired into targets

---

## Explicitly not planned

* **Anything claiming predicted returns.** The platform reports measured results and refuses
  to present generated data as performance. That property is load-bearing.
* **Auto-enabling live trading.** Live requires a mode change, a confirmation phrase, and a
  disengaged kill switch. That friction is the feature.

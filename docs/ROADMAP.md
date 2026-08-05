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
| CLI: `demo`, `analyse`, `terminal`, `backtest`, `pipeline`, `config`, `regime`, `research` | ✅ |
| **Risk-budget guarantee** — entry tolerance + stop-gap allowance, venue-enforced | ✅ tested |
| **Gaussian HMM** — Baum-Welch, causal filter, incremental step, Viterbi (research-only) | ✅ tested |
| **Causal regime detection** — expanding-window refit, no lookahead in fit or inference | ✅ tested |
| **Quant strategies** — momentum, mean reversion, volatility breakout | ✅ |
| **Regime-gated strategies** — the ICT/quant merge, rejects non-causal regime input | ✅ tested |
| **Strategy search** — walk-forward, deflated Sharpe, PBO, parameter stability | ✅ tested |
| **ICT Unicorn model** — breaker ∩ FVG overlap | ✅ tested |
| **ICT Power of Three** — accumulation/manipulation/distribution, causal | ✅ tested |
| **Hugging Face adapter** — load your own datasets with provenance | ✅ |
| **Alpaca adapter** — real US equity/crypto bars, REST, paginated | ✅ |
| **ICT rejection blocks** — wick-based zone, level-anchored | ✅ tested |
| **ICT turtle soup** — failed breakout, reclaim + displacement | ✅ tested |
| **CRT** — HTF candle range sweep (community-attributed, not ICT-original) | ✅ tested |
| **IPDA data ranges** — 20/40/60 trading-day extremes, causal | ✅ tested |

**233 tests passing.**

---

## Known limitations

These are deliberate deferrals, documented so they are not mistaken for oversights.

1. **Cash ledger is float64.** Negligible for research and paper trading relative to
   slippage assumptions; a live ledger should use fixed-point. *(`core/types.py`,
   `portfolio/positions.py`)*

2. **Stop-gap tail risk is bounded but not eliminated.** ~~Realised risk exceeds the
   budget~~ — *fixed*. Sizing now budgets for an entry tolerance and an expected
   stop-gap, and the venue expires entries that would fill beyond tolerance. Worst
   observed loss fell from **7.20× the per-trade budget to 0.57×** on the synthetic
   fixture.

   What remains is genuine: a stop gapping *further* than the allowance cannot be
   prevented by any order type, because the position is already open. It is reported
   as `worst_loss_vs_budget_x`, never hidden. Raising `stop_gap_atr` shrinks size and
   shrinks the residual; only options remove it.

3. **Single-instrument backtests.** The portfolio supports multiple positions; the
   backtester loops one series. Portfolio-level backtesting is not built.

4. **HMM regime detection is slow on long series.** The causal fit refits every
   `refit_every` bars, each an EM run. A year of hourly data takes ~35s. Fine for
   research, too slow to put inside a large parameter sweep.

5. **The strategy search's skill bar is dispersion-sensitive.** The deflated Sharpe
   threshold scales with the variance of results across trials, so a candidate set
   mixing structurally opposite strategies (momentum *and* mean reversion) raises the
   bar until nothing can clear it. That is the formula behaving correctly, but it
   means comparing like with like matters. The threshold is printed so this is visible.

6. **No live venue adapter.** Deliberate — the interface exists, the implementation does
   not, and the router refuses to reach one.

7. **Confluence weights are priors, not estimates.** Stated in the code and in
   `ICT_METHODOLOGY.md`. They should be replaced by measured hit rates.

8. **The synthetic generator is not a market simulator.** It produces structure for testing.
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

### 2. Feed your Hugging Face dataset through the platform

`HuggingFaceProvider` is built. It needs the dataset id, and `HF_TOKEN` if the
dataset is private. Run `provider.inspect()` first to confirm the schema, and
set `kind=DataKind.SYNTHETIC` if the data is generated rather than observed.

### 3. Overfitting defences

* ~~Walk-forward analysis~~ **built** (`axiom.research.walk_forward_splits`)
* ~~Parameter-stability surfaces~~ **built** (`parameter_stability`)
* ~~Multiple-testing correction~~ **built** (deflated Sharpe + PBO)
* Monte Carlo trade-order reshuffling for drawdown distribution — still to do
* Regime-conditional performance attribution — which regimes does an edge live in?

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
* ~~Power of Three~~ **built** (`axiom.ict.power_of_three`)
* ~~Unicorn model~~ **built** (`axiom.ict.find_unicorns`)
* ~~IPDA data ranges~~ **built** (`axiom.ict.compute_ipda_levels`)
* ~~Turtle soup~~ **built** (`axiom.ict.find_turtle_soups`)
* ~~Rejection blocks~~ **built** (`axiom.ict.find_rejection_blocks`)
* ~~CRT~~ **built** (`axiom.ict.find_crt_setups`) — community-attributed
* Quarterly theory, propulsion/vacuum blocks, BPR, named models in `31-models/`
* Weekly and daily bias templates
* Standard-deviation projections wired into targets

---

## Explicitly not planned

* **Anything claiming predicted returns.** The platform reports measured results and refuses
  to present generated data as performance. That property is load-bearing.
* **Auto-enabling live trading.** Live requires a mode change, a confirmation phrase, and a
  disengaged kill switch. That friction is the feature.

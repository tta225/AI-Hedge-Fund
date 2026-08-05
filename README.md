# AXIOM

**Proprietary AI hedge fund platform — ICT alpha engine, research terminal, and paper-first execution.**

AXIOM is a systematic trading platform built around three ideas:

1. **The ICT structural model is the alpha layer.** Order blocks, fair value gaps, liquidity sweeps, market structure shifts, killzones, and SMT divergence are implemented as a first-class typed engine, not as chart annotations.
2. **A backtest that cannot be trusted is worse than no backtest.** Anti-lookahead discipline, mandatory data provenance, and always-on transaction costs are enforced by the type system and the test suite — not by convention.
3. **Nothing reaches a live venue by accident.** Paper and simulation are the default. Live routing requires an explicit mode, an explicit confirmation phrase, and a disengaged kill switch.

> ⚠️ **This is research and engineering infrastructure, not financial advice.** It makes no claim of profitability, and none of the bundled strategies has been validated on real market data. Backtest results on generated data are correctness checks and nothing more. Paper trade first. Only risk capital you can afford to lose.

---

## Quick start

> **Note:** the work is on the branch `claude/ai-hedge-fund-platform-vxebwq`,
> not `main`. Clone, then `git checkout claude/ai-hedge-fund-platform-vxebwq`.

```bash
python -m pip install -e ".[dev]"

# Save your API keys (prompts you; input is hidden). Optional — everything
# below runs offline with --synthetic.
python -m axiom.cli setup

# Full offline demonstration: terminal + ICT engine + backtest, no API keys.
python -m axiom.cli demo

# Structural read of a symbol
python -m axiom.cli analyse --symbol ES --timeframe 15m --synthetic

# The operator terminal
python -m axiom.cli terminal --symbol ES --synthetic

# Backtest, costs always applied
python -m axiom.cli backtest --strategy silver-bullet --synthetic --days 120

# The five-agent pipeline (Research → Debate → Backtest → Risk → Review)
python -m axiom.cli pipeline --synthetic

# Causal HMM regime detection
python -m axiom.cli regime --synthetic --states 3

# Sweep many strategies walk-forward, ranked by deflated Sharpe
python -m axiom.cli research --synthetic --folds 4

# Current safety posture
python -m axiom.cli config
```

`--synthetic` runs against the deterministic generator, so everything works with no
network and no credentials. Every output produced this way is stamped
**SYNTHETIC** and refuses to present itself as performance.

For real data:

```bash
python -m pip install -e ".[yfinance]"
python -m axiom.cli analyse --symbol SPY --timeframe 1d --days 365
```

---

## Architecture

```
axiom.core        domain types, validated OHLCV series, session clock, provenance, settings
axiom.data        provider abstraction + adapters (CSV, yfinance, synthetic, OpenBB*)
axiom.ict         ── the proprietary alpha engine ──
                  swings · structure (BOS/CHoCH/MSS) · fair value gaps · order blocks
                  liquidity pools & sweeps · dealing ranges & OTE · SMT divergence
axiom.quant       Gaussian HMM and causal market-regime detection
axiom.research    strategy search: walk-forward, deflated Sharpe, PBO
axiom.strategy    signal generation — ICT setups, quant families, regime gating
axiom.risk        risk-first position sizing, hard limits, kill switch
axiom.execution   order model, simulated venue, routing choke point
axiom.portfolio   positions and P&L accounting
axiom.backtest    event-driven backtester and performance metrics
axiom.agents      Research → Debate → Backtest → Risk → Review pipeline
axiom.terminal    the operator terminal
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design decisions and their rationale,
and [`docs/ICT_METHODOLOGY.md`](docs/ICT_METHODOLOGY.md) for how each ICT concept is formalised.

\* OpenBB is AGPL-3.0 and deliberately quarantined. See [`docs/LICENSING.md`](docs/LICENSING.md).

---

## The three guarantees

### 1. No lookahead

Every ICT feature carries two positions: `origin_index` (where it sits on the chart) and
`confirmed_index` (the first bar at which it could be *known*). They differ constantly — a
swing high of strength 3 is only confirmed three bars later; an order block is only
identifiable once the displacement leg breaks structure.

`StrategyContext` filters exclusively on `confirmed_index`, so a strategy **cannot** read a
feature the market had not yet formed. The property is tested directly: analysing a prefix
of the tape must produce the same features as filtering a full-history analysis.

### 2. No invented numbers

Every `OHLCVSeries` carries a `Provenance` record. Synthetic and mock data are permanently
tainted — deriving, resampling, or cleaning them cannot launder them into evidence:

```python
Provenance.synthetic("gen").derive("resample").derive("clean").is_evidential
# False
```

Any performance report built from non-evidential data renders a warning banner and answers
`False` to `report.is_evidence`.

### 3. Searching many strategies does not manufacture an edge

Testing 500 candidates and keeping the best produces an impressive Sharpe *even
on pure noise* — with 500 trials the expected maximum is around 2.5–3.0 from
selection alone. `StrategyLab` therefore counts every candidate as a trial and
ranks on the **deflated Sharpe**, the probability the result beats the best of
that many coin flips. It also reports **Probability of Backtest Overfitting**,
which asks the separate question of whether the selection *procedure* finds
strategies that generalise at all.

A search that concludes *"nothing survived"* has done its job. That is the
expected outcome of most searches.

### 4. Risk budgets are upper bounds, not estimates

Sizing budgets for the entry gapping away from its intended price *and* for the
stop gapping through its trigger, and the venue expires entries that would fill
beyond tolerance. On the synthetic fixture this moved the worst trade from
**7.20× the per-trade budget to 0.57×**.

The two halves are not symmetric, and the docs say so: an entry that gaps can be
refused, because you are not in the position yet. A stop that gaps cannot be —
no order type prevents it. That residual is reported, not hidden.

### 5. No accidental execution

* The **kill switch** is checked first in both the risk manager and the router, and cannot be overridden by any argument, mode, or strategy.
* `OrderRouter` **refuses to be constructed** around a live venue unless the platform is in live mode with `AXIOM_LIVE_TRADING_CONFIRMED=I_UNDERSTAND_THE_RISK`.
* The agent pipeline has **no execute stage**. It terminates in a human approval request.

---

## Costs are never optional

The simulated venue is deliberately pessimistic:

| Behaviour | Choice | Why |
|---|---|---|
| Market order fill | Next bar's **open**, never the signal bar's close | Filling on the signal bar earns money that was never available |
| Slippage | Applied against the order, every time | Crossing the spread is not free |
| Commission | Per unit, both sides | |
| Limit orders | Must trade **through**, not merely touch | Touching does not guarantee queue priority |
| Stops that gap | Fill at the open, worse than the trigger | This is what actually happens |
| Same-bar stop *and* target | **Stop assumed first** | The alternative inflates results |

---

## Risk model

Sizing is risk-first: position size is derived from the distance to the stop, so cash at
risk is constant across setups. A strategy that cannot state its stop cannot be sized.

Enforced limits — every rejection is explained in plain language:

* max risk per trade, as % of equity
* daily loss limit (halts trading for the day)
* consecutive-loss circuit breaker (resets daily)
* max concurrent positions
* max gross exposure
* the kill switch

```
REJECTED qty=0 — a single ES unit carries $250,000 notional, which alone exceeds
the $200,000 gross exposure cap (200% of $100,000 equity). This instrument is
untradable at this account size — raise max_gross_exposure_pct or the account equity
```

---

## Development

```bash
make dev      # install with dev extras
make test     # pytest
make lint     # ruff
make type     # mypy --strict
make check    # all three
```

The test suite covers domain invariants, ICT geometry on handcrafted fixtures, the
anti-lookahead properties, risk limits, the fill model, and end-to-end backtest integrity.

---

## Status

Foundation complete and working end to end. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for what
is built, what is deliberately stubbed, and what comes next.

**No strategy in this repository has been validated on real market data.** The bundled ICT
strategies are reference implementations of the methodology, not proven edges.

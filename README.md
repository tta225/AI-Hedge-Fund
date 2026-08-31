# AXIOM

**A systematic trading platform built to tell you when it has not found anything.**

Most research codebases are optimised to produce a result. This one is optimised
to make a false result hard to produce and easy to detect. That is a different
objective, and it shows up in what the code refuses to do rather than in what it
computes.

> ⚠️ **Research and engineering infrastructure, not financial advice.** No
> bundled strategy is a proven edge. Nothing here has cleared the promotion gate.
> Paper trade first, and only risk capital you can afford to lose.

> 📉 **Six research campaigns. No survivors.** ICT structural features, sweep
> continuation, intraday seasonality, meta-labelling, cross-sectional equities,
> and low-turnover factors have each been measured and each failed to clear a
> deflated-Sharpe threshold. Those nulls are the project's main output so far,
> and each one is documented with its mechanism.

---

## What is actually here

Three layers, at different stages of maturity.

**A research harness that is hard to fool.** Causality is enforced by the type
system rather than by convention, provenance is permanent, every candidate
counts as a trial, and results are checked against a reference class of 61
published anomalies before anyone gets excited.

**A live desk that can be operated.** Idempotent order recording, broker-as-
source-of-truth reconciliation, persistent halts, crash-only restart, structured
logs with correlation IDs, health readable from the store alone, deduplicated
alerts with real delivery, pre-trade compliance, backups, a container, and a
promotion gate that has so far refused everything.

**Strategies.** These are reference implementations of methodologies, not edges.
The evidence says so.

---

## Quick start

```bash
python -m pip install -e ".[dev]"

# Everything below runs offline. Synthetic output is stamped SYNTHETIC and
# refuses to present itself as performance.
python -m axiom.cli demo
python -m axiom.cli terminal --symbol ES --synthetic
python -m axiom.cli research --synthetic --folds 4

make check      # ruff + mypy --strict + 1,296 tests
```

With Alpaca credentials (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` — the free
paper keys are enough for market data):

```bash
# A survivorship-free US equity universe: ~11,700 names including the ones
# that stopped trading. Takes about twenty minutes.
python -m scripts.cache_pit_universe --out data/pit

# What deleting the failures was worth
python -m scripts.survivorship_study --root data/pit

# The sixth campaign, on the honest universe
python -m scripts.low_turnover_study --pit-root data/pit
```

---

## The guarantees

### No lookahead

Every feature carries `origin_index` (where it sits) and `confirmed_index` (the
first bar it could be *known*). They differ constantly. `StrategyContext`
filters on the second, so a strategy cannot read a feature the market had not
yet formed. `Panel.history(i)` has an exclusive upper bound, and
`generate_checked` raises `LookaheadError` on a mis-stamped signal.

The property is tested by construction rather than asserted: scoring a bar on a
panel extended with later data must produce an identical signal.

### No invented numbers

Every series carries a `Provenance`. Synthetic and mock data are permanently
tainted — deriving, resampling or cleaning cannot launder them:

```python
Provenance.synthetic("gen").derive("resample").derive("clean").is_evidential
# False
```

A panel takes the *weakest* provenance of its constituents, because a
cross-sectional signal mixes every column into every output.

### Searching many strategies does not manufacture an edge

500 candidates on pure noise produce a best-of Sharpe around 2.5–3.0 from
selection alone. Every candidate counts as a trial, results are ranked on
deflated Sharpe, and PBO asks separately whether the selection *procedure*
generalises.

A search concluding *"nothing survived"* has done its job. It is the expected
outcome, and it has been the actual outcome six times.

### Results are checked against the outside view

61 published systematic strategies: max Sharpe 0.892, median 0.354, 15%
negative, **none above 1.0**. A candidate reporting 1.02 is flagged as needing
an explanation rather than celebrated. In the sixth campaign it got one — the
result inverted on a held-out universe.

### History contains the companies that failed

The universe is point-in-time. A name that had not yet listed and one that has
already died are both untradable, delisting forces a liquidation at the last
close, and the liquidity ranking that selects the universe uses only volume
observed *before* the selection date.

This matters more than it sounds. Silicon Valley Bank stops trading on
2023-03-09 and First Republic on 2023-04-28; a universe built from today's
liquid tickers contains neither, at any point in their history.

### Costs are a function of size

A flat basis-point charge prices a thousand-dollar trade and a hundred-million-
dollar trade identically, which hides the only question a desk must answer: how
much money can this take? The model is commission + half-spread + square-root
impact, so capacity curves are computable — and the honest finding was that it
is **harsher** than the flat 10bp it replaced.

### Nothing reaches a live venue by accident

The kill switch is checked first in both the risk manager and the router.
`OrderRouter` refuses construction around a live venue without
`AXIOM_LIVE_TRADING_CONFIRMED=I_UNDERSTAND_THE_RISK`. Live credentials never
fall back to an unscoped environment variable. The agent pipeline has no execute
stage. Research cannot promote straight to live.

---

## Architecture

```
axiom.core        domain types, validated series, provenance, session clock
axiom.data        providers, adapters, and point-in-time universes
axiom.ict         structural alpha engine — swings, BOS/CHoCH, FVGs, order
                  blocks, liquidity sweeps, dealing ranges, SMT divergence
axiom.alpha       cross-sectional agents, low-turnover factor families, ensemble
axiom.ml          triple-barrier labelling, purged K-fold, meta-labelling
axiom.quant       causal HMM regime detection
axiom.research    walk-forward, deflated Sharpe, PBO, plausibility priors,
                  turnover control, capacity curves
axiom.strategy    signal generation
axiom.risk        risk-first sizing, hard limits, kill switch
axiom.portfolio   positions, P&L, portfolio risk, multi-strategy allocation
axiom.execution   order model, venues, cost models, transaction cost analysis
axiom.store       the desk's memory — orders, fills, halts, backups
axiom.desk        the live loop, guards, compliance, registry, supervisor
axiom.ops         logs, metrics, health, alerts and their delivery, secrets
axiom.backtest    event-driven backtester
axiom.agents      Research → Debate → Backtest → Risk → Review
axiom.terminal    the operator terminal
```

Design decisions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Running it: [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

---

## The research record

Every campaign, and why it failed. These are the point of the project.

| study | finding |
|---|---|
| [`REAL_DATA_FINDINGS.md`](docs/REAL_DATA_FINDINGS.md) | ICT features show zero lift over a meaningless control band; sweeps lean the wrong way in all four datasets |
| [`SWEEP_CONTINUATION_STUDY.md`](docs/SWEEP_CONTINUATION_STUDY.md) | no edge; also uncovered a detector that emitted zero trades for a year |
| [`SEASONALITY_STUDY.md`](docs/SEASONALITY_STUDY.md) | the dispersion trap — a homogeneous candidate set *lowers* the deflated-Sharpe bar |
| [`CROSS_SECTIONAL_STUDY.md`](docs/CROSS_SECTIONAL_STUDY.md) | every candidate above 30× annual turnover lost money |
| [`TURNOVER_AND_CAPACITY.md`](docs/TURNOVER_AND_CAPACITY.md) | **the one thing that replicated**: hysteresis is worth ~0.45 Sharpe and ~5× capacity out of sample. It did not manufacture an edge |
| [`LOW_TURNOVER_STUDY.md`](docs/LOW_TURNOVER_STUDY.md) | slow factors are genuinely cheap to trade (0.8–2.5× turnover) and still produced nothing; the best discovery result inverted out of universe |

Two methodological lessons worth extracting:

**PBO cannot see universe overfitting.** It reshuffles folds within one
universe, so it detects overfitting to a *time window* and is blind to
overfitting to a *set of names*. In the sixth campaign PBO read 24%
("generalises acceptably") and the transfer failed completely.

**Bugs found by tests that pin mechanism, not output.** A residual-momentum
agent was ranking floating-point noise because OLS residuals with an intercept
sum to zero. A risk-parity solver oscillated with period two and silently
returned its equal-weight starting point. Both passed every contract test they
had, and both were caught by a test that asserted what the code was *for*.

---

## Status

The infrastructure is ready for capital. The research is not.

That asymmetry is deliberate and worth stating plainly: enterprise-grade
plumbing around a zero-expectancy signal is a reliable way to lose money on
schedule. The promotion gate exists precisely so that decision is not made by
whoever is feeling optimistic.

The bottleneck is now data, not machinery. Six campaigns is enough evidence for
that. Value and quality — the strongest low-turnover families in the literature
— cannot be tested at all without a fundamentals feed, and building a seventh
price-based factor search would be answering a question already asked.

See [`docs/OPERATIONS.md`](docs/OPERATIONS.md) for the current gap list.

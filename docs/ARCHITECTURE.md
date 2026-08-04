# Architecture

This document records the decisions that shaped AXIOM and, more importantly, *why* — so
that changing them later is a deliberate act rather than an accident.

## Layering

Dependencies point strictly downward. `core` knows nothing about strategies; `ict` knows
nothing about execution; `strategy` cannot reach a venue.

```
terminal ─┐
agents  ──┼──► backtest ──► strategy ──► ict ──► core
cli     ──┘         │           │                 ▲
                    ├──► risk ──┤                 │
                    ├──► execution                │
                    ├──► portfolio ───────────────┘
                    └──► data ────────────────────┘
```

The practical consequence: the ICT engine can be extracted and reused, the risk manager can
be tested without a market, and adding a broker touches exactly one package.

---

## The central problem: lookahead bias

Almost every ICT concept is defined retrospectively. An order block is "the last down candle
before the up move" — you cannot know which candle that was until the up move has happened.
A swing high is only a swing high once enough bars have passed without exceeding it.

A naive implementation computes these over the whole history and hands them to a strategy,
which then "discovers" an order block hours before the market could have. The resulting
equity curve is fiction.

### The solution: two indices on every feature

```python
@dataclass(slots=True)
class ICTFeature:
    origin_index: int      # where it sits on the chart
    confirmed_index: int   # first bar at which it could be KNOWN
```

| Feature | origin | confirmed | lag |
|---|---|---|---|
| Swing point (strength *n*) | the pivot bar | pivot + *n* | *n* bars |
| Fair value gap | middle bar of the three | third bar close | 1 bar |
| Structure event | the swing that broke | the bar that broke it | varies |
| Order block | the origin candle | the structure break | often 5–15 bars |
| Liquidity pool | the extreme | last member's confirmation | varies |

`StrategyContext` exposes only index-filtered accessors. There is no public path from a
strategy to an unfiltered feature list.

### Why the backtester can analyse once

The engine runs a single pass over the full series, and the backtester then filters per bar.
This is O(N) instead of O(N²) — and it is *safe* precisely because `confirmed_index` is
computed from the data available at that bar, not from where the feature happens to fall in
the array.

`test_ict.py::TestLookaheadDiscipline::test_analysing_a_prefix_matches_the_known_view`
verifies this directly: analysing bars `[0..k]` produces the same features as analysing
everything and filtering to `confirmed_index <= k`. If someone breaks the invariant, that
test fails.

### The subtle case: mitigation filters use `>=`, not `>`

```python
b.mitigated_index is None or b.mitigated_index >= index
```

An order block's mitigation index is the *first bar that touches it* — which is exactly the
bar a retracement entry wants to trade. Filtering on `> index` excludes every block on the
one bar it becomes actionable, silently making order-block entries unreachable. This was a
real bug, found by instrumenting the signal funnel and noticing a strategy path that could
never fire.

---

## The second problem: fabricated evidence

The failure mode is mundane. A data feed errors, something falls back to generated bars, a
backtest runs, and a number ends up in a document. Nobody lied; the provenance was simply
lost along the way.

### The solution: provenance that cannot be laundered

```python
Provenance.synthetic("gen").derive("resample").derive("clean").is_evidential  # False
```

`DataKind.REAL` derives to `DERIVED` (still evidential). `SYNTHETIC` and `MOCK` derive to
themselves, permanently. `merge()` takes the weakest of two kinds, so mixing real and
synthetic yields synthetic.

Consequences wired through the system:

* `PerformanceReport.is_evidence` is `False` for generated data, and `render()` leads with a
  warning banner.
* `require_real_data()` raises `SyntheticDataError` on any path that must not run on
  generated numbers.
* The terminal header renders a black-on-yellow banner.
* `DataRegistry` **never** falls back to synthetic implicitly — it must be registered
  explicitly, and the CLI requires `--synthetic` rather than degrading silently.

---

## Numeric policy

Prices are `float64` so the pipeline stays vectorisable. Prices are snapped to the
instrument's tick grid at every boundary where a price becomes an order or a fill, bounding
representation error far below one tick.

**Known limitation:** the cash ledger is also float64. For research and paper trading the
error is negligible relative to slippage assumptions. A production ledger handling real
money should move to fixed-point (`Decimal` or integer minor units). This is tracked in
`ROADMAP.md` and is a deliberate deferral, not an oversight.

---

## Time

Everything is tz-aware UTC internally. The ICT session clock converts **per timestamp** to
`America/New_York`.

This matters more than it sounds. US DST shifts every killzone by an hour in UTC terms. A
backtest spanning March or November with UTC-anchored windows silently mislabels a month of
bars. `test_killzones_survive_the_dst_transition` pins the behaviour: 08:30 New York is
13:30 UTC in winter and 12:30 UTC in summer, and both must land inside the NY AM killzone.

Trading days are anchored to New York midnight — ICT's "true day open", and the natural
boundary for daily loss accounting.

---

## Execution: pessimism by construction

Every choice in the fill model is the unfavourable one. See the table in the README.

The reasoning: a backtest's job is to *fail* a bad strategy cheaply. Optimistic fills invert
that — they let bad strategies survive to the stage where real money finds the truth. The
same-bar stop-versus-target ambiguity is resolved in favour of the stop for this reason;
it is occasionally wrong, and wrong in the safe direction.

`OrderRouter` is the single choke point. Nothing submits to a venue directly. It refuses at
construction time to wrap a live venue outside live mode, so an unsafe configuration fails
at startup rather than at the first order.

---

## Risk: fail closed, and explain

Any check that cannot be evaluated blocks the trade. Rejections carry plain-language
reasons, and the reasons distinguish cases that need different fixes — "the book is full"
versus "one unit of this instrument can never fit inside the cap" look identical to a naive
implementation and require opposite responses.

The consecutive-loss counter resets on the day roll. Without that reset, a strategy taking
its first four losses in a row is disabled for the remainder of the backtest, which
truncates the sample and reports a handful of trades as though they were the whole run.
That was also a real bug, and the signal funnel is what exposed it.

### The signal funnel

`BacktestResult.funnel()` reports signal → order → trade attrition with reasons:

```
signals generated : 26
orders routed     : 12 (incl. bracket legs)
trades completed  : 4
risk rejections:
    22 × 4 consecutive losses ≥ limit of 4
```

A strategy that "takes no trades" is far more often being declined by risk than failing to
generate signals. Conflating the two costs hours.

---

## The agent layer

The five roles from the source workflow — Research, Debate, Backtest, Risk, Review — are
implemented with one structural constraint: **agents never produce numbers.**

`AgentReport` separates `facts` (computed by the deterministic engines) from `narrative`
(generated prose). They are never merged, so a reader always knows which is which. Without
`ANTHROPIC_API_KEY` the pipeline still runs and every stage returns complete measured
findings; only the prose is absent.

The pipeline has no execute stage. It terminates in a human approval request.

---

## Extending

**A new data provider:** subclass `BaseProvider`, implement `_fetch_raw` and `_provenance`,
register it. Nothing downstream changes.

**A new strategy:** subclass `Strategy`, implement `evaluate(context) -> Signal | None`. Use
only `StrategyContext` accessors — reaching into `context.ict` directly is the one thing
strategies must not do.

**A new ICT primitive:** add a dataclass to `ict/models.py` inheriting `ICTFeature` (so it
carries both indices), a detector module, and wire it into `ICTEngine.analyse` in dependency
order. Add it to `ICTState.known()`.

**A live venue:** subclass `ExecutionVenue` with `is_live = True`. The router will refuse to
wrap it outside live mode — that is the safety property, not an obstacle to work around.

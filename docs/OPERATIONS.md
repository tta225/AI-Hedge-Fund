# Running the desk

What exists, how to operate it, and what is still missing. Written for the
person on call at 3am, so it leads with the procedures rather than the design.

---

## Credentials

Secrets resolve through `axiom.ops.secrets`, in this order: mounted files, then
a vault command, then the process environment.

```bash
# Kubernetes/systemd style: one file per secret, owner-readable only.
export AXIOM_SECRETS_DIR=/run/secrets
chmod 600 /run/secrets/*

# Or a vault, without a vault dependency — any command that prints the secret.
export AXIOM_SECRETS_COMMAND='vault read -field=value secret/axiom/{key}'
export AXIOM_SECRETS_COMMAND='op read op://desk/{key}/credential'
```

Names are scoped by environment: `AXIOM_PAPER_APCA_API_KEY_ID`,
`AXIOM_LIVE_APCA_API_KEY_ID`.

**The one rule worth memorising: live never falls back to an unscoped name.**
Paper will happily read a plain `APCA_API_KEY_ID`, so an existing setup keeps
working. Live refuses to, and the error says so. That is deliberate — the
alternative is that leaving a paper variable set arms real money the moment
someone flips an environment flag.

Credentials never render themselves. `str`, `repr`, and f-string formatting all
produce `<NAME paper:a1b2c3d4>`. Call `.reveal()` at the point of use, nowhere
else. `CredentialSet.audit()` is safe to log.

A rotated secret is picked up by `CredentialSet.clear()`; resolution is cached
so a vault is not called on every order.

## Backups

```bash
python -c "
from axiom.store.db import Store
from axiom.store.backup import backup_store, prune_backups
with Store('data/desk.db') as store:
    print(backup_store(store, 'data/backups').render())
print('pruned:', prune_backups('data/backups', keep=14))
"
```

Run it on a timer. It uses SQLite's online backup API, so it is safe against a
running desk — a plain `cp` is not, because WAL mode keeps recent commits in the
`-wal` sidecar and a file copy silently omits exactly the writes an incident
would be about.

### Restore

```bash
python -c "
from axiom.store.backup import restore_store
print(restore_store('data/backups/desk-20260828T031500Z.db', 'data/desk.db', overwrite=True))
"
```

Restore verifies before it writes (integrity check plus schema version), moves
the displaced database aside as `desk.superseded-<stamp>` rather than deleting
it, and proves the result opens under `Store` before returning. `overwrite=True`
is required and is not a formality — restoring the wrong way round is the
failure mode of every restore procedure.

**After any restore, reconcile against the broker before trading.** The store is
now as of the backup's timestamp; the broker is as of now. `DeskRunner.resume()`
does this and will halt on a discrepancy, which is the correct outcome.

## Health

Readable from the store alone, so a watchdog needs no access to the running
process — the one property that matters when the process is wedged, because a
hung desk answers "fine" or does not answer at all.

```bash
axiom desk-health --db data/desk.db   # exits 0 ok, 1 degraded, 2 down
axiom desk-health --alert             # and route anything failing to the alert sinks
```

The container's `HEALTHCHECK` runs exactly this.

## Metrics

`axiom.ops.metrics.REGISTRY.render()` emits Prometheus text format. In-process,
no background thread, no socket — a metrics client that opens a connection is
one more thing that can hang inside the tick loop it is measuring. Expose it
however suits: an HTTP handler, a file the node exporter reads, a log line.

Instruments already wired into `DeskRunner`:

| metric | type | meaning |
|---|---|---|
| `axiom_orders_sent_total` | counter | accepted by a venue |
| `axiom_orders_rejected_total` | counter | venue refused |
| `axiom_signals_total` | counter | strategies produced |
| `axiom_signals_blocked_total` | counter | a guard, risk check or compliance rule stopped it |
| `axiom_halts_total` | counter | desk halted itself |
| `axiom_equity` | gauge | last marked equity |
| `axiom_open_positions` | gauge | instruments held |
| `axiom_gross_exposure` | gauge | sum of absolute notional |
| `axiom_halted` | gauge | 1 when trading is stopped |
| `axiom_tick_seconds` | histogram | wall time of one tick |

Alert on `axiom_halted == 1` and on the p99 of `axiom_tick_seconds`. A tick loop
averaging 200ms with a p99 of 30s misses one bar in a hundred, and the mean says
it is healthy — which is why the histogram exists and an average does not.

## Alerting

`AlertRouter` deduplicates by key with a cooldown and pairs every condition
with a resolution. Sinks are in `axiom.ops.sinks` and are wired from whatever
credentials the environment has:

```bash
export AXIOM_PAPER_SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'
export AXIOM_LIVE_PAGERDUTY_ROUTING_KEY='...'
export AXIOM_PAPER_ALERT_WEBHOOK_URL='https://internal.example/incidents'

axiom desk-health --alert --environment paper
```

The command prints which sinks actually got wired, so a typo in a variable name
shows up as a missing sink rather than as silence.

| sink | when |
|---|---|
| `console_sink` | always — the only one that works when the network is what broke |
| `FileSink` | opt-in via `log_path`; local JSON-lines record, no network |
| `SlackSink` | routine warnings and criticals |
| `PagerDutySink` | criticals only by default; triggers and auto-resolves by dedup key |
| `WebhookSink` | anything else — posts the alert's own fields, not a vendor schema |

Three properties worth knowing:

**Webhook URLs are credentials.** Anyone holding one can post as the desk, so
they resolve through `axiom.ops.secrets` and are environment-scoped. A paper
desk cannot page whoever carries the live one, and a URL never reaches a log
line even on a delivery failure.

**Resolutions bypass the severity threshold.** A PagerDuty incident opened by a
critical alert carries an INFO-severity resolution; if that were filtered, the
incident would never close.

**No retry, no queue.** A sink that hangs inside a desk tick turns a
notification problem into an execution problem, so every sink has a 5s timeout
and swallows its own failures. A missed alert is recovered by the health check,
which reads the store and does not depend on any of this having worked.

## Compliance

Pre-trade rules run in `DeskRunner._send`, **before** the order is recorded. A
blocked order is one that was never permitted to exist; writing it to the order
log first would leave a row reconciliation has to explain away forever.

```python
from axiom.desk.compliance import default_engine

runner = DeskRunner(
    ...,
    compliance=default_engine(
        restricted=frozenset({"GME"}),
        max_position_fraction=0.10,
        max_gross_multiple=1.5,
        max_daily_fraction=0.25,
        long_only=False,
    ),
)
```

The default is an **empty** engine — no rules, everything permitted. That is a
decision, and it is left to the caller to make deliberately rather than having
limits appear that are not in their configuration. `engine.describe()` says
which state you are in.

Every rule runs on every order; breaches are not short-circuited. An order that
breaks three rules reports three, because fixing the first and resubmitting into
the second is how a bad order gets sent on the third attempt.

Note `RequirePrice`, which is first in the default set: every numeric rule
abstains when it cannot value the symbol, so something has to refuse the
unpriceable order outright or it passes every check by default.

## Strategy registry and promotion

```python
from axiom.desk.registry import Evidence, Stage, StrategyRegistry, StrategyVersion

registry = StrategyRegistry(store)
version = registry.register(StrategyVersion("mom12_1", {"lookback": 252, "skip": 21}))
registry.attach_evidence(version.version_id, Evidence(
    sharpe=0.62, deflated_sharpe=0.24, n_trials=12,
    n_observations=727, annual_turnover=2.9, is_evidential=True,
    notes="low-turnover campaign, discovery universe",
))
registry.promote(version.version_id, Stage.PAPER, authorised_by="alice")
```

A version id is a hash of name plus parameters, so a parameter changed and not
recorded is impossible — the id moves whether anyone meant it to or not.

`deflated_sharpe` is the **probability** the research harness reports, on
`[0, 1]`, not a Sharpe ratio. The gate requires 0.95, the same bar
`PanelLab` uses; a test asserts the two stay equal.

The promotion above **will be refused** — 0.24 is below 0.95 — and that is the
correct behaviour. Six campaigns have produced nothing that clears it. `force=True`
exists for the legitimate case of running at nominal size to gather live
evidence; it does not suppress the check, it records that the check failed, who
overrode it, and which conditions were failing.

Research cannot jump straight to live. Paper is the only stage where an
execution bug is free.

## Transaction cost analysis

The only loop that runs from live trading back into research. The impact
coefficient η is currently 1.0 — a deliberately conservative guess, since
published estimates cluster at 0.3–0.6 — and the desk has been recording fills
that can test it.

```python
from axiom.execution.tca import analyse, executions_from_store

loaded = executions_from_store(store)
print(f"coverage {loaded.coverage:.0%} of {loaded.n_rows} fills")
print(analyse(loaded.executions).render())
```

Read the coverage number before the cost number. Fills with no linked decision
price — bracket legs, manual trades, anything recorded before schema v3 — are
dropped rather than defaulted, because substituting the fill's own price reports
zero cost for exactly the executions nobody was supervising.

`model_is_optimistic` is the field that matters: a model that overstates cost
produces conservative research, one that understates it produces backtests that
cannot be traded.

Recalibrate η only once `calibration_is_meaningful` is true (30+ fills).

## Container

```bash
docker build -t axiom:latest .
docker run --rm -v "$PWD/data:/app/data" \
  -e AXIOM_SECRETS_DIR=/run/secrets \
  -v /run/secrets:/run/secrets:ro \
  axiom:latest desk-health --db /app/data/desk.db
```

Two stages, non-root (uid 10001), no build toolchain in the runtime image.
`/app/data` must be a mounted volume — the store lives there, and left inside
the image it is lost on every redeploy, which means losing the record of what
the desk owns.

## Schema migrations

`SCHEMA_VERSION = 3`. Migration is forward-only and automatic on open; a
database *newer* than the running code is refused rather than guessed at.

v2 → v3 adds `reference_price`, `decision_adv`, `decision_volatility`,
`asset_class` and `point_value` to `orders`. All nullable: an existing order
reads as "we did not write this down", which is a different claim from a zero
that would read as a free trade.

**Back up before upgrading a build that changes the schema.** The migration is
in-place and there is no down migration.

---

## What is still missing

Stated here rather than in a design document, because the gap matters most to
whoever is operating this.

- **The backup/restore path is tested but has never been exercised under
  incident conditions.** Schedule a drill.
- **No fundamentals feed**, so value and quality — the strongest low-turnover
  factor families — cannot be tested at all.
- **Market orders only.** No TWAP/VWAP/participation algos, no smart routing.
- **Single-book risk.** `PortfolioRiskManager` runs one book; there is no
  capital allocation or risk aggregation across strategies.
- **No regulatory trade record** beyond the store's own append-only log, and no
  retention policy.
- **Delisting is modelled as an exit at the last close.** A bankruptcy that
  stops trading before the equity is wiped out is still flattered. Much smaller
  than the survivorship bias now removed, and it points the same way.

And the one that governs all the others: **no strategy has cleared the promotion
gate.** The infrastructure described above is ready for capital. Nothing in the
research is.

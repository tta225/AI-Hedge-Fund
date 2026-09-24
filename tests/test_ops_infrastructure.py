"""Secrets, metrics, backup, registry, compliance, TCA.

The infrastructure layer. Each of these modules exists to make a specific
failure impossible, so the tests are mostly about the refusals: a live
credential that will not fall back to a paper variable, a backup that will not
restore over a live store, a promotion that will not clear a gate. A test that
only proves the happy path leaves exactly the property that matters unverified.
"""

from __future__ import annotations

import os
import sqlite3
import stat

import numpy as np
import pandas as pd
import pytest

from axiom.core.types import AssetClass, Instrument, Side
from axiom.desk.compliance import (
    ComplianceEngine,
    Context,
    LongOnly,
    MaxDailyNotional,
    MaxGrossExposure,
    MaxPositionNotional,
    RequirePrice,
    RestrictedList,
    default_engine,
)
from axiom.desk.registry import (
    Evidence,
    PromotionError,
    PromotionGate,
    Stage,
    StrategyRegistry,
    StrategyVersion,
)
from axiom.execution.base import Fill, Order
from axiom.execution.tca import Execution, analyse, executions_from_store
from axiom.ops.metrics import Counter, Gauge, Histogram, Registry
from axiom.ops.secrets import (
    CommandSource,
    Credential,
    CredentialSet,
    Environment,
    EnvSource,
    FileSource,
    SecretsError,
    default_credentials,
)
from axiom.store.backup import (
    BackupError,
    backup_store,
    list_backups,
    prune_backups,
    restore_store,
    verify_backup,
)
from axiom.store.db import Store

EQUITY = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)


def _order(symbol: str = "AAPL", side: Side = Side.BUY, quantity: float = 10.0) -> Order:
    return Order(
        instrument=Instrument(symbol=symbol, asset_class=AssetClass.EQUITY),
        side=side,
        quantity=quantity,
        reference_price=100.0,
    )


# --- secrets ---------------------------------------------------------------


class TestCredential:
    def test_never_renders_its_value(self) -> None:
        credential = Credential("KEY", "supersecret", "env:KEY", Environment.PAPER)
        for rendered in (str(credential), repr(credential), f"{credential}", f"{credential!r}"):
            assert "supersecret" not in rendered
        assert "supersecret" not in str([credential])
        assert credential.reveal() == "supersecret"

    def test_fingerprint_is_stable_and_salted_by_name(self) -> None:
        first = Credential("A", "same", "env:A", Environment.PAPER)
        again = Credential("A", "same", "env:A", Environment.PAPER)
        other = Credential("B", "same", "env:B", Environment.PAPER)
        assert first.fingerprint == again.fingerprint
        # The same secret under two names must not correlate across logs.
        assert first.fingerprint != other.fingerprint

    def test_empty_value_is_refused(self) -> None:
        with pytest.raises(SecretsError, match="empty"):
            Credential("KEY", "", "env:KEY", Environment.PAPER)

    def test_matches_is_constant_time(self) -> None:
        credential = Credential("KEY", "value", "env", Environment.PAPER)
        assert credential.matches("value")
        assert not credential.matches("valuf")


class TestCredentialSet:
    def test_scoped_name_wins_over_unscoped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AXIOM_PAPER_APCA_KEY", "scoped")
        monkeypatch.setenv("APCA_KEY", "unscoped")
        credentials = CredentialSet(Environment.PAPER, (EnvSource(),))
        assert credentials.get("APCA_KEY").reveal() == "scoped"

    def test_paper_falls_back_to_unscoped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AXIOM_PAPER_APCA_KEY", raising=False)
        monkeypatch.setenv("APCA_KEY", "legacy")
        credentials = CredentialSet(Environment.PAPER, (EnvSource(),))
        assert credentials.get("APCA_KEY").reveal() == "legacy"

    def test_live_refuses_the_unscoped_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The control this module exists for.

        A paper variable left set in the environment must never be able to arm
        real money just because the environment was switched to live.
        """
        monkeypatch.delenv("AXIOM_LIVE_APCA_KEY", raising=False)
        monkeypatch.setenv("APCA_KEY", "paper-key-still-set")
        credentials = CredentialSet(Environment.LIVE, (EnvSource(),))
        with pytest.raises(SecretsError, match="deliberately not consulted"):
            credentials.get("APCA_KEY")

    def test_live_accepts_its_own_scoped_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AXIOM_LIVE_APCA_KEY", "deliberate")
        credentials = CredentialSet(Environment.LIVE, (EnvSource(),))
        assert credentials.get("APCA_KEY").reveal() == "deliberate"

    def test_caches_so_a_vault_is_not_hammered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AXIOM_PAPER_K", "first")
        credentials = CredentialSet(Environment.PAPER, (EnvSource(),))
        assert credentials.get("K").reveal() == "first"
        monkeypatch.setenv("AXIOM_PAPER_K", "rotated")
        assert credentials.get("K").reveal() == "first"
        credentials.clear()
        assert credentials.get("K").reveal() == "rotated"

    def test_audit_never_contains_a_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AXIOM_PAPER_K", "topsecret")
        credentials = CredentialSet(Environment.PAPER, (EnvSource(),))
        credentials.get("K")
        rendered = str(credentials.audit())
        assert "topsecret" not in rendered
        assert "env:AXIOM_PAPER_K" in rendered

    def test_error_names_what_was_tried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AXIOM_PAPER_MISSING", raising=False)
        monkeypatch.delenv("MISSING", raising=False)
        credentials = CredentialSet(Environment.PAPER, (EnvSource(),))
        with pytest.raises(SecretsError, match="AXIOM_PAPER_MISSING"):
            credentials.get("MISSING")

    def test_get_optional_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AXIOM_PAPER_NOPE", raising=False)
        monkeypatch.delenv("NOPE", raising=False)
        credentials = CredentialSet(Environment.PAPER, (EnvSource(),))
        assert credentials.get_optional("NOPE") is None


class TestFileSource:
    def test_reads_a_mounted_secret(self, tmp_path) -> None:
        path = tmp_path / "AXIOM_PAPER_K"
        path.write_text("from-a-file\n")
        path.chmod(0o600)
        assert FileSource(tmp_path).fetch("AXIOM_PAPER_K") == "from-a-file"

    def test_refuses_a_world_readable_secret(self, tmp_path) -> None:
        path = tmp_path / "K"
        path.write_text("exposed")
        path.chmod(0o644)
        with pytest.raises(SecretsError, match="readable beyond its owner"):
            FileSource(tmp_path).fetch("K")

    def test_permissive_mode_can_be_opted_out_of(self, tmp_path) -> None:
        path = tmp_path / "K"
        path.write_text("exposed")
        path.chmod(0o644)
        assert FileSource(tmp_path, require_owner_only=False).fetch("K") == "exposed"

    def test_absent_file_is_a_miss_not_an_error(self, tmp_path) -> None:
        assert FileSource(tmp_path).fetch("ABSENT") is None


class TestCommandSource:
    def test_runs_a_command(self) -> None:
        source = CommandSource("printf %s from-a-command")
        assert source.fetch("K") == "from-a-command"

    def test_key_is_not_interpolated_into_a_shell(self) -> None:
        """A key with shell metacharacters must not become a second command."""
        source = CommandSource("printf %s {key}")
        with pytest.raises(SecretsError, match="not safe to interpolate"):
            source.fetch("K; rm -rf /")

    def test_missing_binary_is_reported_clearly(self) -> None:
        source = CommandSource("axiom-no-such-binary {key}")
        with pytest.raises(SecretsError, match="not found"):
            source.fetch("K")

    def test_failure_is_surfaced(self) -> None:
        source = CommandSource("false")
        with pytest.raises(SecretsError, match="failed"):
            source.fetch("K")


def test_default_credentials_orders_files_before_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "AXIOM_PAPER_K"
    path.write_text("from-mount")
    path.chmod(0o600)
    monkeypatch.setenv("AXIOM_SECRETS_DIR", str(tmp_path))
    monkeypatch.setenv("AXIOM_PAPER_K", "from-env")
    monkeypatch.delenv("AXIOM_SECRETS_COMMAND", raising=False)
    assert default_credentials().get("K").reveal() == "from-mount"


# --- metrics ---------------------------------------------------------------


class TestMetrics:
    def test_counter_refuses_to_decrease(self) -> None:
        counter = Counter("c")
        counter.inc(2)
        with pytest.raises(ValueError, match="cannot decrease"):
            counter.inc(-1)
        assert counter.value() == 2

    def test_labels_are_independent_series(self) -> None:
        counter = Counter("c")
        counter.inc(1, venue="alpaca")
        counter.inc(5, venue="sim")
        assert counter.value(venue="alpaca") == 1
        assert counter.value(venue="sim") == 5
        assert counter.value() == 0

    def test_gauge_refuses_non_finite(self) -> None:
        gauge = Gauge("g")
        with pytest.raises(ValueError, match="non-finite"):
            gauge.set(float("nan"))

    def test_histogram_buckets_are_cumulative(self) -> None:
        histogram = Histogram("h", buckets=(1.0, 10.0))
        for value in (0.5, 5.0, 50.0):
            histogram.observe(value)
        rendered = histogram.render()
        assert 'h_bucket{le="1"} 1' in rendered
        assert 'h_bucket{le="10"} 2' in rendered
        assert 'h_bucket{le="+Inf"} 3' in rendered
        assert histogram.count() == 3

    def test_quantile_reports_a_bucket_edge(self) -> None:
        histogram = Histogram("h", buckets=(1.0, 10.0, 100.0))
        for _ in range(99):
            histogram.observe(0.5)
        histogram.observe(50.0)
        assert histogram.quantile(0.5) == 1.0
        assert histogram.quantile(1.0) == 100.0

    def test_quantile_of_nothing_is_nan(self) -> None:
        assert np.isnan(Histogram("h").quantile(0.5))

    def test_timing_records_even_when_the_block_raises(self) -> None:
        """The slow calls are the ones that time out; timing only successes
        keeps a latency panel green through an outage."""
        histogram = Histogram("h")
        with pytest.raises(RuntimeError), histogram.time():
            raise RuntimeError("boom")
        assert histogram.count() == 1

    def test_ascending_buckets_are_required(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            Histogram("h", buckets=(10.0, 1.0))

    def test_registry_is_idempotent_by_name(self) -> None:
        registry = Registry()
        assert registry.counter("c") is registry.counter("c")

    def test_registry_refuses_a_type_change(self) -> None:
        registry = Registry()
        registry.counter("x")
        with pytest.raises(ValueError, match="already registered"):
            registry.gauge("x")

    def test_label_values_are_escaped(self) -> None:
        registry = Registry()
        registry.counter("c").inc(1, detail='say "hi"')
        assert r"\"hi\"" in registry.render()

    def test_render_is_prometheus_shaped(self) -> None:
        registry = Registry()
        registry.gauge("g", "a level").set(1.5)
        rendered = registry.render()
        assert "# HELP g a level" in rendered
        assert "# TYPE g gauge" in rendered
        assert rendered.endswith("\n")


# --- backup ----------------------------------------------------------------


@pytest.fixture
def populated_store(tmp_path) -> Store:
    store = Store(tmp_path / "desk.db")
    order = _order()
    order_id, _ = store.record_order(
        order, "key-1", pd.Timestamp("2026-01-01"), decision_adv=1e8, decision_volatility=0.02
    )
    store.record_fill(
        order_id,
        Fill(
            order_id=order_id,
            instrument=order.instrument,
            side=Side.BUY,
            quantity=10.0,
            price=100.5,
            commission=0.02,
            timestamp=pd.Timestamp("2026-01-01T10:00:00"),
            strategy="test",
        ),
        venue_fill_id="vf-1",
    )
    store.record_equity(pd.Timestamp("2026-01-01"), 100_000.0, 50_000.0)
    return store


class TestBackup:
    def test_snapshot_captures_uncheckpointed_writes(self, populated_store: Store) -> None:
        """The reason for the online backup API rather than a file copy."""
        result = backup_store(populated_store, populated_store.path + "-backups")
        assert result.verified
        assert result.row_counts["fills"] == 1
        assert result.row_counts["orders"] == 1

    def test_verify_rejects_a_corrupt_file(self, tmp_path) -> None:
        broken = tmp_path / "broken.db"
        broken.write_bytes(b"this is not a database")
        assert verify_backup(broken) is False

    def test_verify_rejects_a_stale_schema(self, tmp_path) -> None:
        stale = tmp_path / "stale.db"
        connection = sqlite3.connect(stale)
        connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        connection.commit()
        connection.close()
        assert verify_backup(stale) is False

    def test_restore_refuses_to_overwrite_by_default(self, populated_store: Store, tmp_path) -> None:
        result = backup_store(populated_store, tmp_path / "backups")
        with pytest.raises(BackupError, match="already exists"):
            restore_store(result.path, populated_store.path)

    def test_restore_sets_the_displaced_file_aside(self, populated_store: Store, tmp_path) -> None:
        result = backup_store(populated_store, tmp_path / "backups")
        populated_store.close()
        restored = restore_store(result.path, populated_store.path, overwrite=True)
        superseded = list(restored.parent.glob("desk.superseded-*"))
        assert superseded, "the replaced database must survive a wrong-way restore"
        with Store(restored) as store:
            assert len(store.fills_for()) == 1

    def test_restore_refuses_an_unverified_backup(self, tmp_path) -> None:
        broken = tmp_path / "broken.db"
        broken.write_bytes(b"nope")
        with pytest.raises(BackupError, match="failed verification"):
            restore_store(broken, tmp_path / "target.db")

    def test_prune_keeps_the_newest(self, populated_store: Store, tmp_path) -> None:
        directory = tmp_path / "backups"
        stamps = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(5)]
        for stamp in stamps:
            backup_store(populated_store, directory, at=stamp, verify=False)
        removed = prune_backups(directory, keep=2)
        assert len(removed) == 3
        assert len(list_backups(directory)) == 2

    def test_prune_refuses_to_delete_everything(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            prune_backups(tmp_path, keep=0)

    def test_two_backups_in_one_second_do_not_collide_silently(
        self, populated_store: Store, tmp_path
    ) -> None:
        moment = pd.Timestamp("2026-01-01T00:00:00")
        backup_store(populated_store, tmp_path / "b", at=moment, verify=False)
        with pytest.raises(BackupError, match="already exists"):
            backup_store(populated_store, tmp_path / "b", at=moment, verify=False)


def test_v2_database_migrates_to_v3(tmp_path) -> None:
    """An existing desk must open and carry its history, not be refused.

    Built from raw v2 DDL rather than by stripping columns off a v3 file: the
    point is that a database written by the *previous build* still opens, and
    reconstructing one by mutation would test the mutation.
    """
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE orders (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT    NOT NULL UNIQUE,
            venue_order_id  TEXT,
            symbol          TEXT    NOT NULL,
            side            TEXT    NOT NULL,
            quantity        REAL    NOT NULL,
            order_type      TEXT    NOT NULL,
            limit_price     REAL,
            stop_price      REAL,
            stop_loss       REAL,
            take_profit     REAL,
            strategy        TEXT    NOT NULL DEFAULT '',
            tag             TEXT    NOT NULL DEFAULT '',
            created_us      INTEGER NOT NULL
        );
        CREATE TABLE order_events (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL REFERENCES orders (id),
            status   TEXT    NOT NULL,
            detail   TEXT    NOT NULL DEFAULT '',
            at_us    INTEGER NOT NULL
        );
        CREATE TABLE fills (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id      INTEGER REFERENCES orders (id),
            venue_fill_id TEXT    UNIQUE,
            symbol        TEXT    NOT NULL,
            side          TEXT    NOT NULL,
            quantity      REAL    NOT NULL,
            price         REAL    NOT NULL,
            commission    REAL    NOT NULL DEFAULT 0.0,
            slippage      REAL    NOT NULL DEFAULT 0.0,
            strategy      TEXT    NOT NULL DEFAULT '',
            at_us         INTEGER NOT NULL
        );
        CREATE TABLE equity (
            at_us    INTEGER PRIMARY KEY,
            equity   REAL NOT NULL,
            cash     REAL NOT NULL,
            exposure REAL NOT NULL DEFAULT 0.0
        );
        CREATE TABLE halts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            reason     TEXT    NOT NULL,
            detail     TEXT    NOT NULL DEFAULT '',
            at_us      INTEGER NOT NULL,
            cleared_us INTEGER
        );
        INSERT INTO meta VALUES ('schema_version', '2');
        INSERT INTO orders (idempotency_key, symbol, side, quantity, order_type, created_us)
            VALUES ('legacy-key', 'AAPL', 'buy', 10.0, 'market', 1767225600000000);
        """
    )
    connection.commit()
    connection.close()

    with Store(path) as migrated:
        assert migrated.get_meta("schema_version", "") == "3"
        # The pre-existing order survives, with the new columns null rather
        # than zero — "not written down" is not the same claim as "free".
        row = migrated._connection.execute(
            "SELECT reference_price, point_value FROM orders WHERE idempotency_key='legacy-key'"
        ).fetchone()
        assert row["reference_price"] is None
        assert row["point_value"] is None
        order_id, was_new = migrated.record_order(_order(), "k", pd.Timestamp("2026-01-01"))
        assert was_new and order_id > 0


def test_gate_threshold_matches_the_research_harness() -> None:
    """The gate must not be looser than the bar the research was held to."""
    from axiom.desk.registry import SKILL_THRESHOLD as GATE_THRESHOLD
    from axiom.research.panel_lab import SKILL_THRESHOLD as LAB_THRESHOLD

    assert GATE_THRESHOLD == LAB_THRESHOLD
    assert PromotionGate().min_deflated_sharpe == LAB_THRESHOLD


# --- strategy registry -----------------------------------------------------


def _evidence(**overrides: object) -> Evidence:
    base: dict[str, object] = {
        "sharpe": 0.6,
        "deflated_sharpe": 0.99,
        "n_trials": 12,
        "n_observations": 500,
        "annual_turnover": 4.0,
        "is_evidential": True,
    }
    base.update(overrides)
    return Evidence(**base)  # type: ignore[arg-type]


@pytest.fixture
def registry(tmp_path) -> StrategyRegistry:
    return StrategyRegistry(Store(tmp_path / "desk.db"))


class TestStrategyRegistry:
    def test_version_id_is_derived_from_content(self) -> None:
        first = StrategyVersion("mom", {"lookback": 252, "skip": 21})
        # Same parameters, different insertion order: the same version.
        same = StrategyVersion("mom", {"skip": 21, "lookback": 252})
        different = StrategyVersion("mom", {"lookback": 126, "skip": 21})
        assert first.version_id == same.version_id
        assert first.version_id != different.version_id

    def test_registration_is_idempotent(self, registry: StrategyRegistry) -> None:
        version = StrategyVersion("mom", {"lookback": 252})
        registry.register(version)
        registry.register(StrategyVersion("mom", {"lookback": 252}))
        assert len(registry.all()) == 1

    def test_re_registering_does_not_demote_a_live_version(
        self, registry: StrategyRegistry
    ) -> None:
        version = registry.register(StrategyVersion("mom", {"lookback": 252}))
        registry.attach_evidence(version.version_id, _evidence())
        registry.promote(version.version_id, Stage.PAPER, authorised_by="ops")
        registry.promote(version.version_id, Stage.LIVE, authorised_by="ops")
        again = registry.register(StrategyVersion("mom", {"lookback": 252}))
        assert again.stage is Stage.LIVE

    def test_promotion_without_evidence_is_refused(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("mom", {"lookback": 252}))
        with pytest.raises(PromotionError, match="no recorded evidence"):
            registry.promote(version.version_id, Stage.PAPER, authorised_by="ops")

    def test_gate_refuses_a_deflated_sharpe_below_the_bar(
        self, registry: StrategyRegistry
    ) -> None:
        """The behaviour that should fire most often: five campaigns, no survivors."""
        version = registry.register(StrategyVersion("mom", {"lookback": 252}))
        registry.attach_evidence(version.version_id, _evidence(deflated_sharpe=0.5))
        with pytest.raises(PromotionError, match="deflated Sharpe"):
            registry.promote(version.version_id, Stage.PAPER, authorised_by="ops")

    def test_gate_refuses_high_turnover(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("fast", {"lookback": 5}))
        registry.attach_evidence(version.version_id, _evidence(annual_turnover=45.0))
        with pytest.raises(PromotionError, match="turnover"):
            registry.promote(version.version_id, Stage.PAPER, authorised_by="ops")

    def test_gate_refuses_synthetic_evidence(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("mom", {"lookback": 252}))
        registry.attach_evidence(version.version_id, _evidence(is_evidential=False))
        with pytest.raises(PromotionError, match="synthetic"):
            registry.promote(version.version_id, Stage.PAPER, authorised_by="ops")

    def test_all_failing_conditions_are_reported(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("bad", {}))
        registry.attach_evidence(
            version.version_id,
            _evidence(deflated_sharpe=0.0, sharpe=0.01, annual_turnover=99.0, n_observations=10),
        )
        with pytest.raises(PromotionError) as excinfo:
            registry.promote(version.version_id, Stage.PAPER, authorised_by="ops")
        message = str(excinfo.value)
        for expected in ("deflated Sharpe", "out-of-sample Sharpe", "turnover", "bars"):
            assert expected in message

    def test_research_cannot_jump_straight_to_live(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("mom", {"lookback": 252}))
        registry.attach_evidence(version.version_id, _evidence())
        with pytest.raises(PromotionError, match="cannot go straight to live"):
            registry.promote(version.version_id, Stage.LIVE, authorised_by="ops")

    def test_a_clean_promotion_records_who(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("mom", {"lookback": 252}))
        registry.attach_evidence(version.version_id, _evidence())
        promoted = registry.promote(version.version_id, Stage.PAPER, authorised_by="alice")
        assert promoted.stage is Stage.PAPER
        assert "alice" in promoted.history[-1]
        assert "FORCED" not in promoted.history[-1]

    def test_forcing_records_what_was_overridden(self, registry: StrategyRegistry) -> None:
        """An override that leaves no trace is a gate that was never there."""
        version = registry.register(StrategyVersion("mom", {"lookback": 252}))
        registry.attach_evidence(version.version_id, _evidence(deflated_sharpe=0.0))
        promoted = registry.promote(
            version.version_id, Stage.PAPER, authorised_by="alice", force=True
        )
        assert promoted.stage is Stage.PAPER
        assert "FORCED" in promoted.history[-1]
        assert "deflated Sharpe" in promoted.history[-1]

    def test_promotion_requires_an_authoriser(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("mom", {}))
        with pytest.raises(PromotionError, match="who authorised"):
            registry.promote(version.version_id, Stage.PAPER, authorised_by="")

    def test_retirement_requires_a_reason(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("mom", {}))
        with pytest.raises(ValueError, match="requires a reason"):
            registry.retire(version.version_id, "", authorised_by="ops")

    def test_a_retired_version_cannot_be_promoted(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("mom", {}))
        registry.retire(version.version_id, "stopped working", authorised_by="ops")
        with pytest.raises(PromotionError, match="retired"):
            registry.promote(version.version_id, Stage.PAPER, authorised_by="ops")

    def test_evidence_survives_a_round_trip(self, registry: StrategyRegistry) -> None:
        version = registry.register(StrategyVersion("mom", {"lookback": 252}))
        registry.attach_evidence(version.version_id, _evidence(sharpe=0.42))
        loaded = registry.get(version.version_id)
        assert loaded is not None
        assert loaded.evidence is not None
        assert loaded.evidence.sharpe == pytest.approx(0.42)

    def test_gate_permits_registration_at_research(self) -> None:
        gate = PromotionGate()
        version = StrategyVersion("mom", {})
        assert gate.check(version, Stage.RESEARCH) == []


# --- compliance ------------------------------------------------------------


class TestCompliance:
    def test_restricted_list_blocks_both_directions(self) -> None:
        rule = RestrictedList(frozenset({"aapl"}))
        context = Context(equity=1e6, prices={"AAPL": 100.0})
        assert rule.check(_order(side=Side.BUY), context) is not None
        assert rule.check(_order(side=Side.SELL), context) is not None

    def test_position_limit_counts_the_order_being_checked(self) -> None:
        """Checking against the position before the order permits the breach."""
        rule = MaxPositionNotional(max_fraction=0.10)
        context = Context(equity=100_000.0, positions={"AAPL": 95.0}, prices={"AAPL": 100.0})
        # 95 shares held = $9,500, under the $10,000 limit. Ten more breaches it.
        assert rule.check(_order(quantity=10.0), context) is not None

    def test_position_limit_permits_reducing_a_breach(self) -> None:
        rule = MaxPositionNotional(max_fraction=0.10)
        context = Context(equity=100_000.0, positions={"AAPL": 200.0}, prices={"AAPL": 100.0})
        assert rule.check(_order(side=Side.SELL, quantity=150.0), context) is None

    def test_gross_exposure_counts_the_whole_book(self) -> None:
        rule = MaxGrossExposure(max_multiple=1.0)
        context = Context(
            equity=100_000.0,
            positions={"AAPL": 400.0, "MSFT": -500.0},
            prices={"AAPL": 100.0, "MSFT": 100.0},
        )
        # $90,000 gross already; a $5,000 buy stays under, a $20,000 buy does not.
        assert rule.check(_order(quantity=50.0), context) is None
        assert rule.check(_order(quantity=200.0), context) is not None

    def test_daily_notional_bounds_a_runaway_loop(self) -> None:
        """A strategy oscillating long/short respects every position limit
        while trading its whole book repeatedly."""
        rule = MaxDailyNotional(max_fraction=0.25)
        context = Context(
            equity=100_000.0, prices={"AAPL": 100.0}, traded_today={"AAPL": 24_000.0}
        )
        assert rule.check(_order(quantity=5.0), context) is None
        assert rule.check(_order(quantity=50.0), context) is not None

    def test_long_only_permits_selling_down_to_flat(self) -> None:
        rule = LongOnly()
        context = Context(equity=1e6, positions={"AAPL": 100.0}, prices={"AAPL": 100.0})
        assert rule.check(_order(side=Side.SELL, quantity=100.0), context) is None
        assert rule.check(_order(side=Side.SELL, quantity=101.0), context) is not None

    def test_require_price_stops_the_fail_open_path(self) -> None:
        """Numeric rules abstain without a price; something must refuse."""
        order = Order(
            instrument=Instrument(symbol="XYZ", asset_class=AssetClass.EQUITY),
            side=Side.BUY,
            quantity=1e9,
        )
        context = Context(equity=1000.0)
        assert MaxPositionNotional().check(order, context) is None
        assert RequirePrice().check(order, context) is not None
        assert not default_engine().check(order, context).allowed

    def test_every_breach_is_reported_not_just_the_first(self) -> None:
        engine = default_engine(restricted=frozenset({"AAPL"}), max_position_fraction=0.001)
        context = Context(equity=1000.0, prices={"AAPL": 100.0})
        decision = engine.check(_order(quantity=100.0), context)
        assert not decision.allowed
        rules = {breach.rule for breach in decision.breaches}
        assert "restricted_list" in rules
        assert "max_position" in rules

    def test_a_permitted_order_says_so(self) -> None:
        engine = default_engine()
        context = Context(equity=1_000_000.0, prices={"AAPL": 100.0})
        decision = engine.check(_order(quantity=10.0), context)
        assert decision.allowed
        assert decision.render() == "permitted"

    def test_an_empty_engine_says_it_permits_everything(self) -> None:
        engine = ComplianceEngine()
        assert "No compliance rules" in engine.describe()
        assert engine.check(_order(), Context(equity=1.0)).allowed

    def test_breach_renders_the_arithmetic(self) -> None:
        rule = MaxPositionNotional(max_fraction=0.01)
        context = Context(equity=100_000.0, prices={"AAPL": 100.0})
        breach = rule.check(_order(quantity=100.0), context)
        assert breach is not None
        assert "10,000.00" in breach.render()
        assert "1,000.00" in breach.render()

    def test_max_fraction_must_be_a_fraction(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            MaxPositionNotional(max_fraction=0.0)


# --- TCA -------------------------------------------------------------------


def _execution(
    price: float, side: Side = Side.BUY, *, decision: float = 100.0, quantity: float = 100.0
) -> Execution:
    return Execution(
        fill=Fill(
            order_id=1,
            instrument=EQUITY,
            side=side,
            quantity=quantity,
            price=price,
            commission=2.0,
            timestamp=pd.Timestamp("2026-01-01"),
            strategy="mom",
        ),
        decision_price=decision,
        adv_notional=1e8,
        volatility=0.02,
    )


class TestTCA:
    def test_cost_is_signed_against_the_trade(self) -> None:
        """A buy above the decision price cost money; a sell above it made money."""
        assert _execution(101.0, Side.BUY).signed_cost_bps == pytest.approx(100.0)
        assert _execution(101.0, Side.SELL).signed_cost_bps == pytest.approx(-100.0)

    def test_weighting_is_by_notional(self) -> None:
        """A 50 bp cost on $1k and on $1m are not the same event."""
        report = analyse(
            [
                _execution(100.5, quantity=1.0),  # 50 bp on a tiny trade
                _execution(100.01, quantity=10_000.0),  # 1 bp on a large one
            ]
        )
        assert report.weighted_cost_bps < 5.0
        assert report.median_cost_bps == pytest.approx(25.5, abs=1.0)

    def test_outliers_are_counted_but_not_fitted(self) -> None:
        report = analyse([_execution(100.1), _execution(200.0)])
        assert report.n_fills == 2
        assert report.n_outliers == 1

    def test_a_model_that_underestimates_is_flagged_optimistic(self) -> None:
        # Realised 100 bp against a model predicting a couple of bp.
        report = analyse([_execution(101.0)])
        assert report.model_is_optimistic
        assert report.model_error_bps > 0

    def test_eta_is_fitted_through_the_origin(self) -> None:
        """The square-root law has no intercept; a fitted constant would absorb
        the spread and report an impact coefficient near zero."""
        executions = [_execution(100.0 + 0.05 * (i % 3 + 1)) for i in range(40)]
        report = analyse(executions)
        assert report.calibration_is_meaningful
        assert np.isfinite(report.fitted_impact_coefficient)

    def test_calibration_is_flagged_when_thin(self) -> None:
        report = analyse([_execution(100.1)])
        assert not report.calibration_is_meaningful
        assert "too few fills" in report.render()

    def test_attribution_by_symbol_and_strategy(self) -> None:
        report = analyse([_execution(100.5), _execution(100.1)])
        assert "AAPL" in report.by_symbol
        assert "mom" in report.by_strategy

    def test_an_empty_book_is_an_absence_not_a_zero(self) -> None:
        with pytest.raises(ValueError, match="absence of evidence"):
            analyse([])

    def test_a_zero_decision_price_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _execution(100.0, decision=0.0)

    def test_store_join_drops_fills_with_no_decision_price(self, tmp_path) -> None:
        """A defaulted decision price reports zero cost for exactly the
        executions nobody was supervising."""
        store = Store(tmp_path / "desk.db")
        priced = _order()
        priced_id, _ = store.record_order(
            priced, "k1", pd.Timestamp("2026-01-01"), decision_adv=1e8, decision_volatility=0.02
        )
        unpriced = Order(instrument=EQUITY, side=Side.BUY, quantity=5.0)
        unpriced_id, _ = store.record_order(unpriced, "k2", pd.Timestamp("2026-01-01"))

        for order_id, tag in ((priced_id, "a"), (unpriced_id, "b")):
            store.record_fill(
                order_id,
                Fill(
                    order_id=order_id,
                    instrument=EQUITY,
                    side=Side.BUY,
                    quantity=10.0,
                    price=100.5,
                    commission=0.1,
                    timestamp=pd.Timestamp("2026-01-01T10:00"),
                    strategy="mom",
                ),
                venue_fill_id=tag,
            )

        loaded = executions_from_store(store)
        assert loaded.n_rows == 2
        assert loaded.n_missing_decision == 1
        assert len(loaded.executions) == 1
        assert loaded.coverage == pytest.approx(0.5)
        report = analyse(loaded.executions)
        assert report.n_fills == 1

    def test_unlinked_fills_never_enter_the_join(self, tmp_path) -> None:
        """A bracket leg has no decision price and must not be invented one."""
        store = Store(tmp_path / "desk.db")
        store.record_fill(
            None,
            Fill(
                order_id=0,
                instrument=EQUITY,
                side=Side.SELL,
                quantity=10.0,
                price=99.0,
                commission=0.0,
                timestamp=pd.Timestamp("2026-01-01T11:00"),
            ),
            venue_fill_id="bracket-leg",
        )
        loaded = executions_from_store(store)
        assert loaded.n_rows == 0
        assert loaded.executions == []

    def test_point_value_survives_the_round_trip(self, tmp_path) -> None:
        """Without the multiplier a futures fill reads as its share equivalent
        and every basis-point figure is wrong by that factor."""
        store = Store(tmp_path / "desk.db")
        future = Instrument(
            symbol="ES", asset_class=AssetClass.FUTURES, point_value=50.0, tick_size=0.25
        )
        order = Order(
            instrument=future, side=Side.BUY, quantity=1.0, reference_price=5000.0
        )
        order_id, _ = store.record_order(order, "k", pd.Timestamp("2026-01-01"))
        store.record_fill(
            order_id,
            Fill(
                order_id=order_id,
                instrument=future,
                side=Side.BUY,
                quantity=1.0,
                price=5001.0,
                commission=2.0,
                timestamp=pd.Timestamp("2026-01-01T10:00"),
            ),
            venue_fill_id="f1",
        )
        loaded = executions_from_store(store)
        assert loaded.executions[0].notional == pytest.approx(5001.0 * 50.0)


def test_record_order_persists_the_decision_context(tmp_path) -> None:
    store = Store(tmp_path / "desk.db")
    order = _order()
    store.record_order(
        order, "k", pd.Timestamp("2026-01-01"), decision_adv=5e7, decision_volatility=0.015
    )
    row = store.fills_with_decisions()
    assert row == []  # no fills yet
    stored = store.get_order(1)
    assert stored is not None
    loaded = store._connection.execute(
        "SELECT reference_price, decision_adv, decision_volatility FROM orders"
    ).fetchone()
    assert loaded["reference_price"] == pytest.approx(100.0)
    assert loaded["decision_adv"] == pytest.approx(5e7)
    assert loaded["decision_volatility"] == pytest.approx(0.015)


def test_permissions_helper_is_available_on_this_platform() -> None:
    """Guards the FileSource permission check, which is POSIX-specific."""
    assert stat.S_IRUSR
    assert os.name == "posix"

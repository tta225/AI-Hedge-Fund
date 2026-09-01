"""Which strategy is running, at which parameters, and who let it.

A fill is a permanent record. The configuration that produced it is, at the
moment, a Python object that existed in a process that has since exited. Six
months later the question "what were we running in March" has no answer, and
the question that follows — "did the thing we tested match the thing we ran" —
cannot be asked at all.

That gap is the reason a research result and a live result are allowed to
diverge quietly. This module closes it with three ideas.

**A version is a hash of the configuration, not a number someone bumps.**
:class:`StrategyVersion` derives its id from the strategy name and its
parameters, so two versions are the same version exactly when they would trade
the same way. A parameter changed and not recorded is impossible: the id
changes whether anyone meant it to or not.

**Promotion is a gate with named conditions, not a deployment.** A version
moves ``RESEARCH → PAPER → LIVE`` and each step states what it required.
:class:`PromotionGate` refuses a promotion whose evidence does not clear the
bar, and the refusal names which condition failed. Given that five research
campaigns have produced nothing that clears a deflated-Sharpe threshold, the
gate's most likely correct behaviour is to refuse, and it is built to make
refusing the easy path rather than an act of discipline.

**Retirement is recorded, not implied by absence.** A strategy that stops
appearing in a config file has not been retired; it has been forgotten. A
retired version carries the reason and the date.

The registry stores nothing itself — it hands rows to the
:class:`~axiom.store.db.Store`'s meta table as JSON. That keeps the desk's
memory in one file, which is the one file the backup procedure knows about.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from axiom.store.db import Store

#: Key under which the registry lives in the store's meta table.
REGISTRY_KEY = "strategy_registry"
#: The deflated-Sharpe probability a candidate must reach to count as skill.
#: Duplicated from :mod:`axiom.research.panel_lab` rather than imported, so the
#: desk does not pull the research stack into a live process; a test asserts the
#: two stay equal, which is the part that actually matters.
SKILL_THRESHOLD = 0.95
#: Characters of the config digest that form a version id. Twelve makes an
#: accidental collision across any plausible number of versions negligible
#: while staying short enough to paste into a conversation.
VERSION_CHARS = 12


class Stage(str, Enum):
    """How far a strategy has been allowed to go."""

    #: Exists, has been measured, is not authorised to trade anything.
    RESEARCH = "research"
    #: Trading simulated money against live prices.
    PAPER = "paper"
    #: Trading real money.
    LIVE = "live"
    #: Was one of the above and is no longer, for a recorded reason.
    RETIRED = "retired"

    @property
    def rank(self) -> int:
        return {"research": 0, "paper": 1, "live": 2, "retired": -1}[self.value]


class PromotionError(RuntimeError):
    """A promotion was refused, or was structurally impossible."""


@dataclass(frozen=True, slots=True)
class Evidence:
    """What is known about a version's performance, and on what.

    Every field is something a promotion decision needs and none of it is
    optional-with-a-default, because a default here is a way to promote on
    evidence that was never gathered. ``deflated_sharpe`` in particular has no
    sensible default: zero would read as "measured and worthless" and any
    positive number would read as a result.
    """

    #: Out-of-sample Sharpe, net of the cost model.
    sharpe: float
    #: Deflated Sharpe *probability* on ``[0, 1]`` — the number
    #: :class:`~axiom.research.panel_lab.PanelSearchResult.deflated` reports,
    #: not a Sharpe ratio. Passing a raw Sharpe here would sail through the
    #: gate, so the units are named rather than implied.
    deflated_sharpe: float
    #: Trials in the search that selected this configuration. A deflated Sharpe
    #: computed against the wrong trial count is not a deflated Sharpe.
    n_trials: int
    #: Bars the out-of-sample result covers.
    n_observations: int
    #: Annualised turnover, so a promotion cannot ignore what the desk learned
    #: about turnover being the dominant term.
    annual_turnover: float
    #: Whether the result came from real, non-synthetic bars.
    is_evidential: bool
    #: Free text: the universe, the window, the caveats.
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sharpe": self.sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "annual_turnover": self.annual_turnover,
            "is_evidential": self.is_evidential,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        return cls(
            sharpe=float(data["sharpe"]),
            deflated_sharpe=float(data["deflated_sharpe"]),
            n_trials=int(data["n_trials"]),
            n_observations=int(data["n_observations"]),
            annual_turnover=float(data["annual_turnover"]),
            is_evidential=bool(data["is_evidential"]),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class StrategyVersion:
    """One strategy at one exact configuration."""

    name: str
    parameters: dict[str, Any]
    stage: Stage = Stage.RESEARCH
    created_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now("UTC").tz_localize(None))
    evidence: Evidence | None = None
    #: Free text recorded at each stage change, newest last.
    history: tuple[str, ...] = ()

    @property
    def version_id(self) -> str:
        """A digest of what this version *is*, so identity cannot drift.

        Parameters are serialised with sorted keys so that dictionary ordering
        — which is not part of a configuration's meaning — cannot produce two
        ids for one configuration.
        """
        payload = json.dumps(
            {"name": self.name, "parameters": self.parameters}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:VERSION_CHARS]

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyVersion:
        evidence = data.get("evidence")
        return cls(
            name=str(data["name"]),
            parameters=dict(data["parameters"]),
            stage=Stage(data["stage"]),
            created_at=pd.Timestamp(data["created_at"]),
            evidence=Evidence.from_dict(evidence) if evidence else None,
            history=tuple(data.get("history", ())),
        )


@dataclass(frozen=True, slots=True)
class PromotionGate:
    """The conditions a version must meet to reach a stage.

    The thresholds are the ones this project's own research has been held to,
    which is the point: a gate calibrated more leniently than the research bar
    would let through exactly the candidates the research rejected.
    """

    #: Minimum deflated-Sharpe **probability** — the quantity
    #: :class:`~axiom.research.panel_lab.PanelLab` reports, which is the
    #: probability that the observed Sharpe reflects skill rather than the best
    #: of N trials landing well. It lives on ``[0, 1]``, so a threshold of 0.0
    #: would pass literally everything; it is set to the same 0.95 the research
    #: harness uses, because a promotion gate looser than the research bar
    #: admits exactly the candidates the research rejected.
    min_deflated_sharpe: float = SKILL_THRESHOLD
    #: Raw out-of-sample Sharpe. The published median across 61 anomalies is
    #: 0.354; requiring at least half of that is a low bar deliberately, since
    #: the deflated condition is the binding one.
    min_sharpe: float = 0.2
    #: Bars of out-of-sample evidence. A Sharpe estimated on 60 observations has
    #: a standard error around 0.26 and is not a measurement.
    min_observations: int = 252
    #: Above this, the turnover study says costs dominate whatever direction is
    #: there. Refusing outright rather than warning, because every candidate
    #: above 30x in the cross-sectional campaign lost money.
    max_annual_turnover: float = 30.0
    #: Synthetic bars can never authorise trading.
    require_evidential: bool = True

    def check(self, version: StrategyVersion, target: Stage) -> list[str]:
        """Reasons this promotion should be refused. Empty means it may proceed."""
        if target is Stage.RESEARCH:
            return []
        if target is Stage.RETIRED:
            return []

        failures: list[str] = []
        evidence = version.evidence
        if evidence is None:
            return [
                f"{version.label} has no recorded evidence; a promotion to "
                f"{target.value} on unmeasured performance is not a decision, it is a guess"
            ]

        if self.require_evidential and not evidence.is_evidential:
            failures.append(
                "evidence is not evidential — it traces back to synthetic bars, which "
                "cannot authorise trading at any stage"
            )
        if evidence.deflated_sharpe < self.min_deflated_sharpe:
            failures.append(
                f"deflated Sharpe probability {evidence.deflated_sharpe:.3f} is below "
                f"{self.min_deflated_sharpe:.3f} over {evidence.n_trials} trials"
            )
        if evidence.sharpe < self.min_sharpe:
            failures.append(
                f"out-of-sample Sharpe {evidence.sharpe:.3f} is below {self.min_sharpe:.3f}"
            )
        if evidence.n_observations < self.min_observations:
            failures.append(
                f"{evidence.n_observations} out-of-sample bars is below the "
                f"{self.min_observations} needed for the Sharpe estimate to mean anything"
            )
        if evidence.annual_turnover > self.max_annual_turnover:
            failures.append(
                f"annual turnover {evidence.annual_turnover:.1f}x exceeds "
                f"{self.max_annual_turnover:.1f}x, above which measured cost has "
                "dominated measured direction in every campaign run here"
            )
        return failures


class StrategyRegistry:
    """The record of what has been run, and what is authorised to run.

    Backed by the store's meta table so it lives in the same file as the fills
    it explains — and is therefore covered by the same backup.
    """

    def __init__(self, store: Store, gate: PromotionGate | None = None) -> None:
        self.store = store
        self.gate = gate or PromotionGate()

    def _load(self) -> dict[str, StrategyVersion]:
        raw = self.store.get_meta(REGISTRY_KEY, "")
        if not raw:
            return {}
        data = json.loads(raw)
        return {key: StrategyVersion.from_dict(value) for key, value in data.items()}

    def _save(self, versions: dict[str, StrategyVersion]) -> None:
        self.store.set_meta(
            REGISTRY_KEY,
            json.dumps({key: value.to_dict() for key, value in versions.items()}, default=str),
        )

    def register(self, version: StrategyVersion) -> StrategyVersion:
        """Record a version, or return the one already recorded under its id.

        Registration is idempotent because the id is derived from the content:
        registering the same configuration twice is not two versions, and
        silently creating a second row would break the property that makes the
        registry worth having.

        A configuration that already exists at a *higher* stage is returned
        unchanged rather than reset to research — re-running a script must not
        demote something that is live.
        """
        versions = self._load()
        existing = versions.get(version.version_id)
        if existing is not None:
            return existing
        versions[version.version_id] = version
        self._save(versions)
        return version

    def get(self, version_id: str) -> StrategyVersion | None:
        return self._load().get(version_id)

    def all(self) -> list[StrategyVersion]:
        return sorted(self._load().values(), key=lambda v: (v.name, v.created_at))

    def at_stage(self, stage: Stage) -> list[StrategyVersion]:
        return [version for version in self.all() if version.stage is stage]

    @property
    def live(self) -> list[StrategyVersion]:
        return self.at_stage(Stage.LIVE)

    def attach_evidence(self, version_id: str, evidence: Evidence) -> StrategyVersion:
        """Record what a version measured. Overwrites, and says so in history.

        Overwriting rather than appending because there is one current answer
        to "how does this perform"; the history line preserves that an earlier
        answer existed, which is what makes a quietly improved number visible.
        """
        versions = self._load()
        version = versions.get(version_id)
        if version is None:
            raise KeyError(f"no version {version_id!r} in the registry")
        note = (
            f"{pd.Timestamp.now('UTC').tz_localize(None).isoformat()} evidence recorded: "
            f"sharpe {evidence.sharpe:+.3f}, deflated {evidence.deflated_sharpe:+.3f}, "
            f"{evidence.n_trials} trials"
        )
        updated = StrategyVersion(
            name=version.name,
            parameters=version.parameters,
            stage=version.stage,
            created_at=version.created_at,
            evidence=evidence,
            history=(*version.history, note),
        )
        versions[version_id] = updated
        self._save(versions)
        return updated

    def promote(
        self, version_id: str, target: Stage, *, authorised_by: str, force: bool = False
    ) -> StrategyVersion:
        """Move a version to a higher stage, if its evidence allows it.

        ``force`` exists because there are legitimate reasons to override — a
        strategy being run at nominal size specifically to gather live
        evidence, for instance. It does not suppress the check; it records that
        the check failed and was overridden, by whom, with what failing. An
        override that leaves no trace is indistinguishable from a gate that was
        never there.
        """
        versions = self._load()
        version = versions.get(version_id)
        if version is None:
            raise KeyError(f"no version {version_id!r} in the registry")
        if not authorised_by:
            raise PromotionError("a promotion must record who authorised it")

        if version.stage is Stage.RETIRED and target is not Stage.RETIRED:
            raise PromotionError(
                f"{version.label} is retired. Register the configuration again to "
                "revive it, so the revival is a new decision with its own record."
            )
        # Skipping paper is not a shortcut, it is the removal of the only stage
        # where a mistake is free.
        if target is Stage.LIVE and version.stage is Stage.RESEARCH and not force:
            raise PromotionError(
                f"{version.label} is at research and cannot go straight to live. "
                "Paper trading is the only stage where an execution bug costs "
                "nothing; pass force=True with a reason if it must be skipped."
            )

        failures = self.gate.check(version, target)
        if failures and not force:
            detail = "\n  - ".join(failures)
            raise PromotionError(
                f"refusing to promote {version.label} to {target.value}:\n  - {detail}"
            )

        stamp = pd.Timestamp.now("UTC").tz_localize(None).isoformat()
        if failures:
            note = (
                f"{stamp} FORCED {version.stage.value} -> {target.value} by "
                f"{authorised_by}, overriding: {'; '.join(failures)}"
            )
        else:
            note = f"{stamp} {version.stage.value} -> {target.value} by {authorised_by}"

        updated = StrategyVersion(
            name=version.name,
            parameters=version.parameters,
            stage=target,
            created_at=version.created_at,
            evidence=version.evidence,
            history=(*version.history, note),
        )
        versions[version_id] = updated
        self._save(versions)
        return updated

    def retire(self, version_id: str, reason: str, *, authorised_by: str) -> StrategyVersion:
        """Stop a version, with the reason attached.

        A reason is required. "Retired" without one leaves the next person to
        look at it unable to tell a strategy that stopped working from one that
        was superseded, and those call for opposite responses.
        """
        if not reason:
            raise ValueError(
                "retiring a strategy requires a reason; without one nobody can tell "
                "later whether it broke or was merely replaced"
            )
        versions = self._load()
        version = versions.get(version_id)
        if version is None:
            raise KeyError(f"no version {version_id!r} in the registry")
        stamp = pd.Timestamp.now("UTC").tz_localize(None).isoformat()
        updated = StrategyVersion(
            name=version.name,
            parameters=version.parameters,
            stage=Stage.RETIRED,
            created_at=version.created_at,
            evidence=version.evidence,
            history=(*version.history, f"{stamp} retired by {authorised_by}: {reason}"),
        )
        versions[version_id] = updated
        self._save(versions)
        return updated

    def render(self) -> str:
        versions = self.all()
        if not versions:
            return "Strategy registry is empty."
        lines = [f"Strategy registry — {len(versions)} version(s)"]
        for version in versions:
            evidence = version.evidence
            measure = (
                f"sharpe {evidence.sharpe:+.3f} deflated {evidence.deflated_sharpe:+.3f} "
                f"turn {evidence.annual_turnover:.1f}x"
                if evidence
                else "no evidence recorded"
            )
            lines.append(f"  {version.stage.value:<8} {version.label:<32} {measure}")
        return "\n".join(lines)

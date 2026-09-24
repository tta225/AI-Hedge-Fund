"""Structural enforcement of the rule this package is built on.

The module docstring in :mod:`axiom.agents.base` states the principle plainly:

    *Agents never produce numbers.*

Until now that was enforced by a sentence in a system prompt asking the model
not to invent figures. That is a request, not a control. A model that restates
``profit factor 1.8`` when the measured value was ``1.2`` produces a report that
is confidently, quietly wrong — and every downstream reader treats the narrative
and the facts as equally measured, because they are printed together.

:func:`check_numeric_fidelity` closes that gap. Every number appearing in a
generated narrative must be traceable to a number that was *given to the model*.
Anything else is flagged, attached to the report, and rendered next to the prose
that contains it.

The check is deliberately asymmetric. A false positive costs a reader one glance
at a flagged figure. A false negative puts a fabricated number in front of
someone sizing a position. Where the two trade off, this errs toward flagging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Numbers as they appear in prose: ``1,234.5``, ``-0.75``, ``42%``, ``$1,200``.
_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")

#: Stripped before scanning — every digit inside them is structure, not a claim.
#: An ISO timestamp alone contributes six unrelated integers.
_NOISE = (
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:[.+-]\S*)?"),  # timestamp
    re.compile(r"\d{4}-\d{2}-\d{2}"),  # date
    re.compile(r"\d{1,2}:\d{2}"),  # clock time
)

#: Integers this small are prose, not measurement — "the two sides", "3 reasons",
#: an ordinal, a list marker. Above the ceiling, a bare integer is a claim.
_PROSE_INTEGER_CEILING = 12

#: Relative tolerance for matching a narrative number to a supplied one. Wide
#: enough that a model rounding 1.2345 to 1.23 is not flagged, tight enough that
#: 1.2 and 1.8 never match.
_RELATIVE_TOLERANCE = 0.005


def _parse(token: str) -> float | None:
    """Read a prose number. Percent signs are dropped, not divided.

    ``45%`` is compared against the fact ``45.0``, because that is how these
    facts are stored — ``win_rate_pct`` is 45.0, not 0.45. Dividing here would
    make every correctly-quoted percentage look unsupported.
    """
    cleaned = token.replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract(text: str) -> list[tuple[str, float]]:
    for pattern in _NOISE:
        text = pattern.sub(" ", text)
    found: list[tuple[str, float]] = []
    for match in _NUMBER.finditer(text):
        value = _parse(match.group())
        if value is not None:
            found.append((match.group(), value))
    return found


def _supported_values(facts: dict[str, Any], prompt: str) -> list[float]:
    """Every number the model was legitimately given.

    The prompt is scanned as well as the facts because the prompt *is* how facts
    reach the model, and it also carries thresholds the model may quote back
    ("below 2.0"). Quoting a number from its own instructions is not fabrication.
    """
    values: list[float] = []
    for value in facts.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
        elif isinstance(value, str):
            values.extend(v for _, v in _extract(value))
    values.extend(v for _, v in _extract(prompt))
    return values


def _is_supported(value: float, supported: list[float]) -> bool:
    for candidate in supported:
        if candidate == value:
            return True
        scale = max(abs(candidate), abs(value))
        if scale and abs(candidate - value) <= scale * _RELATIVE_TOLERANCE:
            return True
        # A model quoting 0.62 as "62%" is reporting the measured value, not
        # inventing one. Both directions, since facts store both conventions.
        for converted in (candidate * 100.0, candidate / 100.0):
            scale = max(abs(converted), abs(value))
            if scale and abs(converted - value) <= scale * _RELATIVE_TOLERANCE:
                return True
    return False


@dataclass(frozen=True, slots=True)
class NumericFidelityReport:
    """What the narrative claimed that the facts do not support."""

    #: Numbers as they appeared in the prose, in order of appearance.
    unsupported: tuple[str, ...] = ()
    #: How many numbers were examined — context for an empty result.
    checked: int = 0

    @property
    def clean(self) -> bool:
        return not self.unsupported

    def warning(self) -> str | None:
        """The line to attach to a report, or ``None`` when nothing is wrong."""
        if self.clean:
            return None
        listed = ", ".join(self.unsupported[:6])
        more = "" if len(self.unsupported) <= 6 else f" (+{len(self.unsupported) - 6} more)"
        return (
            f"UNVERIFIED FIGURES in narrative — not traceable to measured facts: "
            f"{listed}{more}. Treat the prose as unreliable and read the facts."
        )


def check_numeric_fidelity(
    narrative: str,
    facts: dict[str, Any],
    prompt: str = "",
    *,
    prose_integer_ceiling: int = _PROSE_INTEGER_CEILING,
) -> NumericFidelityReport:
    """Flag numbers in ``narrative`` that were never given to the model.

    Args:
        narrative: model-generated prose.
        facts: the measured values the agent computed.
        prompt: the text the model actually received. Supplying it sharply
            reduces false positives, because thresholds and framing numbers in
            the instructions are then recognised as given rather than invented.
        prose_integer_ceiling: small integers at or below this are treated as
            prose ("both sides", "three concerns") rather than as claims.

    Returns:
        A report. Empty ``unsupported`` means every figure traced back to an
        input — it does not mean the *reasoning* is sound, only that the
        arithmetic was not hallucinated.
    """
    if not narrative.strip():
        return NumericFidelityReport()

    supported = _supported_values(facts, prompt)
    unsupported: list[str] = []
    checked = 0

    for token, value in _extract(narrative):
        if value.is_integer() and abs(value) <= prose_integer_ceiling:
            continue
        checked += 1
        if not _is_supported(value, supported):
            unsupported.append(token)

    # Order-preserving dedup: repeating a fabricated figure is one defect, and
    # listing it five times buries the others.
    unique = tuple(dict.fromkeys(unsupported))
    return NumericFidelityReport(unsupported=unique, checked=checked)

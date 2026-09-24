"""Rules that stop an order before it is sent, and say which rule stopped it.

:mod:`axiom.desk.guards` answers "is the desk in a fit state to trade at all" —
stale data, an open halt, a failed reconciliation. :mod:`axiom.portfolio.risk`
answers "is this book too concentrated or too levered". Neither answers the
third question, which is the one a compliance officer asks: **was this order
permitted**, irrespective of whether it was wise.

The distinction is not pedantic. A risk limit is a judgement that can be
overridden by someone with the authority to take more risk. A mandate
restriction cannot be overridden by anyone on the desk, because it is not a
statement about risk appetite — it is a statement about what this pool of
capital is allowed to do. Collapsing the two means the second gets waived by
someone exercising authority over the first.

So the rules here are deliberately dumb, deterministic, and checked one at a
time, and every refusal names the rule and the number that broke it. A block
that says "risk limit exceeded" without saying which limit and by how much
teaches the desk nothing and gets disabled within a week.

Evaluation is **all rules, always** — not short-circuit. An order that breaks
three rules should report three, because fixing the first and resubmitting
into the second is how a bad order gets sent on the third attempt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from axiom.core.types import Side
from axiom.execution.base import Order


@dataclass(frozen=True, slots=True)
class Breach:
    """One rule, broken, with the arithmetic that broke it."""

    rule: str
    detail: str
    #: The observed value and the limit it exceeded, when the rule is numeric.
    observed: float | None = None
    limit: float | None = None

    def render(self) -> str:
        if self.observed is not None and self.limit is not None:
            return f"{self.rule}: {self.detail} ({self.observed:,.2f} vs limit {self.limit:,.2f})"
        return f"{self.rule}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether an order may be sent, and why not."""

    allowed: bool
    breaches: tuple[Breach, ...] = ()

    @property
    def reason(self) -> str:
        return "; ".join(breach.render() for breach in self.breaches)

    def render(self) -> str:
        if self.allowed:
            return "permitted"
        return "BLOCKED\n  " + "\n  ".join(breach.render() for breach in self.breaches)


@dataclass(frozen=True, slots=True)
class Context:
    """What a rule may know about the account when it decides.

    Passed explicitly rather than read from a store inside each rule, so a rule
    is a pure function of its inputs and can be tested without a database. It
    also means every rule sees the *same* snapshot: rules that each query
    independently can disagree about the position size mid-evaluation.
    """

    equity: float
    positions: dict[str, float] = field(default_factory=dict)
    #: Last known price per symbol, for valuing the order and the book.
    prices: dict[str, float] = field(default_factory=dict)
    #: Orders already sent today, by symbol, in absolute notional.
    traded_today: dict[str, float] = field(default_factory=dict)

    def price_of(self, symbol: str) -> float | None:
        price = self.prices.get(symbol)
        return price if price and price > 0 else None

    def position_notional(self, symbol: str) -> float:
        price = self.price_of(symbol)
        if price is None:
            return 0.0
        return self.positions.get(symbol, 0.0) * price

    @property
    def gross_notional(self) -> float:
        return sum(abs(self.position_notional(symbol)) for symbol in self.positions)


class Rule(ABC):
    """One compliance condition."""

    name: str = "rule"

    @abstractmethod
    def check(self, order: Order, context: Context) -> Breach | None:
        """``None`` when the order satisfies this rule."""


@dataclass(frozen=True, slots=True)
class RestrictedList(Rule):
    """Symbols this desk may not trade, at all, in either direction.

    The canonical use is a restricted list from a compliance function — names
    under an information barrier, names in a deal the firm is advising on. It
    also serves the mundane case of a symbol known to be broken in the data
    feed, which is worth the same treatment: a name you cannot price is a name
    you cannot trade.

    Matching is exact and case-insensitive. Not a prefix or pattern match:
    a pattern that accidentally matches too much blocks trading silently, and a
    pattern that matches too little fails open, which is worse.
    """

    symbols: frozenset[str]
    reason: str = "on the restricted list"
    name: str = "restricted_list"

    def check(self, order: Order, context: Context) -> Breach | None:
        if order.instrument.symbol.upper() in {s.upper() for s in self.symbols}:
            return Breach(self.name, f"{order.instrument.symbol} is {self.reason}")
        return None


@dataclass(frozen=True, slots=True)
class MaxPositionNotional(Rule):
    """No single name above a fraction of equity, counting the order.

    Counting the order is the whole point. A limit checked against the position
    *before* the order permits the order that breaches it, and then blocks the
    next one — by which time the breach exists.
    """

    max_fraction: float = 0.10
    name: str = "max_position"

    def __post_init__(self) -> None:
        if not 0 < self.max_fraction <= 1:
            raise ValueError("max_fraction must be in (0, 1]")

    def check(self, order: Order, context: Context) -> Breach | None:
        symbol = order.instrument.symbol
        price = context.price_of(symbol) or order.limit_price or order.reference_price
        if not price or context.equity <= 0:
            return None  # unpriceable; MissingPrice is the rule that catches this
        signed = order.quantity * (1.0 if order.side is Side.BUY else -1.0)
        delta = signed * price * order.instrument.point_value
        resulting = abs(context.position_notional(symbol) + delta)
        limit = self.max_fraction * context.equity
        if resulting > limit:
            return Breach(
                self.name,
                f"{symbol} would reach {resulting / context.equity:.1%} of equity",
                observed=resulting,
                limit=limit,
            )
        return None


@dataclass(frozen=True, slots=True)
class MaxGrossExposure(Rule):
    """Total absolute exposure, as a multiple of equity.

    A long/short book can be flat on net and enormously levered on gross, and
    it is gross that determines what a bad day costs.
    """

    max_multiple: float = 1.5
    name: str = "max_gross"

    def check(self, order: Order, context: Context) -> Breach | None:
        symbol = order.instrument.symbol
        price = context.price_of(symbol) or order.limit_price or order.reference_price
        if not price or context.equity <= 0:
            return None
        signed = order.quantity * (1.0 if order.side is Side.BUY else -1.0)
        delta = signed * price * order.instrument.point_value
        current = context.position_notional(symbol)
        # Only this name's contribution changes; the rest of the book is
        # unaffected by this order, so recomputing it would just be noise.
        resulting = context.gross_notional - abs(current) + abs(current + delta)
        limit = self.max_multiple * context.equity
        if resulting > limit:
            return Breach(
                self.name,
                f"gross exposure would reach {resulting / context.equity:.2f}x equity",
                observed=resulting,
                limit=limit,
            )
        return None


@dataclass(frozen=True, slots=True)
class MaxDailyNotional(Rule):
    """A cap on how much a single name can be traded in one session.

    This is the runaway-loop rule. Every other limit here constrains a
    *position*, and a strategy oscillating between long and short can respect
    all of them while trading its entire book every minute and paying the
    spread each time. That failure mode has a name — a fat finger in a loop —
    and it is bounded by notional traded, not by position held.
    """

    max_fraction: float = 0.25
    name: str = "max_daily_notional"

    def check(self, order: Order, context: Context) -> Breach | None:
        symbol = order.instrument.symbol
        price = context.price_of(symbol) or order.limit_price or order.reference_price
        if not price or context.equity <= 0:
            return None
        notional = abs(order.quantity * price * order.instrument.point_value)
        resulting = context.traded_today.get(symbol, 0.0) + notional
        limit = self.max_fraction * context.equity
        if resulting > limit:
            return Breach(
                self.name,
                f"{symbol} would have traded {resulting / context.equity:.1%} of equity today",
                observed=resulting,
                limit=limit,
            )
        return None


@dataclass(frozen=True, slots=True)
class LongOnly(Rule):
    """No short positions, for a mandate that forbids them.

    Checks the *resulting* position rather than the order side, so selling a
    long down to flat is permitted and selling through zero is not. A rule that
    blocked every sell would prevent the mandate from ever exiting.
    """

    name: str = "long_only"

    def check(self, order: Order, context: Context) -> Breach | None:
        if order.side is Side.BUY:
            return None
        symbol = order.instrument.symbol
        resulting = context.positions.get(symbol, 0.0) - order.quantity
        if resulting < 0:
            return Breach(
                self.name,
                f"{symbol} would go short {abs(resulting):,.4f} units under a long-only mandate",
                observed=resulting,
                limit=0.0,
            )
        return None


@dataclass(frozen=True, slots=True)
class RequirePrice(Rule):
    """Refuse an order for a symbol the desk cannot value.

    Every numeric rule above returns ``None`` when it cannot find a price,
    because a rule that guesses is worse than one that abstains. That is only
    safe if something refuses the unpriceable order outright — otherwise an
    order for a symbol missing from the price map passes every check by
    default, which is failing open at exactly the wrong moment.
    """

    name: str = "require_price"

    def check(self, order: Order, context: Context) -> Breach | None:
        symbol = order.instrument.symbol
        if context.price_of(symbol) or order.limit_price or order.reference_price:
            return None
        return Breach(
            self.name,
            f"no price available for {symbol}, so no notional limit can be evaluated",
        )


@dataclass(slots=True)
class ComplianceEngine:
    """Every rule, evaluated against every order, before it is sent.

    Order of evaluation is the order the rules were given, and it does not
    affect the outcome — all rules run regardless of earlier breaches.
    """

    rules: tuple[Rule, ...] = ()

    def check(self, order: Order, context: Context) -> Decision:
        breaches: list[Breach] = []
        for rule in self.rules:
            breach = rule.check(order, context)
            if breach is not None:
                breaches.append(breach)
        return Decision(allowed=not breaches, breaches=tuple(breaches))

    def describe(self) -> str:
        if not self.rules:
            return (
                "No compliance rules configured. Every order will be permitted, which "
                "is a decision — make it deliberately."
            )
        return "Compliance rules: " + ", ".join(rule.name for rule in self.rules)


def default_engine(
    *,
    restricted: frozenset[str] = frozenset(),
    max_position_fraction: float = 0.10,
    max_gross_multiple: float = 1.5,
    max_daily_fraction: float = 0.25,
    long_only: bool = False,
) -> ComplianceEngine:
    """A conservative starting set.

    ``RequirePrice`` is first because it is the rule that makes the others'
    abstention safe. The numeric defaults are not calibrated to any mandate —
    they are round numbers chosen to be tighter than anything the desk
    currently does, so that arriving at them is an event someone notices.
    """
    rules: list[Rule] = [RequirePrice()]
    if restricted:
        rules.append(RestrictedList(restricted))
    rules.append(MaxPositionNotional(max_position_fraction))
    rules.append(MaxGrossExposure(max_gross_multiple))
    rules.append(MaxDailyNotional(max_daily_fraction))
    if long_only:
        rules.append(LongOnly())
    return ComplianceEngine(tuple(rules))

"""Which names existed on a given day — not which names exist now.

Every backtest in this project so far has been run on names that are liquid
*today*, applied backward six years. That construction deletes, from the entire
history, exactly the companies that failed: Silicon Valley Bank stops trading on
2023-03-09, First Republic on 2023-04-28, and a universe assembled from today's
liquid tickers contains neither. A strategy tested on such a universe was never
given the chance to own them, and its measured return is the return of a
portfolio that knew in advance which banks would survive.

That is survivorship bias, and it is not a footnote-sized effect. It is the
reason every figure in `docs/` up to now has carried the words "upper bound".

This module removes it, using data the desk already has access to. Alpaca's
asset endpoint lists ~19,000 **inactive** US equities alongside the active ones,
and its data API still serves their bars — which stop, correctly, on the day
each name stopped trading. That is enough to reconstruct membership as a
function of time.

Three ideas carry the design:

**A listing is an interval, not a flag.** :class:`Listing` records the first and
last bar a name actually traded. Membership on a date is a question about that
interval, so a name that had not yet listed is as absent as one that has died —
both are ways a naive universe leaks the future.

**Selection is itself point-in-time.** :meth:`PointInTimeUniverse.select_at`
ranks candidates by trailing dollar volume computed *strictly before* the
selection date, from the names trading *on* that date. Picking "the 65 most
liquid names" using volume measured over the whole sample is the same bias in a
subtler form — it selects on information from the future even when every
constituent is legitimately alive.

**The bias is measured, not just avoided.** :meth:`PointInTimeUniverse.survivorship_report`
answers "how many of the names we would have held on this date were later
delisted", so the size of what the old universes were deleting is a number
rather than a caveat.

What this still does not do, stated plainly: the last observed bar is not the
same as the terminal value. A name acquired at a premium and a name that fell to
zero both simply stop, and this treats the final close as the exit price. For an
acquisition that is roughly right. For a bankruptcy that stops trading before
the equity is wiped out, it is optimistic — the residual loss is not modelled.
That is a smaller bias than the one being removed, and it points the same way,
so results here remain an upper bound; a considerably tighter one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Exchanges whose names are in scope. OTC is excluded deliberately: it is the
#: majority of the inactive list by count, is thinly and irregularly traded, and
#: including it would swamp the universe with names no desk would trade while
#: adding mostly noise to any cross-sectional ranking.
MAJOR_EXCHANGES = frozenset({"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS"})
#: Trailing window for the liquidity ranking that selects the universe.
DEFAULT_SELECTION_LOOKBACK = 252
#: A name must have traded at least this fraction of the lookback to be
#: selectable. A name that printed on three days has an average dollar volume
#: that is not an estimate of anything.
MIN_COVERAGE = 0.6
#: A name whose last bar is within this many days of the end of the data is
#: treated as still trading rather than as delisted. Without it, a name that
#: simply did not print on the final session — thin, halted, a holiday — is
#: indistinguishable from one that died.
ALIVE_TOLERANCE_DAYS = 7


@dataclass(frozen=True, slots=True)
class Listing:
    """One security's observed trading life.

    Derived from bars rather than from a metadata field, because the bars are
    what a backtest can actually trade. A name whose asset record says "active"
    but which last printed in 2021 is, for every purpose here, delisted in 2021.
    """

    symbol: str
    first_bar: pd.Timestamp
    last_bar: pd.Timestamp
    #: Bars actually observed between first and last. Sparse coverage is the
    #: signal that a name is too thin to rank.
    n_bars: int
    exchange: str = ""
    name: str = ""
    #: Whether the venue still lists the security. Recorded but never used for
    #: membership: the bars decide that.
    is_active: bool = True

    def __post_init__(self) -> None:
        if self.last_bar < self.first_bar:
            raise ValueError(f"{self.symbol}: last_bar precedes first_bar")

    def was_trading_at(self, date: pd.Timestamp) -> bool:
        """Was this name trading on ``date``?

        Inclusive at both ends. The last bar is a day the name traded, so it is
        a day the book could still have exited.
        """
        return bool(self.first_bar <= date <= self.last_bar)

    @property
    def delisted(self) -> bool:
        return not self.is_active

    def died_before(self, date: pd.Timestamp) -> bool:
        return bool(self.last_bar < date)

    @property
    def span_days(self) -> int:
        return int((self.last_bar - self.first_bar).days)


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """The universe as it stood on one date, and what it cost to know that."""

    as_of: pd.Timestamp
    symbols: tuple[str, ...]
    #: Names that were trading but did not make the liquidity cut.
    n_eligible: int
    #: Members of this snapshot that had stopped trading one year later. The
    #: number a survivorship-biased universe silently sets to zero.
    n_later_delisted: int = 0

    @property
    def attrition_rate(self) -> float:
        return self.n_later_delisted / len(self.symbols) if self.symbols else 0.0

    def render(self) -> str:
        return (
            f"{self.as_of.date()}: {len(self.symbols)} of {self.n_eligible} eligible, "
            f"{self.n_later_delisted} later delisted ({self.attrition_rate:.1%})"
        )


@dataclass
class PointInTimeUniverse:
    """Listings, and membership as a function of date.

    Holds no prices. Selection needs volume history, which is passed in at the
    point of use rather than stored here, so this object stays small enough to
    serialise and cheap enough to construct from a metadata file.
    """

    listings: dict[str, Listing] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.listings)

    def add(self, listing: Listing) -> None:
        self.listings[listing.symbol] = listing

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.listings))

    @property
    def delisted(self) -> tuple[str, ...]:
        return tuple(sorted(s for s, listing in self.listings.items() if listing.delisted))

    def members_at(self, date: pd.Timestamp) -> tuple[str, ...]:
        """Every name trading on ``date``, liquid or not."""
        date = pd.Timestamp(date)
        return tuple(
            sorted(s for s, listing in self.listings.items() if listing.was_trading_at(date))
        )

    def select_at(
        self,
        date: pd.Timestamp,
        *,
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
        top_n: int,
        lookback: int = DEFAULT_SELECTION_LOOKBACK,
        min_coverage: float = MIN_COVERAGE,
        exclude: Iterable[str] = (),
    ) -> UniverseSnapshot:
        """The ``top_n`` most liquid names trading on ``date``.

        Liquidity is trailing dollar volume over ``lookback`` bars ending
        **strictly before** ``date``. That exclusivity is the entire point: a
        universe chosen using the volume of the day it starts trading knows
        something the trader did not.

        Args:
            closes, volumes: wide frames indexed by timestamp, columns by
                symbol, containing NaN where a name was not trading.
            top_n: how many names to keep.
            lookback: bars of volume history behind the ranking.
            min_coverage: fraction of the lookback a name must have traded.
            exclude: symbols to skip — a restricted list, or the discovery
                universe when building a disjoint confirmation set.
        """
        date = pd.Timestamp(date)
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        trading = set(self.members_at(date)) - set(exclude)
        if not trading:
            return UniverseSnapshot(as_of=date, symbols=(), n_eligible=0)

        # Strictly before: `.loc[:date]` would include the selection date.
        history = volumes.index < date
        window = volumes.loc[history].tail(lookback)
        price_window = closes.loc[history].tail(lookback)
        if window.empty:
            return UniverseSnapshot(as_of=date, symbols=(), n_eligible=len(trading))

        candidates = [s for s in trading if s in window.columns]
        notional = (price_window[candidates] * window[candidates]).astype(float)
        coverage = notional.notna().mean()
        eligible = coverage[coverage >= min_coverage].index.tolist()
        if not eligible:
            return UniverseSnapshot(as_of=date, symbols=(), n_eligible=len(candidates))

        ranked = notional[eligible].mean().sort_values(ascending=False)
        chosen = tuple(ranked.head(top_n).index)

        # Attrition is only observable inside the data. Near the end of the
        # sample every name's last bar precedes "one year from now", so a naive
        # comparison reports 100% attrition on the final snapshot — an artefact
        # of the data stopping, not of companies failing. The horizon is
        # therefore capped at the last bar in the sample, less a few days so a
        # name that simply did not print on the final session is not counted as
        # dead.
        data_end = pd.Timestamp(volumes.index[-1]) - pd.Timedelta(days=ALIVE_TOLERANCE_DAYS)
        horizon = min(date + pd.Timedelta(days=365), data_end)
        later_delisted = sum(1 for s in chosen if self.listings[s].died_before(horizon))
        return UniverseSnapshot(
            as_of=date,
            symbols=chosen,
            n_eligible=len(eligible),
            n_later_delisted=later_delisted,
        )

    def survivorship_report(
        self,
        dates: Sequence[pd.Timestamp],
        *,
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
        top_n: int,
        **kwargs: object,
    ) -> pd.DataFrame:
        """How much a today-only universe would have been deleting, per date.

        The column that matters is ``n_later_delisted``. It is the count of
        names a survivorship-biased universe removes from history entirely,
        and it is the difference between the two constructions expressed as
        something countable rather than as a caveat.
        """
        rows = []
        for date in dates:
            snapshot = self.select_at(
                date, closes=closes, volumes=volumes, top_n=top_n, **kwargs  # type: ignore[arg-type]
            )
            rows.append(
                {
                    "as_of": snapshot.as_of,
                    "n_members": len(snapshot.symbols),
                    "n_eligible": snapshot.n_eligible,
                    "n_later_delisted": snapshot.n_later_delisted,
                    "attrition_rate": snapshot.attrition_rate,
                }
            )
        return pd.DataFrame(rows)

    def survivor_only(self, as_of: pd.Timestamp) -> PointInTimeUniverse:
        """The universe a today-only construction would have produced.

        Not a utility — an experimental control. Running the same strategy over
        this and over the full universe isolates survivorship bias as a
        measured quantity rather than an argued one, which is the only way to
        say how much the earlier campaigns were flattered.
        """
        as_of = pd.Timestamp(as_of)
        return PointInTimeUniverse(
            listings={
                symbol: listing
                for symbol, listing in self.listings.items()
                if not listing.died_before(as_of)
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": listing.symbol,
                    "first_bar": listing.first_bar,
                    "last_bar": listing.last_bar,
                    "n_bars": listing.n_bars,
                    "exchange": listing.exchange,
                    "name": listing.name,
                    "is_active": listing.is_active,
                }
                for listing in sorted(self.listings.values(), key=lambda item: item.symbol)
            ]
        )

    def save(self, path: str) -> None:
        self.to_frame().to_csv(path, index=False)

    @classmethod
    def load(cls, path: str) -> PointInTimeUniverse:
        frame = pd.read_csv(path, parse_dates=["first_bar", "last_bar"])
        universe = cls()
        for row in frame.itertuples(index=False):
            universe.add(
                Listing(
                    symbol=str(row.symbol),
                    first_bar=pd.Timestamp(row.first_bar),
                    last_bar=pd.Timestamp(row.last_bar),
                    n_bars=int(row.n_bars),
                    exchange=str(row.exchange) if not pd.isna(row.exchange) else "",
                    name=str(row.name) if not pd.isna(row.name) else "",
                    is_active=bool(row.is_active),
                )
            )
        return universe

    @classmethod
    def from_frames(
        cls,
        closes: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
        *,
        active: Iterable[str] = (),
        exchanges: dict[str, str] | None = None,
    ) -> PointInTimeUniverse:
        """Derive listings from observed bars.

        The bars are authoritative, not the metadata. A name whose asset record
        says "active" but which last printed three years ago is delisted for
        every purpose a backtest has, and taking the venue's word for it would
        put a name in the universe that cannot be traded.
        """
        active_set = set(active)
        universe = cls()
        for symbol in closes.columns:
            column = closes[symbol].dropna()
            if column.empty:
                continue
            universe.add(
                Listing(
                    symbol=str(symbol),
                    first_bar=pd.Timestamp(column.index[0]),
                    last_bar=pd.Timestamp(column.index[-1]),
                    n_bars=int(column.size),
                    exchange=(exchanges or {}).get(str(symbol), ""),
                    is_active=str(symbol) in active_set,
                )
            )
        return universe


def listed_mask(
    universe: PointInTimeUniverse, index: pd.DatetimeIndex, symbols: Sequence[str]
) -> np.ndarray:
    """``(n_bars, n_symbols)`` boolean: was each name trading at each bar?

    The array a backtester needs in order to refuse to hold a name that has not
    listed or has already died. Computed from the listing intervals rather than
    from ``notna`` on the price matrix, so a mid-life data gap — a halt, a
    missing vendor day — does not read as a delisting and force a spurious
    liquidation.
    """
    mask = np.zeros((len(index), len(symbols)), dtype=bool)
    values = index.values
    for column, symbol in enumerate(symbols):
        listing = universe.listings.get(symbol)
        if listing is None:
            continue
        mask[:, column] = (values >= listing.first_bar.to_datetime64()) & (
            values <= listing.last_bar.to_datetime64()
        )
    return mask

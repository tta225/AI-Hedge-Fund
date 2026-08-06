"""Options structures, pricing and greeks.

The platform has **no options data feed**. This package defines structures and
prices them from a model volatility, which supports scenario analysis and
structure comparison. It cannot backtest an options strategy, and anything that
would require real quotes raises `ChainRequiredError` rather than inventing a number.
See `axiom.options.catalogue`.
"""

from axiom.options.catalogue import (
    CATALOGUE,
    ChainRequiredError,
    Direction,
    Leg,
    OptionStrategy,
    StructureSpec,
    VolatilityView,
    build,
    by_outlook,
    catalogue_text,
    undefined_risk,
)
from axiom.options.pricing import (
    Greeks,
    OptionType,
    black_scholes,
    greeks,
    implied_volatility,
    realised_volatility,
)

__all__ = [
    "CATALOGUE", "ChainRequiredError", "Direction", "Greeks", "Leg", "OptionStrategy",
    "OptionType", "StructureSpec", "VolatilityView", "black_scholes", "build",
    "by_outlook", "catalogue_text", "greeks", "implied_volatility",
    "realised_volatility", "undefined_risk",
]

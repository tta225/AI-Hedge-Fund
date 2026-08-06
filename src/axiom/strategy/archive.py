"""The strategy archive — one catalogue every consumer reads.

Why a registry and not just a dict
----------------------------------
The CLI, the web API, the research lab and the agent pipeline all need to answer
the same questions: what strategies exist, what family does each belong to, what
does it claim, what parameters can be swept, and which are safe to combine. Four
hardcoded lists is four chances for them to disagree, and a strategy the agents
can propose but the backtester cannot run is worse than one that does not exist.

So: one `ARCHIVE`, and every surface reads from it.

What an entry carries
---------------------
`StrategyEntry` records more than a constructor. It carries the **family**
(so a search does not mistake ten trend rules for a diverse candidate set), the
**thesis** (the mechanism claimed — an agent that cannot state one is proposing
noise), the **grid** (the parameter ranges honest to sweep), and the
**evidence** status.

`evidence` defaults to `UNTESTED` for everything, and that is not modesty. Not
one strategy here has been validated on real market data by this platform, and
`docs/REAL_DATA_FINDINGS.md` records that the first real measurement found *no*
edge in the ICT features. A registry that let a strategy call itself
"validated" without a citation would be the most dangerous file in the
repository.

Combining with ICT
------------------
`regime_gated` and `confluence_gated` wrap any entry: the ICT engine finds the
setup, the HMM or the confluence scorer decides whether the current state is one
where it has any business working. That composition is the point of keeping ICT
and quant strategies in a single archive rather than two.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from axiom.strategy.base import Strategy
from axiom.strategy.ict_strategies import LiquidityRaidReversal, SilverBulletStrategy
from axiom.strategy.institutional import (
    DonchianBreakout,
    DualMovingAverage,
    KalmanTrend,
    KeltnerBreakout,
    MACDTrend,
    OpeningRangeBreakout,
    OrnsteinUhlenbeck,
    RSIReversal,
    TurnOfMonth,
    VolatilityTargetedMomentum,
    VWAPReversion,
)
from axiom.strategy.price_action import (
    DoubleTopBottom,
    EngulfingReversal,
    InsideBarBreakout,
    LevelRetest,
    PinBarRejection,
    RoundNumberReversion,
    ThreeBarReversal,
    WyckoffSpring,
)
from axiom.strategy.quant_strategies import (
    MeanReversionZScore,
    TimeSeriesMomentum,
    VolatilityBreakout,
)
from axiom.strategy.seasonality import (
    DayOfWeek,
    FOMCDrift,
    JanuaryEffect,
    OptionExpiryWeek,
    SantaClausRally,
    SellInMay,
)
from axiom.strategy.statistical import (
    AmihudIlliquidity,
    AutocorrelationReversal,
    FiftyTwoWeekHigh,
    GapFade,
    HurstAdaptive,
    NarrowRangeBreakout,
    OrderFlowImbalance,
    PreviousDayLevels,
    SkewnessReversal,
    VarianceRatio,
)
from axiom.strategy.technical import (
    ADXTrend,
    BollingerSqueeze,
    CCIReversal,
    IchimokuCloud,
    OBVDivergence,
    ParabolicSAR,
    StochasticReversal,
    Supertrend,
    WilliamsR,
)


class Family(str, Enum):
    """What a strategy structurally *is*.

    Used by the research lab to avoid the classic error of searching fifty
    parameterisations of one idea and reporting the best as a discovery. A
    candidate set spanning one family says nothing about robustness.
    """

    TREND = "trend"
    MEAN_REVERSION = "mean-reversion"
    BREAKOUT = "breakout"
    ICT_STRUCTURAL = "ict-structural"
    STAT_ARB = "stat-arb"
    MICROSTRUCTURE = "microstructure"
    SEASONALITY = "seasonality"
    #: Discretionary chart/candle patterns encoded as explicit rules. Kept
    #: separate because the encoding is a weaker claim than the discretionary
    #: practice — see `axiom.strategy.price_action`.
    PRICE_ACTION = "price-action"
    #: Rules that measure a property (persistence, autocorrelation) and switch
    #: behaviour on the estimate rather than assuming one.
    ADAPTIVE = "adaptive"


class Evidence(str, Enum):
    """How much real-data support a strategy has *in this repository*."""

    #: Implemented and unit-tested; never run against real market history here.
    UNTESTED = "untested"
    #: Run on real bars, result recorded, no significant edge found.
    MEASURED_NO_EDGE = "measured-no-edge"
    #: Run on real bars with a control, edge survived out-of-sample.
    MEASURED_EDGE = "measured-edge"
    #: Present as a deliberate null — expected to fail, kept to check the harness.
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class StrategyEntry:
    """One catalogued strategy."""

    key: str
    factory: Callable[..., Strategy]
    family: Family
    #: The mechanism claimed, in one sentence. Not a performance claim.
    thesis: str
    #: Where the idea comes from, so a reader can check it.
    source: str
    #: Parameter ranges honest to sweep. Keys must be constructor arguments.
    grid: dict[str, Sequence[Any]] = field(default_factory=dict)
    evidence: Evidence = Evidence.UNTESTED
    #: True when the strategy reads ICT state — a meaningful cost on long runs.
    uses_ict: bool = False
    tags: tuple[str, ...] = ()
    #: Set when a strategy should not be traded on its own.
    caveat: str = ""

    def build(self, **overrides: Any) -> Strategy:
        return self.factory(**overrides)

    def describe(self) -> str:
        lines = [
            f"{self.key}  [{self.family.value}]",
            f"  thesis   : {self.thesis}",
            f"  source   : {self.source}",
            f"  evidence : {self.evidence.value}",
        ]
        if self.grid:
            params = ", ".join(f"{k}={list(v)}" for k, v in self.grid.items())
            lines.append(f"  grid     : {params}")
        if self.caveat:
            lines.append(f"  caveat   : {self.caveat}")
        return "\n".join(lines)


#: Every strategy the platform can run, keyed by CLI/API name.
ARCHIVE: dict[str, StrategyEntry] = {
    # ── ICT structural ──────────────────────────────────────────────────────
    "silver-bullet": StrategyEntry(
        key="silver-bullet",
        factory=SilverBulletStrategy,
        family=Family.ICT_STRUCTURAL,
        thesis=(
            "Within a narrow post-open window, a market structure shift into a "
            "discount array marks the algorithm's intended delivery direction."
        ),
        source="ICT — Silver Bullet model",
        grid={
            "min_confluence": (0.4, 0.5, 0.6, 0.7),
            "min_rr": (1.5, 2.0, 2.5),
            "mss_lookback": (8, 12, 20),
        },
        uses_ict=True,
        tags=("ict", "session", "killzone"),
    ),
    "liquidity-raid": StrategyEntry(
        key="liquidity-raid",
        factory=LiquidityRaidReversal,
        family=Family.ICT_STRUCTURAL,
        thesis=(
            "A sweep beyond a liquidity pool that closes back inside took stops "
            "rather than breaking out, and price reverses away from the pool."
        ),
        source="ICT — liquidity raid / stop hunt",
        grid={
            "min_confluence": (0.35, 0.45, 0.55),
            "min_rr": (1.5, 2.0, 3.0),
            "sweep_lookback": (12, 24, 36),
        },
        evidence=Evidence.MEASURED_NO_EDGE,
        uses_ict=True,
        tags=("ict", "liquidity", "reversal"),
        caveat=(
            "docs/REAL_DATA_FINDINGS.md measured sweep reversal against a "
            "same-bar control on ~38,700 real bars and found it negative in all "
            "four datasets — price continued in the sweep's direction more often "
            "than it reversed. This strategy trades the reversal reading."
        ),
    ),
    # ── trend ───────────────────────────────────────────────────────────────
    "momentum": StrategyEntry(
        key="momentum",
        factory=TimeSeriesMomentum,
        family=Family.TREND,
        thesis="Trailing return over a lookback predicts the sign of the next move.",
        source="Jegadeesh & Titman (1993); Moskowitz, Ooi & Pedersen (2012)",
        grid={
            "lookback": (48, 96, 192, 384),
            "stop_atr": (1.5, 2.0, 3.0),
            "min_strength": (0.3, 0.5, 0.8),
        },
        tags=("momentum", "baseline"),
    ),
    "vol-momentum": StrategyEntry(
        key="vol-momentum",
        factory=VolatilityTargetedMomentum,
        family=Family.TREND,
        thesis=(
            "Momentum normalised by realised volatility, so the signal means the "
            "same thing across instruments and volatility regimes."
        ),
        source="Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum'",
        grid={
            "lookback": (60, 120, 250),
            "vol_window": (30, 60, 120),
            "min_signal": (0.3, 0.4, 0.6),
        },
        tags=("momentum", "tsmom", "vol-targeted"),
    ),
    "donchian": StrategyEntry(
        key="donchian",
        factory=DonchianBreakout,
        family=Family.BREAKOUT,
        thesis="Price breaking an N-bar extreme continues in that direction.",
        source="Donchian; Dennis & Eckhardt 'Turtle' rules",
        grid={
            "lookback": (20, 55, 100),
            "stop_atr": (1.5, 2.0, 3.0),
            "target_atr": (4.0, 6.0, 8.0),
        },
        tags=("trend", "managed-futures", "turtle"),
    ),
    "dual-ma": StrategyEntry(
        key="dual-ma",
        factory=DualMovingAverage,
        family=Family.TREND,
        thesis="A fast average crossing a slow one marks a change in trend regime.",
        source="Classical CTA trend following (AHL, Winton, Dunn)",
        grid={"fast": (10, 20, 50), "slow": (50, 100, 200)},
        tags=("trend", "crossover", "cta"),
    ),
    "kalman": StrategyEntry(
        key="kalman",
        factory=KalmanTrend,
        family=Family.TREND,
        thesis=(
            "A local-linear-trend filter estimates the trend adaptively, with no "
            "fixed lookback to overfit."
        ),
        source="Kalman (1960); state-space trend estimation",
        grid={
            "process_var": (1e-5, 1e-4, 1e-3),
            "min_slope_atr": (0.02, 0.05, 0.1),
        },
        tags=("trend", "adaptive", "state-space"),
    ),
    "macd": StrategyEntry(
        key="macd",
        factory=MACDTrend,
        family=Family.TREND,
        thesis="Convergence/divergence of two EMAs marks momentum turning points.",
        source="Appel (1979)",
        grid={"fast": (8, 12, 20), "slow": (21, 26, 50), "signal": (5, 9, 14)},
        tags=("trend", "control"),
        caveat=(
            "Kept mainly as a control. A new idea that cannot beat MACD net of "
            "costs is not novel in any way that matters."
        ),
    ),
    # ── breakout / volatility ───────────────────────────────────────────────
    "vol-breakout": StrategyEntry(
        key="vol-breakout",
        factory=VolatilityBreakout,
        family=Family.BREAKOUT,
        thesis="Expansion out of a compressed range continues in the break direction.",
        source="Volatility-contraction literature; structurally near ICT displacement",
        grid={"squeeze_lookback": (20, 40, 60), "breakout_lookback": (5, 10, 20)},
        tags=("breakout", "volatility"),
    ),
    "keltner": StrategyEntry(
        key="keltner",
        factory=KeltnerBreakout,
        family=Family.BREAKOUT,
        thesis="A close outside an ATR channel around an EMA signals trend initiation.",
        source="Keltner (1960); Chester Keltner channel",
        grid={"span": (10, 20, 40), "width_atr": (1.5, 2.0, 3.0)},
        tags=("breakout", "channel"),
    ),
    "opening-range": StrategyEntry(
        key="opening-range",
        factory=OpeningRangeBreakout,
        family=Family.BREAKOUT,
        thesis=(
            "The range established in the first minutes of a session frames the "
            "day; a break of it sets the day's direction."
        ),
        source="Crabel (1990); institutional intraday ORB",
        grid={"range_minutes": (15, 30, 60), "stop_atr": (1.0, 1.5, 2.0)},
        tags=("breakout", "session", "intraday"),
    ),
    # ── mean reversion / stat arb ───────────────────────────────────────────
    "mean-reversion": StrategyEntry(
        key="mean-reversion",
        factory=MeanReversionZScore,
        family=Family.MEAN_REVERSION,
        thesis="Stretched deviations from a moving anchor revert.",
        source="Classical statistical arbitrage",
        grid={"lookback": (20, 50, 100), "entry_z": (1.5, 2.0, 2.5)},
        tags=("mean-reversion", "baseline"),
    ),
    "ou-reversion": StrategyEntry(
        key="ou-reversion",
        factory=OrnsteinUhlenbeck,
        family=Family.STAT_ARB,
        thesis=(
            "Fit an Ornstein-Uhlenbeck process and trade only when the estimated "
            "half-life says the series genuinely reverts on a tradeable horizon."
        ),
        source="Ornstein-Uhlenbeck (1930); Avellaneda & Lee (2010)",
        grid={
            "window": (60, 120, 250),
            "entry_sigma": (1.5, 2.0, 2.5),
            "max_half_life": (30.0, 60.0, 120.0),
        },
        tags=("mean-reversion", "stat-arb", "fitted"),
    ),
    "rsi-reversal": StrategyEntry(
        key="rsi-reversal",
        factory=RSIReversal,
        family=Family.MEAN_REVERSION,
        thesis=(
            "Very short-horizon oversold readings revert, when taken only in the "
            "direction of the longer trend."
        ),
        source="Connors & Alvarez (2008), 'Short Term Trading Strategies That Work'",
        grid={
            "period": (2, 3, 4),
            "oversold": (5.0, 10.0, 20.0),
            "trend_filter": (100, 200),
        },
        tags=("mean-reversion", "rsi", "short-horizon"),
    ),
    "vwap-reversion": StrategyEntry(
        key="vwap-reversion",
        factory=VWAPReversion,
        family=Family.MICROSTRUCTURE,
        thesis=(
            "Institutional execution is benchmarked to VWAP, so extension from it "
            "meets real mandated flow in the opposite direction."
        ),
        source="Execution-benchmark microstructure; Berkowitz, Logue & Noser (1988)",
        grid={"entry_sigma": (1.5, 2.0, 2.5), "min_bars": (10, 20, 40)},
        tags=("mean-reversion", "vwap", "microstructure"),
    ),
    # ── controls ────────────────────────────────────────────────────────────
    "turn-of-month": StrategyEntry(
        key="turn-of-month",
        factory=TurnOfMonth,
        family=Family.SEASONALITY,
        thesis="Pension and payroll flows cluster around the month boundary.",
        source="Ariel (1987); Lakonishok & Smidt (1988)",
        grid={"days_before": (1, 2, 3), "days_after": (2, 3, 5)},
        evidence=Evidence.CONTROL,
        tags=("seasonality", "calendar", "control"),
        caveat=(
            "A calendar rule has no mechanism that reacts to price. If it clears "
            "the same walk-forward and deflated-Sharpe bar as everything else, "
            "that is evidence the bar is too low — not a discovery."
        ),
    ),
}


# ── technical indicators (widely published; useful mainly as controls) ──────
ARCHIVE.update({
    "supertrend": StrategyEntry(
        key="supertrend", factory=Supertrend, family=Family.TREND,
        thesis="An ATR band that ratchets with the trend; a close through it marks a flip.",
        source="Olivier Seban, SuperTrend indicator",
        grid={"period": (7, 10, 14), "multiplier": (2.0, 3.0, 4.0)},
        tags=("trend", "atr-band", "control"),
        caveat="Heavily correlated with keltner and donchian. Not an independent confirmation of either.",
    ),
    "parabolic-sar": StrategyEntry(
        key="parabolic-sar", factory=ParabolicSAR, family=Family.TREND,
        thesis="An accelerating trailing stop; the flip marks a trend change.",
        source="Wilder (1978), 'New Concepts in Technical Trading Systems'",
        grid={"step": (0.01, 0.02, 0.04), "max_step": (0.1, 0.2, 0.3)},
        tags=("trend", "psar", "control"),
        caveat="Designed as an exit mechanism. Using it as an entry is a repurposing the original does not support.",
    ),
    "adx-trend": StrategyEntry(
        key="adx-trend", factory=ADXTrend, family=Family.TREND,
        thesis="Take the directional-index cross only when ADX says a trend exists.",
        source="Wilder (1978), Directional Movement System",
        grid={"period": (10, 14, 20), "min_adx": (20.0, 25.0, 30.0)},
        tags=("trend", "adx", "filtered"),
    ),
    "ichimoku": StrategyEntry(
        key="ichimoku", factory=IchimokuCloud, family=Family.TREND,
        thesis="A break of the forward-displaced cloud, confirmed by the tenkan/kijun cross.",
        source="Goichi Hosoda, Ichimoku Kinko Hyo",
        grid={"conversion": (7, 9, 12), "base": (22, 26, 34)},
        tags=("trend", "ichimoku", "breakout"),
        caveat="The displaced cloud is a common source of lookahead bugs; this implementation reads the historical span deliberately.",
    ),
    "bollinger-squeeze": StrategyEntry(
        key="bollinger-squeeze", factory=BollingerSqueeze, family=Family.BREAKOUT,
        thesis="A contraction in realised volatility precedes an expansion.",
        source="Bollinger (1980s); Carr's squeeze formulation",
        grid={"window": (14, 20, 30), "squeeze_percentile": (10.0, 20.0, 30.0)},
        tags=("breakout", "volatility", "bollinger"),
    ),
    "stochastic": StrategyEntry(
        key="stochastic", factory=StochasticReversal, family=Family.MEAN_REVERSION,
        thesis="%K crossing %D out of an extreme marks short-horizon exhaustion.",
        source="George Lane, stochastic oscillator",
        grid={"k_period": (9, 14, 21), "oversold": (10.0, 20.0, 30.0)},
        tags=("mean-reversion", "oscillator", "control"),
    ),
    "cci": StrategyEntry(
        key="cci", factory=CCIReversal, family=Family.MEAN_REVERSION,
        thesis="Typical price reverting from an extreme, scaled by mean absolute deviation.",
        source="Donald Lambert (1980), Commodity Channel Index",
        grid={"period": (14, 20, 30), "threshold": (100.0, 150.0, 200.0)},
        tags=("mean-reversion", "cci"),
    ),
    "williams-r": StrategyEntry(
        key="williams-r", factory=WilliamsR, family=Family.MEAN_REVERSION,
        thesis="Recovery from an extreme of the high-low range, with the trend.",
        source="Larry Williams, %R",
        grid={"period": (10, 14, 21)},
        tags=("mean-reversion", "oscillator", "control"),
        caveat="Mathematically identical to the stochastic %K, shifted by 100. Agreement between the two is one observation.",
    ),
    "obv-divergence": StrategyEntry(
        key="obv-divergence", factory=OBVDivergence, family=Family.MICROSTRUCTURE,
        thesis="A price extreme unconfirmed by cumulative volume is unsupported by participation.",
        source="Joseph Granville (1963), On-Balance Volume",
        grid={"lookback": (20, 40, 60)},
        tags=("divergence", "volume"),
        caveat="Meaningless on feeds with unreliable volume — FX, and any single-venue equity feed such as IEX.",
    ),
})

# ── statistical / econometric ──────────────────────────────────────────────
ARCHIVE.update({
    "hurst-adaptive": StrategyEntry(
        key="hurst-adaptive", factory=HurstAdaptive, family=Family.ADAPTIVE,
        thesis="Estimate persistence and apply the rule the estimate supports; trade nothing near a random walk.",
        source="Hurst (1951); Mandelbrot rescaled range",
        grid={"window": (120, 200, 400), "band": (0.05, 0.08, 0.12)},
        tags=("adaptive", "hurst", "regime"),
    ),
    "variance-ratio": StrategyEntry(
        key="variance-ratio", factory=VarianceRatio, family=Family.ADAPTIVE,
        thesis="The variance ratio says whether returns are trending or reverting; trade accordingly.",
        source="Lo & MacKinlay (1988)",
        grid={"window": (150, 250, 500), "period": (3, 5, 10)},
        tags=("adaptive", "econometric"),
        caveat="Measures the same property as hurst-adaptive with a different estimator. Agreement is one observation, disagreement is informative.",
    ),
    "order-flow": StrategyEntry(
        key="order-flow", factory=OrderFlowImbalance, family=Family.MICROSTRUCTURE,
        thesis="Net buying pressure inferred from where bars close within their range predicts short-horizon direction.",
        source="Cont, Kukanov & Stoikov (2014), order flow imbalance",
        grid={"window": (10, 20, 40), "threshold": (0.25, 0.35, 0.5)},
        tags=("microstructure", "order-flow", "volume"),
        caveat="A bar-level PROXY. Real OFI is built from order-book messages; the literature's results are for the real thing and do not transfer unmeasured.",
    ),
    "nr7": StrategyEntry(
        key="nr7", factory=NarrowRangeBreakout, family=Family.BREAKOUT,
        thesis="The narrowest bar in N marks a contraction that resolves with expansion.",
        source="Toby Crabel (1990), 'Day Trading with Short Term Price Patterns'",
        grid={"lookback": (4, 7, 10)},
        tags=("breakout", "volatility", "crabel"),
    ),
    "gap-fade": StrategyEntry(
        key="gap-fade", factory=GapFade, family=Family.MEAN_REVERSION,
        thesis="Ordinary opening gaps fill as real liquidity arrives after the open.",
        source="Gap-fill literature; opening auction microstructure",
        grid={"min_gap_atr": (0.3, 0.5, 0.8), "max_gap_atr": (2.0, 3.0, 5.0)},
        tags=("mean-reversion", "gap", "session"),
        caveat="The upper bound matters: news gaps keep going, and fading them produces this strategy's worst losses.",
    ),
    "prev-day-levels": StrategyEntry(
        key="prev-day-levels", factory=PreviousDayLevels, family=Family.BREAKOUT,
        thesis="Resting orders cluster at the prior session's extremes; the break continues.",
        source="Intraday level trading; the quantitative analogue of ICT liquidity pools",
        grid={"buffer_atr": (0.05, 0.1, 0.2)},
        tags=("breakout", "session", "liquidity"),
        caveat="Trades the CONTINUATION reading of a level break. liquidity-raid trades the reversal reading of the same event; running both measures which the data supports.",
    ),
    "52-week-high": StrategyEntry(
        key="52-week-high", factory=FiftyTwoWeekHigh, family=Family.TREND,
        thesis="Nearness to the trailing extreme predicts returns better than raw momentum — an anchoring effect.",
        source="George & Hwang (2004), 'The 52-Week High and Momentum Investing'",
        grid={"window": (120, 252, 500), "proximity": (0.95, 0.98, 0.99)},
        tags=("momentum", "anchoring"),
        caveat="`window` is in BARS. 252 is a year on daily bars and four days on 15m bars; setting it without regard to timeframe misapplies the effect.",
    ),
    "autocorr-reversal": StrategyEntry(
        key="autocorr-reversal", factory=AutocorrelationReversal, family=Family.STAT_ARB,
        thesis="Fade the last move only when lag-1 autocorrelation is measurably negative.",
        source="Short-horizon reversal literature; Lo & MacKinlay",
        grid={"window": (60, 120, 250), "max_autocorr": (-0.02, -0.05, -0.1)},
        tags=("mean-reversion", "econometric", "tested-premise"),
    ),
    "skewness": StrategyEntry(
        key="skewness", factory=SkewnessReversal, family=Family.STAT_ARB,
        thesis="Positively-skewed assets are overpriced by investors who pay for lottery payoffs.",
        source="Boyer, Mitton & Vorkink (2010); Bali, Cakici & Whitelaw (2011)",
        grid={"window": (60, 120, 250), "threshold": (0.3, 0.5, 1.0)},
        tags=("factor", "skewness"),
        caveat="The original result is CROSS-SECTIONAL. Applied here as a time-series filter, which is a weaker and different claim.",
    ),
    "amihud": StrategyEntry(
        key="amihud", factory=AmihudIlliquidity, family=Family.MICROSTRUCTURE,
        thesis="Price moving on thin volume is unsupported and tends to retrace.",
        source="Amihud (2002), 'Illiquidity and stock returns'",
        grid={"window": (30, 60, 120), "spike_percentile": (85.0, 90.0, 95.0)},
        tags=("mean-reversion", "liquidity"),
        caveat="Bar-level volume. Inherits every caveat about feed quality.",
    ),
})

# ── discretionary price action ─────────────────────────────────────────────
ARCHIVE.update({
    "pin-bar": StrategyEntry(
        key="pin-bar", factory=PinBarRejection, family=Family.PRICE_ACTION,
        thesis="A long wick shows price was pushed to a level and rejected before the close.",
        source="Price-action tradition; equivalent to ICT rejection blocks",
        grid={"wick_ratio": (0.5, 0.6, 0.7), "trend_filter": (20, 50, 100)},
        tags=("price-action", "rejection"),
        caveat="Same object as an ICT rejection block. If both fire, that is one observation.",
    ),
    "engulfing": StrategyEntry(
        key="engulfing", factory=EngulfingReversal, family=Family.PRICE_ACTION,
        thesis="A bar that fully covers the previous body at a swing extreme marks a momentum shift.",
        source="Candlestick tradition (Nison, 'Japanese Candlestick Charting')",
        grid={"extreme_lookback": (10, 20, 40), "min_body_ratio": (1.0, 1.2, 1.5)},
        tags=("price-action", "reversal"),
    ),
    "inside-bar": StrategyEntry(
        key="inside-bar", factory=InsideBarBreakout, family=Family.PRICE_ACTION,
        thesis="A bar contained within its predecessor is a one-bar consolidation that resolves on the break.",
        source="Price-action tradition; Crabel's contraction patterns",
        grid={"trend_filter": (20, 50, 100)},
        tags=("price-action", "breakout"),
        caveat="Correlated with nr7 and bollinger-squeeze — all three are contraction-expansion rules.",
    ),
    "wyckoff-spring": StrategyEntry(
        key="wyckoff-spring", factory=WyckoffSpring, family=Family.PRICE_ACTION,
        thesis="A shallow break of a range boundary that reclaims signals exhausted supply beyond it.",
        source="Richard Wyckoff; identical to ICT turtle soup",
        grid={"range_lookback": (20, 40, 80), "max_penetration_atr": (0.5, 1.0, 1.5)},
        tags=("price-action", "false-break", "wyckoff"),
        caveat="The same pattern as ICT turtle soup and a liquidity sweep. REAL_DATA_FINDINGS.md measured the ICT version and found the reversal reading negative on real bars.",
    ),
    "double-top-bottom": StrategyEntry(
        key="double-top-bottom", factory=DoubleTopBottom, family=Family.PRICE_ACTION,
        thesis="Two failures at one level, then a break of the neckline between them.",
        source="Classical chart-pattern analysis (Edwards & Magee)",
        grid={"lookback": (40, 60, 100), "tolerance_atr": (0.3, 0.5, 0.8)},
        tags=("price-action", "reversal", "pattern"),
    ),
    "three-bar-reversal": StrategyEntry(
        key="three-bar-reversal", factory=ThreeBarReversal, family=Family.PRICE_ACTION,
        thesis="Three consecutive bars against the trend, then a reclaim, marks short-horizon exhaustion.",
        source="Price-action tradition ('three pushes')",
        grid={"trend_filter": (50, 100, 200)},
        tags=("price-action", "exhaustion"),
    ),
    "level-retest": StrategyEntry(
        key="level-retest", factory=LevelRetest, family=Family.PRICE_ACTION,
        thesis="A broken level holds from the other side — polarity flip.",
        source="Support/resistance tradition; equivalent to ICT breaker blocks",
        grid={"swing_lookback": (20, 30, 50), "proximity_atr": (0.3, 0.4, 0.6)},
        tags=("price-action", "retest", "polarity"),
        caveat="Same object as an ICT breaker block.",
    ),
    "round-number": StrategyEntry(
        key="round-number", factory=RoundNumberReversion, family=Family.MICROSTRUCTURE,
        thesis="Limit orders cluster at round prices, so price decelerates into them.",
        source="FX microstructure (Osler, 2003)",
        grid={"proximity_atr": (0.15, 0.25, 0.4)},
        tags=("microstructure", "order-clustering"),
    ),
})

# ── calendar effects. Every one is a CONTROL. ──────────────────────────────
ARCHIVE.update({
    "january-effect": StrategyEntry(
        key="january-effect", factory=JanuaryEffect, family=Family.SEASONALITY,
        thesis="Tax-loss selling in December reverses in January, lifting small caps.",
        source="Rozeff & Kinney (1976); Keim (1983)",
        grid={"days": (5, 10, 15)}, evidence=Evidence.CONTROL,
        tags=("seasonality", "control"),
        caveat="The effect is a SIZE premium. Applied to a large-cap index it is at best diluted.",
    ),
    "santa-rally": StrategyEntry(
        key="santa-rally", factory=SantaClausRally, family=Family.SEASONALITY,
        thesis="Equities drift up over the last December sessions into early January.",
        source="Yale Hirsch (1972), Stock Trader's Almanac",
        evidence=Evidence.CONTROL, tags=("seasonality", "control"),
        caveat="Approximated on calendar dates; the exchange holiday calendar is not modelled.",
    ),
    "sell-in-may": StrategyEntry(
        key="sell-in-may", factory=SellInMay, family=Family.SEASONALITY,
        thesis="November-April outperforms May-October — the Halloween indicator.",
        source="Bouman & Jacobsen (2002)",
        evidence=Evidence.CONTROL, tags=("seasonality", "control"),
    ),
    "day-of-week": StrategyEntry(
        key="day-of-week", factory=DayOfWeek, family=Family.SEASONALITY,
        thesis="A nominated weekday carries a directional bias.",
        source="French (1980), weekend effect",
        grid={"weekday": (0, 1, 2, 3, 4), "direction": (1, -1)},
        evidence=Evidence.CONTROL, tags=("seasonality", "control", "null"),
        caveat="The archive's purest null. Ten combinations guarantee one looks good in any finite sample. A search that ranks this highly is broken.",
    ),
    "opex-week": StrategyEntry(
        key="opex-week", factory=OptionExpiryWeek, family=Family.SEASONALITY,
        thesis="Dealer gamma hedging into a large expiry supports the underlying.",
        source="Option-expiration week effect (Stivers & Sun, 2010)",
        evidence=Evidence.CONTROL, tags=("seasonality", "gamma"),
        caveat="Unusually for this family, the mechanism is plausible — dealer hedging is a real flow.",
    ),
    "fomc-drift": StrategyEntry(
        key="fomc-drift", factory=FOMCDrift, family=Family.SEASONALITY,
        thesis="Equities drift up in the 24 hours before scheduled FOMC announcements.",
        source="Lucca & Moench (2015), 'The Pre-FOMC Announcement Drift'",
        evidence=Evidence.CONTROL, tags=("seasonality", "macro"),
        caveat="The FOMC dates here are APPROXIMATED, not loaded from the Fed calendar. This measures 'mid-month in eight months' — do not read a result as a test of the paper.",
    ),
})


# ── lookup helpers ─────────────────────────────────────────────────────────


def get(key: str) -> StrategyEntry:
    try:
        return ARCHIVE[key]
    except KeyError:
        raise KeyError(
            f"unknown strategy {key!r}. Available: {', '.join(sorted(ARCHIVE))}"
        ) from None


def build(key: str, **overrides: Any) -> Strategy:
    """Instantiate a catalogued strategy by name."""
    return get(key).build(**overrides)


def by_family(family: Family) -> list[StrategyEntry]:
    return [e for e in ARCHIVE.values() if e.family is family]


def by_tag(tag: str) -> list[StrategyEntry]:
    return [e for e in ARCHIVE.values() if tag in e.tags]


def families() -> dict[Family, list[str]]:
    out: dict[Family, list[str]] = {}
    for entry in ARCHIVE.values():
        out.setdefault(entry.family, []).append(entry.key)
    return {k: sorted(v) for k, v in sorted(out.items(), key=lambda kv: kv[0].value)}


def diversified(limit_per_family: int = 1, *, exclude_controls: bool = True) -> list[StrategyEntry]:
    """A candidate set spanning families rather than parameterisations.

    This is what a strategy search should be handed. Ten trend rules produce a
    best-of-ten that is almost entirely selection noise; one rule from each of
    ten families produces a comparison that means something.

    Controls are excluded by default — they are meant to fail, and including
    them in a search inflates the dispersion the deflated Sharpe threshold is
    computed from.
    """
    out: list[StrategyEntry] = []
    seen: dict[Family, int] = {}
    for entry in sorted(ARCHIVE.values(), key=lambda e: e.key):
        if exclude_controls and entry.evidence is Evidence.CONTROL:
            continue
        count = seen.get(entry.family, 0)
        if count >= limit_per_family:
            continue
        seen[entry.family] = count + 1
        out.append(entry)
    return out


def catalogue() -> str:
    """Human-readable listing, grouped by family."""
    blocks: list[str] = []
    for family, keys in families().items():
        blocks.append(f"\n{family.value.upper()}")
        blocks.append("─" * 68)
        for key in keys:
            blocks.append(ARCHIVE[key].describe())
            blocks.append("")
    return "\n".join(blocks)


def all_entries() -> list[StrategyEntry]:
    """Every catalogued strategy, in key order."""
    return [ARCHIVE[k] for k in sorted(ARCHIVE)]

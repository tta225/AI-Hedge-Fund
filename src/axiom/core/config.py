"""Runtime configuration, loaded from environment / ``.env``.

The safety-relevant settings live here rather than in call sites so there is
exactly one place to audit before anything touches a real venue.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from axiom.core.types import TradingMode

LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_THE_RISK"


class RiskSettings(BaseSettings):
    """Hard limits. Every one is enforced by :class:`axiom.risk.RiskManager`."""

    model_config = SettingsConfigDict(env_prefix="AXIOM_", env_file=".env", extra="ignore")

    account_equity: float = Field(100_000.0, gt=0)
    max_risk_per_trade_pct: float = Field(0.5, gt=0, le=100)
    daily_loss_limit_pct: float = Field(2.0, gt=0, le=100)
    max_gross_exposure_pct: float = Field(200.0, gt=0)
    max_positions: int = Field(5, ge=1)
    max_consecutive_losses: int = Field(4, ge=1)

    @model_validator(mode="after")
    def _coherent(self) -> RiskSettings:
        if self.max_risk_per_trade_pct > self.daily_loss_limit_pct:
            raise ValueError(
                f"max_risk_per_trade_pct ({self.max_risk_per_trade_pct}) exceeds "
                f"daily_loss_limit_pct ({self.daily_loss_limit_pct}): a single "
                "losing trade would breach the daily limit on entry"
            )
        return self

    @property
    def max_risk_per_trade_cash(self) -> float:
        return self.account_equity * self.max_risk_per_trade_pct / 100.0

    @property
    def daily_loss_limit_cash(self) -> float:
        return self.account_equity * self.daily_loss_limit_pct / 100.0


class ExecutionSettings(BaseSettings):
    """Cost assumptions. Applied to every backtest — never optional."""

    model_config = SettingsConfigDict(env_prefix="AXIOM_", env_file=".env", extra="ignore")

    commission_per_unit: float = Field(0.85, ge=0)
    slippage_ticks: float = Field(1.0, ge=0)


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    agent_model: str = Field("claude-opus-5", alias="AXIOM_AGENT_MODEL")

    @property
    def enabled(self) -> bool:
        return bool(self.anthropic_api_key)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AXIOM_", env_file=".env", extra="ignore")

    trading_mode: TradingMode = TradingMode.PAPER
    live_trading_confirmed: str = ""
    kill_switch: bool = False

    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)

    @model_validator(mode="after")
    def _guard_live(self) -> Settings:
        if (
            self.trading_mode is TradingMode.LIVE
            and self.live_trading_confirmed != LIVE_CONFIRMATION_PHRASE
        ):
            raise ValueError(
                "trading_mode=live requires AXIOM_LIVE_TRADING_CONFIRMED="
                f"{LIVE_CONFIRMATION_PHRASE}. Refusing to arm live trading."
            )
        return self

    @property
    def orders_permitted(self) -> bool:
        """False whenever the kill switch is engaged, regardless of mode."""
        return not self.kill_switch


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings — used by tests that patch the environment."""
    get_settings.cache_clear()

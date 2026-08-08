"""Databento provider. Offline — the SDK is stubbed.

Databento is the first source here that can supply futures, and futures bring
one problem no other provider in this package has: **contracts expire**. A
multi-year series is several contracts stitched together, and at each stitch the
price jumps for a reason that is not a return. Databento's continuous contracts
are unadjusted, so those jumps are present in the data.

That is the failure this module is arranged around. A 2% overnight move and a 2%
roll gap look identical in a price series — only the contract name distinguishes
them — and a backtest that counts the roll as a return is measuring the calendar.
`TestRollDetection` is the part that matters.

The other two are ordinary but expensive to get wrong: Databento bills by bytes
delivered, so an unpriced request is an invoice rather than a rate-limit error,
and the symbology has to resolve or nothing comes back at all.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from axiom.core.types import get_instrument
from axiom.data.base import BarRequest, ProviderError, ProviderUnavailableError
from axiom.data.databento import (
    DEFAULT_DATASET,
    DatabentoProvider,
    RollEvent,
    _detect_rolls,
    _explain,
    _resample,
)

ES = get_instrument("ES")
SPY = get_instrument("SPY")


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real key in the environment would turn "no credentials" tests into
    billed network calls. Databento charges per byte, so that is not merely a
    slow test — it is a charge on the user's account."""
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)


def _request(symbol: str = "ES", timeframe: str = "1m", days: int = 5) -> BarRequest:
    return BarRequest.lookback(get_instrument(symbol), timeframe, days)


def _bars(
    n: int = 6, start: float = 5000.0, symbols: list[str] | None = None
) -> pd.DataFrame:
    index = pd.date_range("2026-03-02 14:30", periods=n, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [start + i for i in range(n)],
            "high": [start + i + 2 for i in range(n)],
            "low": [start + i - 2 for i in range(n)],
            "close": [start + i + 1 for i in range(n)],
            "volume": [100 + i for i in range(n)],
        },
        index=index,
    )
    if symbols is not None:
        frame["symbol"] = symbols
    return frame


class _Store:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.to_df_kwargs: dict[str, Any] = {}

    def to_df(self, **kwargs: Any) -> pd.DataFrame:
        self.to_df_kwargs = kwargs
        return self.frame


class _StubProvider(DatabentoProvider):
    """Serves a canned frame instead of calling Databento."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        cost: float | Exception = 0.01,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("api_key", "db-test")
        super().__init__(**kwargs)
        self.store = _Store(frame if frame is not None else _bars())
        self.cost = cost
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        return True

    def _client(self) -> Any:
        provider = self

        class _Metadata:
            def get_cost(self, **kwargs: Any) -> float:
                if isinstance(provider.cost, Exception):
                    raise provider.cost
                return provider.cost

        class _Timeseries:
            def get_range(self, **kwargs: Any) -> _Store:
                provider.calls.append(kwargs)
                if provider.error is not None:
                    raise provider.error
                return provider.store

        class _Client:
            metadata = _Metadata()
            timeseries = _Timeseries()

        return _Client()


class TestAvailability:
    def test_no_key_is_unavailable(self) -> None:
        assert not DatabentoProvider().is_available()

    def test_the_missing_key_message_names_the_variable(self) -> None:
        with pytest.raises(ProviderUnavailableError, match="DATABENTO_API_KEY"):
            DatabentoProvider()._client()

    def test_an_unknown_roll_rule_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="unknown roll_rule"):
            DatabentoProvider(roll_rule="nonsense")


class TestSymbology:
    def test_a_bare_futures_root_becomes_a_continuous_contract(self) -> None:
        """'ES' names a family of contracts, not one of them. Passing it
        through unresolved returns nothing at all."""
        assert _StubProvider().symbol_for(_request("ES")) == ("ES.v.0", "continuous")

    def test_the_roll_rule_selects_the_suffix(self) -> None:
        provider = _StubProvider(roll_rule="calendar")
        assert provider.symbol_for(_request("ES"))[0] == "ES.c.0"
        assert _StubProvider(roll_rule="open_interest").symbol_for(
            _request("ES")
        )[0] == "ES.n.0"

    def test_an_explicit_contract_is_passed_through(self) -> None:
        """Someone who named a month meant that month."""
        assert _StubProvider().symbol_for(_request("ESH6")) == ("ESH6", "raw_symbol")

    def test_an_already_continuous_symbol_is_not_double_suffixed(self) -> None:
        symbol, _ = _StubProvider().symbol_for(_request("ES.v.0"))
        assert symbol == "ES.V.0"
        assert symbol.count(".v.0") == 0  # not ES.v.0.v.0

    def test_continuous_can_be_turned_off(self) -> None:
        assert _StubProvider(continuous=False).symbol_for(_request("ES")) == (
            "ES", "raw_symbol"
        )

    def test_equities_use_raw_symbols(self) -> None:
        assert _StubProvider().symbol_for(_request("SPY", "1d")) == ("SPY", "raw_symbol")


class TestDatasetAndSchema:
    def test_futures_go_to_cme_globex(self) -> None:
        assert _StubProvider().dataset_for(_request("ES")) == DEFAULT_DATASET

    def test_equities_do_not_go_to_the_futures_dataset(self) -> None:
        """GLBX.MDP3 carries no equities — a wrong dataset returns nothing,
        which reads as a symbol problem rather than a routing one."""
        assert _StubProvider().dataset_for(_request("SPY", "1d")) != DEFAULT_DATASET

    def test_an_explicit_dataset_overrides_the_default(self) -> None:
        assert _StubProvider(dataset="XCME.MBO").dataset_for(_request("ES")) == "XCME.MBO"

    def test_a_native_timeframe_maps_straight_across(self) -> None:
        assert _StubProvider().schema_for(_request("ES", "1m")) == ("ohlcv-1m", "")

    def test_a_non_native_timeframe_names_what_it_resamples_from(self) -> None:
        """Databento has no 15m schema. Building it from 1m costs 15× the bytes,
        which is worth knowing before requesting ten years of it."""
        assert _StubProvider().schema_for(_request("ES", "15m")) == ("ohlcv-1m", "1m")

    def test_an_unroutable_timeframe_is_an_error_not_a_guess(self) -> None:
        with pytest.raises(ProviderError, match="no route to"):
            _StubProvider().schema_for(_request("ES", "3m"))


class TestCost:
    def test_an_expensive_request_is_refused_before_it_is_made(self) -> None:
        """Databento bills by bytes delivered. The feedback for asking for too
        much is an invoice, not a 429."""
        provider = _StubProvider(cost=42.0, max_cost_usd=5.0)
        with pytest.raises(ProviderError, match=r"costs \$42"):
            provider.fetch_bars(_request())
        assert provider.calls == []  # nothing was fetched, so nothing was billed

    def test_a_cheap_request_goes_through(self) -> None:
        provider = _StubProvider(cost=0.02, max_cost_usd=5.0)
        provider.fetch_bars(_request())
        assert provider.last.cost_usd == pytest.approx(0.02)

    def test_the_ceiling_can_be_disabled_deliberately(self) -> None:
        provider = _StubProvider(cost=999.0, max_cost_usd=None)
        provider.fetch_bars(_request())
        assert provider.last.cost_usd is None

    def test_a_broken_pricer_stops_the_request_rather_than_billing_blind(self) -> None:
        """A caller who set a ceiling asked not to be billed above it. Fetching
        anyway because the *pricer* failed honours the letter of the request
        while breaking the only part of it that costs money."""
        provider = _StubProvider(cost=RuntimeError("pricing down"), max_cost_usd=5.0)
        with pytest.raises(ProviderError, match="could not price this request"):
            provider.fetch_bars(_request())
        assert provider.calls == []

    def test_the_override_is_named_in_the_pricing_failure(self) -> None:
        provider = _StubProvider(cost=RuntimeError("down"), max_cost_usd=5.0)
        with pytest.raises(ProviderError, match="max_cost_usd=None"):
            provider.fetch_bars(_request())


class TestRollDetection:
    def test_a_contract_change_is_recorded_as_a_roll(self) -> None:
        provider = _StubProvider(
            _bars(6, symbols=["ESH6"] * 3 + ["ESM6"] * 3)
        )
        provider.fetch_bars(_request())
        assert len(provider.last.rolls) == 1
        roll = provider.last.rolls[0]
        assert roll.from_contract == "ESH6"
        assert roll.to_contract == "ESM6"

    def test_the_gap_is_measured_across_the_stitch(self) -> None:
        frame = _bars(4, symbols=["ESH6", "ESH6", "ESM6", "ESM6"])
        # Shift the whole later contract up, rather than only its close — a bar
        # whose close sits outside its own high/low is rejected as corrupt long
        # before roll detection sees it, which is correct and would make this
        # test pass for the wrong reason.
        later = frame.index[2:]
        frame.loc[later, ["open", "high", "low", "close"]] += 98.0
        provider = _StubProvider(frame)
        provider.fetch_bars(_request())
        # close went 5002 → 5101 at the stitch: a 99-point gap that is not a move.
        assert provider.last.rolls[0].gap == pytest.approx(99.0)

    def test_the_first_bar_is_not_a_roll(self) -> None:
        """It starts a contract; it does not change one."""
        provider = _StubProvider(_bars(4, symbols=["ESH6"] * 4))
        provider.fetch_bars(_request())
        assert provider.last.rolls == []

    def test_several_rolls_are_all_reported(self) -> None:
        provider = _StubProvider(
            _bars(6, symbols=["ESH6", "ESH6", "ESM6", "ESM6", "ESU6", "ESU6"])
        )
        provider.fetch_bars(_request())
        assert [r.to_contract for r in provider.last.rolls] == ["ESM6", "ESU6"]

    def test_rolls_are_detected_from_the_contract_not_the_price(self) -> None:
        """A 2% overnight move and a 2% roll gap look identical in a price
        series. Only the symbol tells them apart, so only the symbol is used."""
        jumpy = _bars(4, symbols=["ESH6"] * 4)
        jumpy.loc[jumpy.index[2], "close"] = 5500.0  # a big move, same contract
        assert _detect_rolls(jumpy) == []

    def test_a_frame_without_symbols_reports_no_rolls_rather_than_guessing(self) -> None:
        assert _detect_rolls(_bars(4)) == []

    def test_the_contracts_seen_are_listed(self) -> None:
        provider = _StubProvider(_bars(4, symbols=["ESH6", "ESH6", "ESM6", "ESM6"]))
        provider.fetch_bars(_request())
        assert provider.last.contracts == ["ESH6", "ESM6"]

    def test_the_summary_says_a_roll_gap_is_not_a_return(self) -> None:
        """The number alone invites the wrong reading. The sentence is the
        point of reporting it at all."""
        provider = _StubProvider(_bars(4, symbols=["ESH6", "ESH6", "ESM6", "ESM6"]))
        provider.fetch_bars(_request())
        assert "NOT a return" in provider.last.render()

    def test_provenance_carries_the_roll_count(self) -> None:
        provider = _StubProvider(_bars(4, symbols=["ESH6", "ESH6", "ESM6", "ESM6"]))
        series = provider.fetch_bars(_request())
        assert "rolls=1" in series.provenance.detail


class TestFetch:
    def test_prices_are_requested_as_floats_not_raw_fixed_point(self) -> None:
        """DBN carries prices as int64 scaled by 1e-9. Getting this wrong makes
        ES read as 5,000,000,000,000 — and the conversion is the vendor's to do,
        not ours."""
        provider = _StubProvider()
        provider.fetch_bars(_request())
        assert provider.store.to_df_kwargs["price_type"] == "float"

    def test_symbols_are_mapped_so_rolls_can_be_seen_at_all(self) -> None:
        provider = _StubProvider()
        provider.fetch_bars(_request())
        assert provider.store.to_df_kwargs["map_symbols"] is True

    def test_the_series_is_real_not_synthetic(self) -> None:
        assert _StubProvider().fetch_bars(_request()).provenance.is_evidential

    def test_an_empty_response_explains_the_likely_cause(self) -> None:
        provider = _StubProvider(pd.DataFrame())
        with pytest.raises(ProviderError, match="continuous symbology"):
            provider.fetch_bars(_request())

    def test_a_non_ohlcv_schema_is_named_rather_than_crashing_downstream(self) -> None:
        frame = pd.DataFrame(
            {"price": [1.0], "size": [2]},
            index=pd.date_range("2026-03-02", periods=1, tz="UTC"),
        )
        with pytest.raises(ProviderError, match="missing"):
            _StubProvider(frame).fetch_bars(_request())

    def test_the_request_carries_the_resolved_symbology(self) -> None:
        provider = _StubProvider()
        provider.fetch_bars(_request("ES"))
        assert provider.calls[0]["symbols"] == "ES.v.0"
        assert provider.calls[0]["stype_in"] == "continuous"


class TestResampling:
    def test_bars_are_labelled_at_the_start_of_their_period(self) -> None:
        """A bar stamped 14:30 covers [14:30, 14:45)."""
        out = _resample(_bars(30), "15min")
        assert out.index[0] == pd.Timestamp("2026-03-02 14:30", tz="UTC")

    def test_the_resample_convention_matches_the_platform(self) -> None:
        """The convention is arbitrary; having two of them is not.

        If this adapter labelled bars differently from `OHLCVSeries.resample`,
        a 15m ES bar built here and a 15m SPY bar built there would be offset by
        a quarter of an hour while looking interchangeable — and every
        cross-instrument comparison would quietly compare different windows.
        This pins the two together so a change to one fails rather than drifts.
        """
        from axiom.core.provenance import Provenance
        from axiom.core.series import OHLCVSeries

        frame = _bars(30)
        via_platform = OHLCVSeries.from_dataframe(
            frame[["open", "high", "low", "close", "volume"]],
            ES, "1m", Provenance.real("test"),
        ).resample("15m")
        via_adapter = _resample(frame, "15min")

        assert list(via_adapter.index) == list(via_platform.df.index)
        pd.testing.assert_frame_equal(
            via_adapter[["open", "high", "low", "close", "volume"]],
            via_platform.df[["open", "high", "low", "close", "volume"]],
            check_freq=False,
            check_names=False,  # the platform names its index 'timestamp'
            # The platform casts to float64 on ingest; the adapter's frame is
            # cast by the same path afterwards. Only the values and labels are
            # what this test is about.
            check_dtype=False,
        )

    def test_aggregation_takes_the_extremes_not_the_endpoints(self) -> None:
        frame = _bars(30)
        out = _resample(frame, "15min")
        first = frame.iloc[:15]
        assert out["high"].iloc[0] == first["high"].max()
        assert out["low"].iloc[0] == first["low"].min()
        assert out["volume"].iloc[0] == first["volume"].sum()
        assert out["open"].iloc[0] == first["open"].iloc[0]
        assert out["close"].iloc[0] == first["close"].iloc[-1]

    def test_incomplete_periods_are_dropped_not_filled(self) -> None:
        out = _resample(_bars(3), "15min")
        assert not out[["open", "high", "low", "close"]].isna().any().any()

    def test_a_resampled_fetch_records_where_it_came_from(self) -> None:
        provider = _StubProvider(_bars(30))
        provider.fetch_bars(_request("ES", "15m"))
        assert provider.last.resampled_from == "1m"
        assert "resampled from 1m" in provider.last.render()


class TestErrorMessages:
    def test_a_401_points_at_the_key(self) -> None:
        assert "DATABENTO_API_KEY" in _explain(RuntimeError("HTTP 401 unauthorized"))

    def test_a_402_says_out_of_credit_rather_than_misconfigured(self) -> None:
        """Different fix entirely — checking the key would waste the afternoon."""
        assert "out of credit" in _explain(RuntimeError("HTTP 402 payment required"))

    def test_a_403_distinguishes_entitlement_from_a_blocked_host(self) -> None:
        message = _explain(RuntimeError("HTTP 403 forbidden"))
        assert "entitled" in message and "blocked" in message

    def test_a_symbol_error_names_continuous_symbology(self) -> None:
        assert "ES.v.0" in _explain(RuntimeError("HTTP 422 symbol not found"))

    def test_an_unrecognised_error_is_passed_through_unchanged(self) -> None:
        """Inventing a cause for an error we do not recognise is worse than
        showing it verbatim."""
        assert _explain(RuntimeError("something new")) == "something new"

    def test_a_vendor_exception_becomes_a_provider_error(self) -> None:
        provider = _StubProvider(error=RuntimeError("HTTP 401 unauthorized"))
        with pytest.raises(ProviderError, match="DATABENTO_API_KEY"):
            provider.fetch_bars(_request())


class TestRollEvent:
    def test_it_renders_both_contracts_and_the_gap(self) -> None:
        text = str(
            RollEvent(
                timestamp=pd.Timestamp("2026-03-13", tz="UTC"),
                from_contract="ESH6",
                to_contract="ESM6",
                gap=-12.5,
            )
        )
        assert "ESH6" in text and "ESM6" in text and "-12.50" in text

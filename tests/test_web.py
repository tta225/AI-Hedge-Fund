"""Web API contract tests. Offline — every case runs on generated bars.

Two properties matter more than the rest and are tested first:

* **Synthetic data never substitutes for a failed feed.** A request that asks
  for real data and cannot get it must fail, not quietly return generated bars.
  A fabricated equity curve rendered in a polished UI is worse than an error.
* **Every price-bearing response carries provenance, and `is_evidential` is
  honest.** The browser is a long way from the engine; the flag is the only
  thing standing between a chart and a misreading.
"""

from __future__ import annotations

import json

import pytest

from axiom.data.base import ProviderError

fastapi = pytest.importorskip("fastapi", reason="the web extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402

from axiom.web.server import app  # noqa: E402

SYNTH = {"symbol": "ES", "timeframe": "15m", "days": 20, "synthetic": True, "seed": 3}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


class TestMeta:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/api/health").json()["status"] == "ok"

    def test_instruments_and_menus(self, client: TestClient) -> None:
        payload = client.get("/api/instruments").json()
        symbols = {i["symbol"] for i in payload["instruments"]}
        assert {"ES", "BTC-USD", "SPY"} <= symbols
        assert "silver-bullet" in payload["strategies"]
        assert "15m" in payload["timeframes"]

    def test_strategy_menu_matches_the_cli(self, client: TestClient) -> None:
        """A strategy one surface can run and the other cannot is a bug."""
        from axiom.cli import STRATEGIES

        assert client.get("/api/instruments").json()["strategies"] == sorted(STRATEGIES)

    def test_console_is_served_at_the_root(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "AXIOM" in response.text


class TestProvenance:
    def test_generated_bars_are_labelled_not_evidential(self, client: TestClient) -> None:
        provenance = client.post("/api/analyse", json=SYNTH).json()["series"]["provenance"]
        assert provenance["kind"] == "synthetic"
        assert provenance["is_evidential"] is False

    def test_a_backtest_carries_its_own_provenance_and_warning(
        self, client: TestClient
    ) -> None:
        bt = client.post("/api/backtest", json={**SYNTH, "days": 90}).json()["backtest"]
        assert bt["is_evidence"] is False
        assert bt["warning"], "generated-data results must carry a visible warning"
        assert bt["provenance"]["kind"] == "synthetic"

    def test_caveats_survive_serialisation(self, client: TestClient) -> None:
        """The notes carry the hedges. A UI that drops them publishes a claim."""
        bt = client.post("/api/backtest", json={**SYNTH, "days": 90}).json()["backtest"]
        assert bt["notes"], "the report's notes must reach the client"
        assert any("slippage" in n or "Costs" in n for n in bt["notes"])


class TestNoSilentFallback:
    def test_a_failed_feed_is_an_error_not_generated_bars(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property this whole module exists to protect."""

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise ProviderError("all providers failed")

        monkeypatch.setattr("axiom.data.registry.DataRegistry.fetch_bars", refuse)
        response = client.post("/api/analyse", json={**SYNTH, "synthetic": False})

        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "no_market_data"

    def test_the_error_tells_the_user_what_to_do(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_args: object, **_kwargs: object) -> None:
            raise ProviderError("all providers failed")

        monkeypatch.setattr("axiom.data.registry.DataRegistry.fetch_bars", refuse)
        detail = client.post("/api/backtest", json={**SYNTH, "synthetic": False}).json()["detail"]
        assert "synthetic" in detail["hint"].lower() or "generated" in detail["hint"].lower()


class TestAnalyse:
    def test_returns_bars_and_every_ict_layer(self, client: TestClient) -> None:
        payload = client.post("/api/analyse", json=SYNTH).json()
        series, ict = payload["series"], payload["ict"]

        assert series["bars"] == len(series["close"]) == len(series["time"])
        for layer in ("swings", "structure_events", "fair_value_gaps",
                      "order_blocks", "liquidity_pools", "sweeps"):
            assert layer in ict
        assert ict["bias"] in {"bullish", "bearish", "neutral"}

    def test_feature_indices_are_rebased_onto_the_returned_window(
        self, client: TestClient
    ) -> None:
        """The chart addresses bars by payload position.

        If the server sends absolute indices for a truncated window, every zone
        is drawn against the wrong candle — and it still *looks* like a chart,
        which is why this is asserted rather than eyeballed.
        """
        payload = client.post(
            "/api/analyse", json={**SYNTH, "days": 60, "max_bars": 200}
        ).json()
        bars = payload["series"]["bars"]
        assert bars == 200
        assert payload["series"]["index_offset"] > 0

        confirmed = [g["confirmed_index"] for g in payload["ict"]["fair_value_gaps"]]
        assert confirmed, "expected some gaps in a 60-day window"
        # Rebased indices may be negative (confirmed before the window opened),
        # but none may point past its end.
        assert max(confirmed) < bars

    def test_strict_preset_is_not_more_permissive(self, client: TestClient) -> None:
        loose = client.post("/api/analyse", json=SYNTH).json()["ict"]["counts"]
        strict = client.post(
            "/api/analyse", json={**SYNTH, "strict": True}
        ).json()["ict"]["counts"]
        assert strict["order_blocks"] <= loose["order_blocks"]

    def test_max_bars_caps_the_payload(self, client: TestClient) -> None:
        payload = client.post("/api/analyse", json={**SYNTH, "max_bars": 120}).json()
        assert payload["series"]["bars"] == 120
        assert payload["series"]["total_bars"] > 120


class TestBacktest:
    def test_runs_and_reports_metrics(self, client: TestClient) -> None:
        bt = client.post("/api/backtest", json={**SYNTH, "days": 90}).json()["backtest"]
        assert bt["starting_equity"] == 250_000.0
        assert "sharpe" in bt["metrics"]
        assert len(bt["equity_curve"]["time"]) == len(bt["equity_curve"]["value"])
        assert set(bt["funnel"]) >= {"signals", "orders", "trades"}

    def test_unknown_strategy_names_the_alternatives(self, client: TestClient) -> None:
        response = client.post("/api/backtest", json={**SYNTH, "strategy": "nope"})
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["error"] == "unknown_strategy"
        assert "silver-bullet" in detail["available"]

    def test_every_trade_carries_an_r_multiple(self, client: TestClient) -> None:
        bt = client.post("/api/backtest", json={**SYNTH, "days": 120}).json()["backtest"]
        for trade in bt["trades"]:
            assert {"side", "pnl", "r_multiple", "exit_reason"} <= set(trade)


class TestJsonSafety:
    def test_no_bare_nan_reaches_the_client(self, client: TestClient) -> None:
        """`json.dumps` emits bare NaN, which `JSON.parse` rejects.

        One non-finite metric would otherwise fail the entire response rather
        than one field, so the whole panel would go blank.
        """
        raw = client.post("/api/backtest", json={**SYNTH, "days": 30}).text
        assert "NaN" not in raw
        assert "Infinity" not in raw
        json.loads(raw)  # strict parse: no NaN/Infinity extensions allowed

    def test_out_of_range_lookback_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/analyse", json={**SYNTH, "days": 10_000}).status_code == 422
        assert client.post("/api/analyse", json={**SYNTH, "days": 0}).status_code == 422


class TestStatus:
    def test_separates_credentialed_from_reachable(self, client: TestClient) -> None:
        """The distinction the data-check fix exists to preserve."""
        registry = client.get("/api/status").json()["registry"]
        assert set(registry) >= {"order", "credentialed", "reachable"}
        assert set(registry["reachable"]) <= set(registry["credentialed"])

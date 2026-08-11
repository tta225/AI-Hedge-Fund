"""The HTML report.

The properties under test are the ones that make the file safe to hand to
someone who was not present when it ran: it carries only measured values, it
states its provenance in a way that cannot be scrolled past, and it never
renders an unavailable metric as a number.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from axiom.agents.base import AgentRole
from axiom.agents.pipeline import AgentPipeline, PipelineResult
from axiom.core.config import AgentSettings, RiskSettings
from axiom.core.provenance import Provenance
from axiom.core.series import OHLCVSeries
from axiom.report import build_payload, render_html, write_report
from axiom.report.html import _fmt
from axiom.risk.manager import RiskManager
from axiom.strategy.ict_strategies import SilverBulletStrategy

RISK_SETTINGS = RiskSettings(account_equity=250_000, max_gross_exposure_pct=2000.0)


@pytest.fixture(scope="module")
def run(synthetic_series_module: OHLCVSeries) -> PipelineResult:
    return AgentPipeline(AgentSettings(ANTHROPIC_API_KEY="")).run(
        synthetic_series_module, SilverBulletStrategy(), risk_settings=RISK_SETTINGS
    )


@pytest.fixture(scope="module")
def payload(
    run: PipelineResult, synthetic_series_module: OHLCVSeries
) -> dict[str, Any]:
    return build_payload(
        run,
        synthetic_series_module,
        strategy_name="silver-bullet",
        risk=RiskManager(RISK_SETTINGS),
    )


@pytest.fixture(scope="module")
def page(payload: dict[str, Any]) -> str:
    return render_html(payload)


class TestPipelineCarriesItsEvidence:
    def test_result_keeps_the_run_it_reported_on(self, run: PipelineResult) -> None:
        assert run.backtest is not None
        assert run.ict_state is not None
        assert run.backtest.report.metrics["trades"] == len(run.backtest.trades)


class TestPayload:
    def test_is_json_serialisable_without_nan(self, payload: dict[str, Any]) -> None:
        # allow_nan=False is the assertion: NaN is valid Python but not valid
        # JSON, and a page whose payload cannot be parsed renders nothing.
        json.dumps(payload, allow_nan=False)

    def test_carries_every_stage(self, payload: dict[str, Any]) -> None:
        roles = [stage["role"] for stage in payload["stages"]]
        assert roles == [role.value for role in AgentRole]

    def test_metrics_match_the_backtest(
        self, payload: dict[str, Any], run: PipelineResult
    ) -> None:
        assert run.backtest is not None
        assert payload["backtest"]["metrics"]["trades"] == run.backtest.report.metrics["trades"]
        assert len(payload["backtest"]["trades"]) == len(run.backtest.trades)

    def test_unavailable_metrics_are_none_not_zero(self) -> None:
        from axiom.report.payload import _clean

        assert _clean(float("nan")) is None
        assert _clean(float("inf")) is None
        assert _clean(0.0) == 0.0

    def test_equity_curve_keeps_its_final_observation(
        self, payload: dict[str, Any], run: PipelineResult
    ) -> None:
        assert run.backtest is not None
        curve = payload["backtest"]["equity_curve"]
        assert curve
        assert curve[-1]["equity"] == pytest.approx(
            float(run.backtest.report.equity_curve.iloc[-1])
        )

    def test_refuses_a_result_with_nothing_measured(
        self, synthetic_series_module: OHLCVSeries
    ) -> None:
        with pytest.raises(ValueError, match="no backtest"):
            build_payload(
                PipelineResult(),
                synthetic_series_module,
                strategy_name="silver-bullet",
                risk=RiskManager(RISK_SETTINGS),
            )


class TestRendering:
    def test_page_is_self_contained(self, page: str) -> None:
        # No external fetches: the CSP on a hosted artifact blocks them, and a
        # file on disk has nothing to fetch from.
        assert "<script src=" not in page
        assert "<link " not in page
        assert not re.search(r"(src|href)=\"https?://", page)

    def test_embedded_payload_parses(self, page: str) -> None:
        raw = page.split('<script type="application/json" id="payload">')[1]
        raw = raw.split("</script>")[0]
        assert json.loads(raw.replace("<\\/", "</"))["version"] == 1

    def test_synthetic_run_states_it_is_not_evidence(self, page: str) -> None:
        assert "not market history" in page
        assert "not evidence of profitability" in page

    def test_approval_gate_is_present(self, page: str) -> None:
        assert "Human approval required" in page
        assert "no route to a venue" in page

    def test_measured_values_appear(self, page: str, payload: dict[str, Any]) -> None:
        trades = payload["backtest"]["metrics"]["trades"]
        assert f"{trades:,.0f}" in page

    def test_escapes_hostile_content(self, payload: dict[str, Any]) -> None:
        hostile = dict(payload)
        hostile["stages"] = [
            {
                "role": "research",
                "facts": {"symbol": "<img src=x onerror=alert(1)>"},
                "warnings": ["</script><script>alert(2)</script>"],
                "narrative": "",
                "model": "",
                "used_llm": False,
                "produced_at": "",
            }
        ]
        rendered = render_html(hostile)
        body, embedded = rendered.split('<script type="application/json" id="payload">')

        # Nothing hostile survives into the document body...
        assert "<img src=x" not in body
        assert "<script>alert(2)" not in body
        assert "&lt;img src=x" in body
        # ...and the payload block cannot be closed early to escape into HTML.
        assert "</script><script>" not in embedded


class TestFormatting:
    def test_missing_values_read_na(self) -> None:
        assert _fmt(None) == "n/a"
        assert _fmt(None, "pct") == "n/a"

    def test_negative_cash_keeps_the_sign_outside(self) -> None:
        assert _fmt(-203.4, "cash") == "-$203.40"

    def test_percentages_and_r_multiples(self) -> None:
        assert _fmt(7.869, "pct") == "7.87%"
        assert _fmt(1.0757, "r") == "+1.08R"


class TestWriting:
    def test_writes_html_and_payload(
        self, payload: dict[str, Any], tmp_path: Any
    ) -> None:
        html_path = tmp_path / "nested" / "report.html"
        json_path = tmp_path / "nested" / "report.json"
        written = write_report(payload, html_path, json_path)

        assert written == html_path
        assert html_path.read_text(encoding="utf-8").startswith("<meta charset")
        assert json.loads(json_path.read_text(encoding="utf-8"))["version"] == 1


class TestProvenanceRendering:
    def test_real_data_drops_the_banner(self, payload: dict[str, Any]) -> None:
        real = json.loads(json.dumps(payload))
        real["provenance"] = {
            **real["provenance"],
            "kind": "real",
            "is_evidential": True,
            "label": "coinbase/real",
        }
        rendered = render_html(real)
        # The banner is what a reader cannot scroll past; stage warnings from
        # the run itself are quoted verbatim and stay whatever they were.
        assert 'class="banner"' not in rendered
        assert "coinbase/real" in rendered
        assert 'class="banner"' in render_html(payload)

    def test_provenance_block_reflects_the_series(
        self, payload: dict[str, Any], synthetic_series_module: OHLCVSeries
    ) -> None:
        provenance: Provenance = synthetic_series_module.provenance
        assert payload["provenance"]["label"] == provenance.label
        assert payload["provenance"]["is_evidential"] is False

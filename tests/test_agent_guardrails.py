"""The numeric-fidelity check.

This is the test file that earns its keep. The module's stated principle is
that agents never produce numbers, and until this check existed that was a
polite request in a system prompt. The property under test: a figure that
appears in generated prose but not in the measured facts is caught, and a
figure that was legitimately supplied is not.

The asymmetry is deliberate and tested in both directions — a false positive
costs a reader one glance, a false negative puts a fabricated number in front
of someone sizing a position.
"""

from __future__ import annotations

import pytest

from axiom.agents.guardrails import check_numeric_fidelity

FACTS = {
    "symbol": "ES",
    "profit_factor": 1.24,
    "win_rate_pct": 41.5,
    "trades": 87,
    "max_drawdown_pct": 12.75,
    "avg_r": -0.08,
    "range_high": 5312.25,
    "costs_included": True,
}


class TestCatchesFabrication:
    def test_a_number_absent_from_the_facts_is_flagged(self) -> None:
        report = check_numeric_fidelity("Profit factor of 1.81 is healthy.", FACTS)
        assert "1.81" in report.unsupported

    def test_a_plausible_near_miss_is_still_caught(self) -> None:
        """1.24 vs 1.42 is the dangerous case — it reads right and isn't."""
        report = check_numeric_fidelity("The profit factor is 1.42.", FACTS)
        assert "1.42" in report.unsupported

    def test_an_invented_price_level_is_flagged(self) -> None:
        report = check_numeric_fidelity("Resistance sits at 5480.50.", FACTS)
        assert "5480.50" in report.unsupported

    def test_every_distinct_fabrication_is_listed(self) -> None:
        report = check_numeric_fidelity("Saw 2.5 and 99.9 and 1234.5.", FACTS)
        assert set(report.unsupported) == {"2.5", "99.9", "1234.5"}

    def test_a_repeated_fabrication_is_listed_once(self) -> None:
        """Listing one defect five times buries the other four."""
        report = check_numeric_fidelity("1.81 — and again, 1.81, plus 1.81.", FACTS)
        assert report.unsupported == ("1.81",)

    def test_the_warning_names_the_figures(self) -> None:
        warning = check_numeric_fidelity("Profit factor 1.81.", FACTS).warning()
        assert warning is not None
        assert "1.81" in warning and "UNVERIFIED" in warning


class TestAcceptsSuppliedFigures:
    def test_a_quoted_fact_passes(self) -> None:
        report = check_numeric_fidelity(
            "Profit factor of 1.24 across 87 trades.", FACTS
        )
        assert report.clean

    def test_reasonable_rounding_passes(self) -> None:
        report = check_numeric_fidelity("Drawdown was about 12.75%.", FACTS)
        assert report.clean

    def test_a_negative_value_passes(self) -> None:
        report = check_numeric_fidelity("Average R is -0.08, i.e. losing.", FACTS)
        assert report.clean

    def test_thousands_separators_and_currency_pass(self) -> None:
        report = check_numeric_fidelity(
            "The high was 5,312.25.", {"range_high": 5312.25}
        )
        assert report.clean

    def test_a_percentage_of_a_fractional_fact_passes(self) -> None:
        """A model writing 0.62 as "62%" is reporting, not inventing."""
        report = check_numeric_fidelity("That is 62%.", {"share": 0.62})
        assert report.clean

    def test_a_threshold_from_the_prompt_passes(self) -> None:
        """Quoting a number from its own instructions is not fabrication."""
        prompt = "Flag any reward:risk below 2.0."
        report = check_numeric_fidelity(
            "Reward:risk is under the 2.0 threshold.", FACTS, prompt
        )
        assert report.clean

    def test_numbers_inside_string_facts_are_supported(self) -> None:
        facts = {"decision": "risking $500.00 over a 11.5000 point stop"}
        report = check_numeric_fidelity("It risks $500.00 on an 11.5 point stop.", facts)
        assert report.clean

    def test_an_empty_narrative_is_clean(self) -> None:
        assert check_numeric_fidelity("", FACTS).clean

    def test_prose_with_no_numbers_is_clean(self) -> None:
        assert check_numeric_fidelity("The structure looks weak.", FACTS).clean


class TestFalsePositiveControl:
    @pytest.mark.parametrize(
        "text",
        [
            "There are 3 concerns worth naming.",
            "Both sides have merit; 2 scenarios dominate.",
            "The first 10 bars are warmup.",
        ],
    )
    def test_small_integers_read_as_prose_not_measurement(self, text: str) -> None:
        assert check_numeric_fidelity(text, FACTS).clean

    def test_a_large_bare_integer_is_still_a_claim(self) -> None:
        report = check_numeric_fidelity("Across 450 sessions.", FACTS)
        assert "450" in report.unsupported

    def test_timestamps_do_not_contribute_figures(self) -> None:
        """An ISO timestamp alone is six unrelated integers."""
        report = check_numeric_fidelity(
            "As of 2026-03-14T09:30:00+00:00 the read is unchanged.", FACTS
        )
        assert report.clean

    def test_a_bare_date_does_not_contribute_figures(self) -> None:
        assert check_numeric_fidelity("Since 2024-01-15 the range held.", FACTS).clean

    def test_the_checked_count_reflects_examined_figures(self) -> None:
        report = check_numeric_fidelity("Values 1.24 and 99.9 and 3 items.", FACTS)
        assert report.checked == 2  # "3" is prose


class TestBooleanFactsAreNotNumbers:
    def test_a_true_fact_does_not_license_the_figure_one(self) -> None:
        """Python's bool is an int; treating True as 1.0 would license "1"."""
        report = check_numeric_fidelity("A ratio of 1.0000001 was seen.", FACTS)
        assert report.unsupported

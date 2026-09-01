"""Tests for the outside view on a reported Sharpe ratio."""

from __future__ import annotations

import pytest

from axiom.research.reference import (
    REFERENCE_MAX_SHARPE,
    REFERENCE_MEDIAN_SHARPE,
    assess_plausibility,
    reference_percentile,
    sharpe_confidence_interval,
    sharpe_standard_error,
)


class TestStandardError:
    def test_error_shrinks_with_sample_size(self) -> None:
        """The whole reason short backtests are not comparable to long ones."""
        assert sharpe_standard_error(1.0, 40) > sharpe_standard_error(1.0, 4000)

    def test_error_grows_with_sharpe(self) -> None:
        assert sharpe_standard_error(3.0, 500) > sharpe_standard_error(0.3, 500)

    def test_known_value(self) -> None:
        # sqrt((1 + 0) / 100) = 0.1 for a Sharpe of exactly zero.
        assert sharpe_standard_error(0.0, 100) == pytest.approx(0.1)

    def test_rejects_degenerate_sample(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            sharpe_standard_error(1.0, 1)


class TestConfidenceInterval:
    def test_interval_brackets_the_estimate(self) -> None:
        low, high = sharpe_confidence_interval(1.5, 200)
        assert low < 1.5 < high

    def test_wider_confidence_gives_a_wider_interval(self) -> None:
        narrow = sharpe_confidence_interval(1.0, 200, confidence=0.80)
        wide = sharpe_confidence_interval(1.0, 200, confidence=0.99)
        assert wide[0] < narrow[0] and wide[1] > narrow[1]

    def test_short_sample_interval_swallows_the_literature(self) -> None:
        """A modest Sharpe from 40 observations cannot separate itself."""
        low, high = sharpe_confidence_interval(0.6, 40)
        assert low < REFERENCE_MEDIAN_SHARPE < high

    def test_rejects_impossible_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            sharpe_confidence_interval(1.0, 200, confidence=1.5)


class TestPercentile:
    def test_monotonic(self) -> None:
        values = [-1.0, -0.272, 0.0, 0.142, 0.354, 0.559, 0.892, 2.0]
        percentiles = [reference_percentile(v) for v in values]
        assert percentiles == sorted(percentiles)

    def test_quantile_knots_land_where_they_should(self) -> None:
        assert reference_percentile(REFERENCE_MEDIAN_SHARPE) == pytest.approx(0.50)
        assert reference_percentile(REFERENCE_MAX_SHARPE) == pytest.approx(1.0)

    def test_bounded(self) -> None:
        assert reference_percentile(-99.0) == 0.0
        assert reference_percentile(99.0) == 1.0


class TestPlausibility:
    def test_flags_a_sharpe_above_the_whole_literature(self) -> None:
        """The case the module exists for."""
        check = assess_plausibility(3.5, 500)
        assert check.exceeds_best_published
        assert check.is_suspicious
        assert "ABOVE THE LITERATURE" in check.verdict

    def test_does_not_flag_an_ordinary_result(self) -> None:
        check = assess_plausibility(0.4, 2000)
        assert not check.is_suspicious
        # 56th percentile. Statistically separable from the median at this
        # sample size, but separable is not the same as strong.
        assert "ORDINARY" in check.verdict

    def test_a_strong_long_sample_result_is_credited(self) -> None:
        check = assess_plausibility(0.8, 7500)
        assert not check.exceeds_best_published
        assert "STRONG FOR THE CLASS" in check.verdict

    def test_short_samples_are_called_indistinguishable(self) -> None:
        """40 observations cannot separate a good anomaly from an average one."""
        check = assess_plausibility(0.6, 40)
        assert check.indistinguishable_from_median

    def test_long_samples_can_separate(self) -> None:
        check = assess_plausibility(0.6, 5000)
        assert not check.indistinguishable_from_median

    def test_verdict_never_rejects_outright(self) -> None:
        """This is a prior, not a gate — HFT lives outside the class honestly."""
        check = assess_plausibility(5.0, 1000)
        assert "not impossible" in check.verdict

"""Unit tests for the Poisson O/U conversion."""

from __future__ import annotations

import numpy as np
import pytest

from football_llm.eval import poisson


class TestPOver25:
    def test_known_values(self):
        """Hand-computed: P(X > 2 | Poisson(λ)) = 1 - e^-λ (1 + λ + λ²/2)."""
        assert poisson.p_over_25(0.0) == 0.0
        # λ = 1: 1 - e^-1 * (1 + 1 + 0.5) = 1 - 0.3679 * 2.5 ≈ 0.0803
        assert poisson.p_over_25(1.0) == pytest.approx(0.0803, abs=1e-4)
        # λ = 2: 1 - e^-2 * (1 + 2 + 2) = 1 - 0.1353 * 5 ≈ 0.3233
        assert poisson.p_over_25(2.0) == pytest.approx(0.3233, abs=1e-4)
        # λ = 2.5: 1 - e^-2.5 * (1 + 2.5 + 3.125) = 1 - 0.0821 * 6.625 ≈ 0.4562
        assert poisson.p_over_25(2.5) == pytest.approx(0.4562, abs=1e-4)
        # λ = 4: 1 - e^-4 * (1 + 4 + 8) = 1 - 0.01832 * 13 ≈ 0.7619
        assert poisson.p_over_25(4.0) == pytest.approx(0.7619, abs=1e-4)

    def test_monotone_increasing(self):
        """Higher expected goals → higher P(over 2.5)."""
        lambdas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        probs = [poisson.p_over_25(l) for l in lambdas]
        assert all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1))

    def test_bounded_in_unit_interval(self):
        for l in [0.01, 0.5, 1.0, 5.0, 10.0, 100.0]:
            p = poisson.p_over_25(l)
            assert 0.0 <= p <= 1.0

    def test_vectorized_matches_scalar(self):
        lambdas = [0.5, 1.5, 2.5, 3.5, 4.5]
        scalar = np.array([poisson.p_over_25(l) for l in lambdas])
        vector = poisson.p_over_25_vectorized(lambdas)
        np.testing.assert_allclose(scalar, vector, atol=1e-12)

    def test_negative_lambda_returns_zero(self):
        """Defensive: negative λ (can happen for regressor outputs) → 0."""
        assert poisson.p_over_25(-1.0) == 0.0


class TestPOverLine:
    def test_line_1_5(self):
        """P(over 1.5 | λ=2) = 1 - P(X ≤ 1 | λ=2) = 1 - e^-2(1+2) = 1 - 0.4060 ≈ 0.5940."""
        assert poisson.p_over_line(2.0, line=1.5) == pytest.approx(0.5940, abs=1e-4)

    def test_line_3_5(self):
        """P(over 3.5 | λ=3) = 1 - P(X ≤ 3 | λ=3)."""
        # P(X ≤ 3 | λ=3) = e^-3 * (1 + 3 + 4.5 + 4.5) = 0.0498 * 13 = 0.6472
        assert poisson.p_over_line(3.0, line=3.5) == pytest.approx(0.3528, abs=1e-3)

    def test_line_2_5_matches_dedicated(self):
        """p_over_line(λ, 2.5) should match p_over_25(λ)."""
        for l in [0.5, 1.5, 2.5, 3.5]:
            assert poisson.p_over_line(l, 2.5) == pytest.approx(poisson.p_over_25(l), abs=1e-12)

import math
import sys
from itertools import product
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evrl_math import finite_n_gaussian_calibration, gaussian_max_credit


def normal_ei(delta: float, sigma: float) -> float:
    z = delta / sigma
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return delta * cdf + sigma * phi


def expected_max_pair(mu_a: float, sigma_a: float, mu_b: float, sigma_b: float) -> float:
    difference_sigma = math.sqrt(sigma_a * sigma_a + sigma_b * sigma_b)
    return mu_b + normal_ei(mu_a - mu_b, difference_sigma)


def test_finite_n_calibration_matches_reported_value():
    assert finite_n_gaussian_calibration(1) == 0.0
    assert abs(finite_n_gaussian_calibration(32) - 2.0696688279) < 1e-8


def test_two_candidate_credit_matches_closed_form():
    mu = torch.tensor([0.3, -0.2])
    sigma = torch.tensor([0.7, 0.4])
    credit, diagnostics = gaussian_max_credit(mu, sigma, 32, 256, 1e-9)
    difference_sigma = math.sqrt(0.7**2 + 0.4**2)
    expected = torch.tensor(
        [
            normal_ei(0.5, difference_sigma),
            normal_ei(-0.5, difference_sigma),
        ]
    )
    assert torch.allclose(credit, expected, atol=2e-7, rtol=2e-7)
    assert diagnostics.converged


def test_deterministic_candidates_are_supported_without_variance_floor():
    credit, diagnostics = gaussian_max_credit(
        torch.tensor([2.0, 1.0, -1.0]),
        torch.zeros(3),
        initial_order=32,
        max_order=256,
        tolerance=1e-9,
    )
    assert torch.allclose(credit, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert diagnostics.deterministic_breaks == 3


def test_n1_credit_is_mean_reward_exactly():
    credit, diagnostics = gaussian_max_credit(torch.tensor([1.75]), torch.tensor([0.9]))
    assert credit.item() == 1.75
    assert diagnostics.converged


def test_quadrature_credit_matches_finite_difference_policy_gradient():
    # Two possible responses and N=2 iid policy samples.
    mus = [0.6, -0.1]
    sigmas = [0.5, 0.9]

    def objective(theta: float) -> float:
        p = 1.0 / (1.0 + math.exp(-theta))
        probabilities = [p, 1.0 - p]
        value = 0.0
        for first, second in product(range(2), repeat=2):
            value += (
                probabilities[first]
                * probabilities[second]
                * expected_max_pair(mus[first], sigmas[first], mus[second], sigmas[second])
            )
        return value

    theta = 0.37
    p = 1.0 / (1.0 + math.exp(-theta))
    probabilities = [p, 1.0 - p]
    score = [1.0 - p, -p]
    gradient = 0.0
    for first, second in product(range(2), repeat=2):
        group_mu = torch.tensor([mus[first], mus[second]])
        group_sigma = torch.tensor([sigmas[first], sigmas[second]])
        credit, _ = gaussian_max_credit(group_mu, group_sigma, 64, 256, 1e-9)
        gradient += (
            probabilities[first]
            * probabilities[second]
            * (credit[0].item() * score[first] + credit[1].item() * score[second])
        )

    step = 1e-5
    finite_difference = (objective(theta + step) - objective(theta - step)) / (2.0 * step)
    assert abs(gradient - finite_difference) < 3e-6


if __name__ == "__main__":
    tests = sorted(name for name in globals() if name.startswith("test_"))
    for name in tests:
        globals()[name]()
        print(f"PASS {name}")

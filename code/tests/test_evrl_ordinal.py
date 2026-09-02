import math
import sys
from itertools import product
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evrl_ordinal import (
    exact_expected_max,
    exact_marginal_max_credit,
    exact_probability_any_rating,
    robust_group_statistics,
    robust_lower_probabilities,
    scalar_max_marginal_credit,
    validate_probabilities,
)


def brute_force(probabilities: torch.Tensor) -> tuple[float, float]:
    expected_max = 0.0
    any_top = 0.0
    for ratings in product(range(5), repeat=probabilities.shape[0]):
        probability = math.prod(
            float(probabilities[index, rating]) for index, rating in enumerate(ratings)
        )
        expected_max += probability * max(ratings)
        any_top += probability * float(4 in ratings)
    return expected_max, any_top


def test_exact_metrics_match_brute_force_enumeration():
    probabilities = torch.tensor(
        [
            [0.10, 0.15, 0.20, 0.25, 0.30],
            [0.35, 0.25, 0.20, 0.15, 0.05],
            [0.05, 0.10, 0.20, 0.25, 0.40],
        ],
        dtype=torch.float64,
    )
    expected_max, any_top = brute_force(probabilities)
    batch = probabilities.unsqueeze(0)
    assert abs(exact_expected_max(batch).item() - expected_max) < 1e-12
    assert abs(exact_probability_any_rating(batch).item() - any_top) < 1e-12


def test_marginal_credit_matches_direct_definition():
    probabilities = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.25, 0.15],
            [0.30, 0.25, 0.20, 0.15, 0.10],
            [0.05, 0.10, 0.15, 0.30, 0.40],
        ],
        dtype=torch.float64,
    )
    expected = []
    for candidate in range(probabilities.shape[0]):
        value = 0.0
        for ratings in product(range(5), repeat=probabilities.shape[0]):
            probability = math.prod(
                float(probabilities[index, rating])
                for index, rating in enumerate(ratings)
            )
            opponents = [rating for index, rating in enumerate(ratings) if index != candidate]
            value += probability * max(ratings[candidate] - max(opponents), 0)
        expected.append(value)
    credit = exact_marginal_max_credit(probabilities.unsqueeze(0))[0]
    assert torch.allclose(credit, torch.tensor(expected, dtype=credit.dtype), atol=1e-12)


def test_n1_credit_reduces_to_expected_rating():
    probabilities = torch.tensor([[[0.05, 0.10, 0.15, 0.30, 0.40]]])
    expected_rating = sum(index * float(value) for index, value in enumerate(probabilities[0, 0]))
    assert abs(exact_marginal_max_credit(probabilities).item() - expected_rating) < 1e-6


def test_credit_matches_finite_difference_policy_gradient():
    response_distributions = torch.tensor(
        [
            [0.10, 0.15, 0.20, 0.25, 0.30],
            [0.35, 0.25, 0.20, 0.15, 0.05],
        ],
        dtype=torch.float64,
    )

    def objective(theta: float) -> float:
        policy = [1.0 / (1.0 + math.exp(-theta)), 0.0]
        policy[1] = 1.0 - policy[0]
        value = 0.0
        for first, second in product(range(2), repeat=2):
            group = response_distributions[[first, second]].unsqueeze(0)
            value += policy[first] * policy[second] * exact_expected_max(group).item()
        return value

    theta = 0.37
    policy = [1.0 / (1.0 + math.exp(-theta)), 0.0]
    policy[1] = 1.0 - policy[0]
    score = [1.0 - policy[0], -policy[0]]
    gradient = 0.0
    for first, second in product(range(2), repeat=2):
        group = response_distributions[[first, second]].unsqueeze(0)
        credit = exact_marginal_max_credit(group)[0]
        gradient += policy[first] * policy[second] * (
            credit[0].item() * score[first] + credit[1].item() * score[second]
        )
    step = 1e-5
    finite_difference = (objective(theta + step) - objective(theta - step)) / (2.0 * step)
    assert abs(gradient - finite_difference) < 2e-9


def test_invalid_probabilities_fail_loudly():
    cases = (
        (torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]), "sum to one"),
        (torch.tensor([[0.2, 0.2, 0.2, 0.5, -0.1]]), "nonnegative"),
        (torch.tensor([[0.2, 0.2, 0.2, 0.2, float("nan")]]), "finite"),
    )
    for probabilities, message in cases:
        try:
            validate_probabilities(probabilities)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"invalid probabilities did not raise: {probabilities}")


def test_robust_envelope_is_valid_and_epsilon_zero_is_nominal():
    probabilities = torch.tensor(
        [[[0.10, 0.15, 0.20, 0.25, 0.30], [0.35, 0.25, 0.20, 0.15, 0.05]]],
        dtype=torch.float64,
    )
    assert torch.allclose(
        robust_lower_probabilities(probabilities, 0.0),
        probabilities,
        atol=1e-15,
    )
    robust = robust_lower_probabilities(probabilities, 0.12)
    assert torch.allclose(robust.sum(dim=-1), torch.ones_like(robust[..., 0]))
    assert (robust >= 0.0).all()
    nominal_cdf = probabilities.cumsum(dim=-1)[..., :-1]
    robust_cdf = robust.cumsum(dim=-1)[..., :-1]
    assert torch.allclose(robust_cdf, (nominal_cdf + 0.12).clamp_max(1.0))


def test_robust_expected_max_is_monotone_in_radius_and_n1_reduces_to_robust_mean():
    probabilities = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.25, 0.15], [0.05, 0.10, 0.15, 0.30, 0.40]]],
        dtype=torch.float64,
    )
    nominal = robust_group_statistics(probabilities, 0.0)
    robust = robust_group_statistics(probabilities, 0.15)
    assert robust["robust_expected_max"].item() <= nominal["robust_expected_max"].item()
    single = probabilities[:, :1]
    single_stats = robust_group_statistics(single, 0.15)
    assert torch.allclose(
        single_stats["robust_marginal_credit"],
        single_stats["robust_expected_rating"],
        atol=1e-12,
    )


def test_scalar_max_credit_matches_leave_one_out_definition():
    values = torch.tensor([[0.2, 0.8, 0.5], [1.4, 1.4, -0.2]])
    expected = torch.tensor([[0.0, 0.3, 0.0], [0.0, 0.0, 0.0]])
    assert torch.allclose(scalar_max_marginal_credit(values), expected)
    single = torch.tensor([[-0.4], [2.1]])
    assert torch.equal(scalar_max_marginal_credit(single), single)


def test_robust_credit_matches_finite_difference_policy_gradient():
    response_distributions = torch.tensor(
        [
            [0.10, 0.15, 0.20, 0.25, 0.30],
            [0.35, 0.25, 0.20, 0.15, 0.05],
        ],
        dtype=torch.float64,
    )
    epsilon = 0.08

    def objective(theta: float) -> float:
        policy = [1.0 / (1.0 + math.exp(-theta)), 0.0]
        policy[1] = 1.0 - policy[0]
        value = 0.0
        for first, second in product(range(2), repeat=2):
            group = response_distributions[[first, second]].unsqueeze(0)
            robust = robust_group_statistics(group, epsilon)
            value += (
                policy[first]
                * policy[second]
                * robust["robust_expected_max"].item()
            )
        return value

    theta = -0.21
    policy = [1.0 / (1.0 + math.exp(-theta)), 0.0]
    policy[1] = 1.0 - policy[0]
    score = [1.0 - policy[0], -policy[0]]
    gradient = 0.0
    for first, second in product(range(2), repeat=2):
        group = response_distributions[[first, second]].unsqueeze(0)
        credit = robust_group_statistics(group, epsilon)["robust_marginal_credit"][0]
        gradient += policy[first] * policy[second] * (
            credit[0].item() * score[first] + credit[1].item() * score[second]
        )
    step = 1e-5
    finite_difference = (objective(theta + step) - objective(theta - step)) / (2.0 * step)
    assert abs(gradient - finite_difference) < 2e-9


if __name__ == "__main__":
    for name in sorted(name for name in globals() if name.startswith("test_")):
        globals()[name]()
        print(f"PASS {name}")

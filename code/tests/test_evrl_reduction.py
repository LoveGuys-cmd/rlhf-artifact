import math
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evrl_experiment import (
    adaptive_kl_update,
    clustered_one_sided_calibration_radius,
    decode_ordinal_reward_distribution,
    decode_ordinal_reward_moments,
    mean_constrained_credit,
    method_credit,
    projected_dual_update,
    quality_constrained_credit,
    restore_trainable_update,
    snapshot_trainable_update,
    verify_trainable_snapshot,
)
from evrl_ordinal import robust_group_statistics
from reward_hacking_diagnostic import ordinal_group_metrics


def ordinal_fixture():
    probabilities = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.3, 0.1]],
            [[0.4, 0.3, 0.2, 0.1, 0.0]],
            [[0.0, 0.1, 0.2, 0.3, 0.4]],
        ]
    )
    values = torch.arange(5, dtype=probabilities.dtype)
    mu = (probabilities * values).sum(dim=-1)
    sigma = (
        probabilities * (values - mu.unsqueeze(-1)).square()
    ).sum(dim=-1).sqrt()
    return probabilities, mu, sigma


def test_n1_zero_radius_ev_ppo_credit_matches_vanilla_ppo():
    probabilities, mu, sigma = ordinal_fixture()
    ev = method_credit("ev_ppo", probabilities, mu, sigma, 0.0, 1, 64, 1024, 1e-6, 0)
    vanilla = method_credit(
        "vanilla_ppo", probabilities, mu, sigma, 0.0, 1, 64, 1024, 1e-6, 0
    )
    assert torch.allclose(ev[0], vanilla[0])
    assert ev[1] == vanilla[1] == 1.0
    assert ev[2] and vanilla[2]


def test_n1_robust_ev_ppo_reduces_to_robust_mean():
    probabilities, mu, sigma = ordinal_fixture()
    epsilon = 0.15
    ev = method_credit(
        "ev_ppo",
        probabilities,
        mu,
        sigma,
        0.0,
        1,
        64,
        1024,
        1e-6,
        0,
        epsilon,
    )
    expected = robust_group_statistics(probabilities, epsilon)["robust_expected_rating"]
    assert torch.equal(ev[0], expected)
    assert ev[3] == "robust_ordinal_expected_rating_n1_exact_reduction"


def test_n1_nominal_and_scalar_max_reduce_to_expected_rating():
    probabilities, mu, sigma = ordinal_fixture()
    nominal = method_credit(
        "nominal_ev_ppo", probabilities, mu, sigma, 0.0, 1, 64, 1024, 1e-6, 0
    )
    scalar = method_credit(
        "scalar_max_ppo", probabilities, mu, sigma, 0.0, 1, 64, 1024, 1e-6, 0
    )
    assert torch.allclose(nominal[0], mu)
    assert torch.allclose(scalar[0], mu)


def test_entropic_ppo_is_pessimistic_and_exact_on_point_masses():
    probabilities, mu, sigma = ordinal_fixture()
    entropic = method_credit(
        "entropic_ppo",
        probabilities,
        mu,
        sigma,
        0.0,
        1,
        64,
        1024,
        1e-6,
        0,
        0.0,
        1.0,
    )
    assert torch.all(entropic[0] <= mu + 1e-6)
    assert entropic[3] == "pessimistic_entropic_ordinal_certainty_equivalent"
    point = torch.zeros((1, 1, 5), dtype=torch.float64)
    point[..., 3] = 1.0
    point_mu = torch.full((1, 1), 3.0, dtype=torch.float64)
    point_sigma = torch.zeros_like(point_mu)
    exact = method_credit(
        "entropic_ppo",
        point,
        point_mu,
        point_sigma,
        0.0,
        1,
        64,
        1024,
        1e-6,
        0,
        0.0,
        1.0,
    )
    assert torch.allclose(exact[0], point_mu)


def test_rejected_update_restores_parameters_and_optimizer_state():
    model = torch.nn.Linear(3, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    warmup_loss = model(torch.ones(2, 3)).square().mean()
    warmup_loss.backward()
    optimizer.step()
    snapshot = snapshot_trainable_update(model, optimizer)
    original = model.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(model.weight, original)
    assert optimizer.state

    restore_trainable_update(model, optimizer, snapshot)
    assert torch.equal(model.weight, original)
    assert verify_trainable_snapshot(model, optimizer, snapshot) == (True, True)

    optimizer.zero_grad(set_to_none=True)
    second_loss = model(torch.ones(2, 3)).square().mean()
    second_loss.backward()
    optimizer.step()
    assert not torch.equal(model.weight, original)


def test_distribution_export_reconstructs_legacy_moments():
    config = {
        "attribute_names": ["helpfulness"],
        "reward_attribute_index": 0,
        "rating_min": 0.0,
        "rating_max": 4.0,
        "sigma_floor": 0.001,
        "sigma_temperature": 0.95,
        "latent_mu_parameterization": "unbounded_latent_utility_raw_mu",
        "latent_sigma_parameterization": "softplus(raw_sigma)",
        "reward_moment_mapping": "ordinal_induced_observable_rating_moments",
        "ordinal_cutpoints": [[-1.2, -0.7, -0.2, 0.4]],
    }
    logits = torch.tensor([[0.3, -0.2], [-0.8, 0.7]], dtype=torch.float64)
    probabilities, exported_mu, exported_sigma = decode_ordinal_reward_distribution(
        logits, config
    )
    legacy_mu, legacy_sigma = decode_ordinal_reward_moments(logits, config)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    assert torch.equal(exported_mu, legacy_mu)
    assert torch.equal(exported_sigma, legacy_sigma)


def test_quality_lagrangian_scaling_is_invariant_to_ev_group_scale():
    credit = torch.tensor([[0.1, 0.3]])
    quality = torch.tensor([[1.5, -0.5]])
    adjusted, metrics = quality_constrained_credit(
        credit, quality, quality_floor=0.0, dual_value=2.0, gradient_scale=32.0
    )
    scaled = 32.0 * adjusted
    assert torch.allclose(scaled, 32.0 * credit + 2.0 * quality)
    assert metrics["quality_constraint_active"] is True


def test_mean_lagrangian_scaling_is_invariant_to_ev_group_scale():
    credit = torch.tensor([[0.1, 0.3]])
    nominal_mean = torch.tensor([[1.5, 0.5]])
    adjusted, metrics = mean_constrained_credit(
        credit, nominal_mean, mean_floor=1.0, dual_value=3.0, gradient_scale=32.0
    )
    assert torch.allclose(32.0 * adjusted, 32.0 * credit + 3.0 * nominal_mean)
    assert metrics["mean_constraint_active"] is True


def test_rejected_transition_does_not_advance_controller_or_duals():
    assert projected_dual_update(1.5, 0.7, 0.1, 20.0, False) == 1.5
    assert adaptive_kl_update(0.02, 0.2, 0.02, 1.5, 1.5, 1e-5, 10.0, False) == 0.02
    assert projected_dual_update(1.5, 0.7, 0.1, 20.0, True) > 1.5
    assert adaptive_kl_update(0.02, 0.2, 0.02, 1.5, 1.5, 1e-5, 10.0, True) > 0.02


def test_clustered_calibration_uses_prompt_count():
    residuals = torch.tensor(
        [[0.10, -0.05, 0.00, 0.02], [0.20, 0.05, -0.10, 0.04]],
        dtype=torch.float64,
    )
    epsilon, means, concentration = clustered_one_sided_calibration_radius(
        residuals, 0.05
    )
    expected_concentration = math.sqrt(math.log(80.0) / 4.0)
    assert abs(concentration - expected_concentration) < 1e-12
    assert torch.allclose(
        torch.tensor(means, dtype=torch.float64),
        torch.tensor([0.15, 0.0, -0.05, 0.03], dtype=torch.float64),
    )
    assert abs(epsilon - min(1.0, 0.15 + concentration)) < 1e-12


def test_reward_hacking_group_metric_matches_training_math():
    probabilities = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.3, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]]],
        dtype=torch.float64,
    )
    group = {
        "candidates": [
            {"probabilities": row.tolist()} for row in probabilities[0]
        ]
    }
    epsilon = 0.07
    diagnostic = ordinal_group_metrics(group, epsilon)
    expected = robust_group_statistics(probabilities, epsilon)
    assert abs(
        diagnostic["robust_expected_max"]
        - float(expected["robust_expected_max"].item())
    ) < 1e-12
    assert abs(
        diagnostic["nominal_expected_max"]
        - float(expected["nominal_expected_max"].item())
    ) < 1e-12

if __name__ == "__main__":
    for name in sorted(name for name in globals() if name.startswith("test_")):
        globals()[name]()
        print(f"PASS {name}")

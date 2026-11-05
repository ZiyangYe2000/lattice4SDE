from pathlib import Path

import torch

from qmc_sde_v4.config import load_config, selection_summary
from qmc_sde_v4.models import AnalyticLangevinModel, DimensionUniformModel, OUModel, normalized_fwht
from qmc_sde_v4.estimators import RuntimeOptions
from qmc_sde_v4.sobol import sobol_rule_blocks, sobol_rule_value, validate_sobol_design

ROOT = Path(__file__).resolve().parents[1]


def test_production_cases_and_shared_references():
    cfg = load_config(ROOT / "configs/production.json")
    assert len(cfg["cases"]) == 20
    assert len(cfg["reference_cases"]) == 15
    assert cfg["methods"] == ["mc", "qmc1", "median3", "median13"]
    assert selection_summary(cfg)["formal_records"] == 30720
    rotated = next(c for c in cfg["cases"] if c["case_id"] == "p4-rotated-d16-m256")
    assert rotated["reference_case_id"] == "p3-aligned-d16-m256"


def test_hadamard_is_orthonormal_and_models_run():
    x = torch.randn(7, 16, 3, dtype=torch.float64)
    recovered = normalized_fwht(normalized_fwht(x, dim=1), dim=1)
    assert torch.allclose(recovered, x, atol=1e-12, rtol=1e-12)
    z = torch.randn(5, 4 * 8, dtype=torch.float64)
    for factorization in ("aligned", "rotated"):
        values = DimensionUniformModel(d=4, factorization=factorization).evaluate(z, 8)
        assert values.shape == (5,)
        assert torch.isfinite(values).all()


def test_v10_cases_and_analytic_langevin_run():
    cfg = load_config(ROOT / "configs/production.json")
    p5 = [case for case in cfg["cases"] if case["model"] == "analytic_langevin"]
    assert {(case["d"], case["m"]) for case in p5} == {
        (8, 64), (8, 256), (8, 1024), (4, 256), (64, 256)
    }
    z = torch.randn(5, 8 * 16, dtype=torch.float64)
    values = AnalyticLangevinModel(d=8).evaluate(z, 16)
    assert values.shape == (5,)
    assert torch.isfinite(values).all()


def test_aligned_and_rotated_models_have_the_same_euler_law():
    d, m = 16, 8
    z = torch.randn(5, d * m, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(12))
    rotated_coordinates = normalized_fwht(z.reshape(5, m, d), dim=2).reshape(5, -1)
    aligned = DimensionUniformModel(d=d, factorization="aligned").evaluate(z, m)
    rotated = DimensionUniformModel(d=d, factorization="rotated").evaluate(rotated_coordinates, m)
    assert torch.allclose(aligned, rotated, atol=1e-14, rtol=1e-14)


def test_ou_exact_euler_matches_large_mc():
    model = OUModel(payoff="quadratic")
    z = torch.randn(50000, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    estimate = model.evaluate(z, 16).mean().item()
    assert abs(estimate - model.exact_euler(16)) < 0.01


def test_scrambled_sobol_is_reproducible_and_accurate_for_ou():
    model = OUModel(payoff="quadratic")
    options = RuntimeOptions(device="cpu", dtype=torch.float64, cpu_chunk_points=128)
    first = sobol_rule_value(model, 1024, 16, 12345, options)
    repeated = sobol_rule_value(model, 1024, 16, 12345, options)
    assert first == repeated
    assert abs(first - model.exact_euler(16)) < 0.005
    validate_sobol_design(16384, 2**22, 16)
    blocks = list(sobol_rule_blocks(model, 64, 16, 54321, 0, 16, options))
    resumed = list(sobol_rule_blocks(model, 64, 16, 54321, 32, 16, options))
    assert sum(block[2] for block in blocks) == (
        sum(block[2] for block in blocks[:2]) + sum(block[2] for block in resumed)
    )


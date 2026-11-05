# lattice4SDE

This repository contains the code and data for the numerical experiments in
version 10 of the paper. The experiments compare Monte Carlo, single randomized
QMC, median-of-three QMC, and median-of-thirteen QMC for Euler approximations of
time-dependent SDEs. The reference values are computed independently using
scrambled Sobol rules. All models use the final time `T = 1`.

## Repository layout

```text
configs/production.json       experiment configuration used in the paper
configs/smoke.json            small end-to-end test configuration
configs/sobol_reference.json  scrambled Sobol reference configuration
data/raw/formal.jsonl         raw estimator records
data/references/scrambled_sobol_reference/
                              Sobol components and reference values used for RMSE
figures/paper/                the ten figures used in the paper
scripts/run_sobol_references.py
                              compute the reference values
scripts/run_experiments.py    run the four estimators
scripts/make_tables.py        compute RMSE and fitted slopes
scripts/plot_results.py       generate the paper figures
src/qmc_sde_v4/               simulation and analysis code
tests/                        unit and workflow tests
```

The method-comparison figures show all four estimators. The parameter-slice
figures compare Monte Carlo with median-of-thirteen QMC.

## Installation and tests

```bash
python -m pip install -e ".[test]"
pytest -q
```

The workload can be inspected without running the simulations:

```bash
python scripts/run_sobol_references.py --dry-run
python scripts/run_experiments.py --dry-run
```

## Rebuild the tables and figures

```bash
python scripts/make_tables.py \
  --formal data/raw/formal.jsonl \
  --reference-summary data/references/scrambled_sobol_reference/reference_summary.csv \
  --output results/production/analysis

python scripts/plot_results.py \
  --rmse results/production/analysis/tables/rmse_summary.csv \
  --output results/production/figures
```

## Rerun the simulations

```bash
python scripts/check_platform.py
python scripts/run_sobol_references.py --device cuda
python scripts/run_experiments.py --device cuda
```

Use `--device cpu` on a machine without CUDA. Both simulation commands use
resumable checkpoints under `results/production`.

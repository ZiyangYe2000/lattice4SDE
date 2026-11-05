"""Pilot the scrambled Sobol reference workload and extrapolate wall time."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmc_sde_v4.models import OUModel, model_from_case
from qmc_sde_v4.randomization import stable_seed
from qmc_sde_v4.runtime import runtime_options, synchronize
from qmc_sde_v4.sobol import sobol_rule_value
from qmc_sde_v4.sobol_references import load_sobol_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate scrambled Sobol reference runtime.")
    parser.add_argument("--config", default=str(ROOT / "configs/sobol_reference.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pilot-points", type=int, default=16384)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.pilot_points < 2 or args.pilot_points & (args.pilot_points - 1):
        raise ValueError("pilot-points must be a power of two")
    spec, cfg = load_sobol_config(args.config)
    torch.set_num_threads(int(cfg.get("cpu_threads", max(1, min(8, os.cpu_count() or 1)))))
    options = replace(
        runtime_options(cfg, args.device, reference=True),
        max_chunk_points=spec["max_chunk_points"],
    )
    cases = [case for case in cfg["reference_cases"]
             if not isinstance(model_from_case(case), OUModel)]
    details, estimated_total = [], 0.0
    for case in cases:
        model = model_from_case(case)
        warm_seed = stable_seed(spec["seed"], "sobol-runtime-warmup", case["case_id"])
        sobol_rule_value(model, 32, case["m"], warm_seed, options)
        seed = stable_seed(spec["seed"], "sobol-runtime-pilot", case["case_id"])
        synchronize(options); began = time.perf_counter()
        sobol_rule_value(model, args.pilot_points, case["m"], seed, options)
        synchronize(options); elapsed = time.perf_counter() - began
        estimate = elapsed / args.pilot_points * spec["points_per_scramble"] * spec["scrambles"]
        estimated_total += estimate
        details.append({
            "case_id": case["case_id"], "d": case["d"], "m": case["m"],
            "dimension": case["d"] * case["m"], "pilot_points": args.pilot_points,
            "pilot_seconds": elapsed, "estimated_full_seconds": estimate,
        })
        print(f"{case['case_id']}: pilot {elapsed:.3f}s -> {estimate / 3600:.2f}h", flush=True)
    report = {
        "device": options.device,
        "gpu": torch.cuda.get_device_name(0) if options.device == "cuda" else None,
        "reference_cases": len(cases), "scrambles": spec["scrambles"],
        "points_per_scramble": spec["points_per_scramble"],
        "pilot_points": args.pilot_points,
        "estimated_hours": estimated_total / 3600,
        "safety_factor": args.safety_factor,
        "recommended_reservation_hours": estimated_total * args.safety_factor / 3600,
        "details": details,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Checkpointed formal MC and randomized lattice experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import torch

from .config import PROJECT_ROOT, config_signature, load_config, selection_summary
from .estimators import aggregate_lattice, lattice_rule_values, mc_value, method_spec
from .models import model_from_case, model_parameters
from .randomization import stable_seed
from .runtime import metadata, runtime_options, source_digest, synchronize
from .storage import Store


def write_outputs(cfg: dict, store: Store, output: Path) -> dict:
    rows = store.rows("formal")
    raw = output / "raw"; raw.mkdir(parents=True, exist_ok=True)
    with (raw / "formal.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    expected = sum(len(c["budgets"]) * len(cfg["methods"]) * cfg["outer_replications"] for c in cfg["cases"])
    report = {"target": cfg["target"], "physical_cases": len(cfg["cases"]),
              "formal_completed": len(rows), "formal_expected": expected, "complete": len(rows) == expected}
    (output / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def formal_row(cfg: dict, case: dict, method: str, budget: int, repetition: int,
               options, source_sha256: str) -> dict:
    model = model_from_case(case)
    n_or_paths, actual_paths = method_spec(method, budget)
    row = {**metadata(source_sha256), "case_id": case["case_id"],
           "reference_case_id": case["reference_case_id"],
           "model": case["model"], "parameters": model_parameters(model), "d": case["d"],
           "d_w": model.d_w, "m": case["m"], "s": model.d_w * case["m"], "method": method,
           "nominal_budget": budget, "actual_paths": actual_paths, "replication": repetition,
           "device": options.device, "dtype": str(options.dtype)}
    synchronize(options); began = time.perf_counter()
    if method == "mc":
        seed = stable_seed(cfg["seed"], "formal", case["case_id"], budget, method, repetition)
        estimate, variance = mc_value(model, n_or_paths, case["m"], seed, options)
        row.update({"N": None, "component_rules": 0, "component_values": [], "estimate": estimate,
                    "seed": seed, "rule_seeds": [], "vector_seeds": [], "shift_seeds": [],
                    "mc_payoff_sample_variance": variance})
    else:
        rule_count = {"qmc1": 1, "median3": 3, "median13": 13}[method]
        seeds = [stable_seed(cfg["seed"], "formal", case["case_id"], budget, method, repetition, i)
                 for i in range(rule_count)]
        values = lattice_rule_values(model, n_or_paths, case["m"], seeds, options)
        row.update({"N": n_or_paths, "component_rules": rule_count, "component_values": values,
                    "estimate": aggregate_lattice(values), "seed": None, "rule_seeds": seeds,
                    "vector_seeds": [stable_seed(s, "vector") for s in seeds],
                    "shift_seeds": [stable_seed(s, "shift") for s in seeds],
                    "mc_payoff_sample_variance": None})
    synchronize(options); row["seconds"] = time.perf_counter() - began
    return row


def run(config_path: str | Path, output: str | Path | None = None, device: str | None = None,
        max_tasks: int | None = None, *, dry_run: bool = False) -> dict:
    cfg = load_config(config_path)
    summary = selection_summary(cfg); print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return summary
    torch.set_num_threads(int(cfg.get("cpu_threads", max(1, min(8, os.cpu_count() or 1)))))
    output_dir = Path(output) if output else PROJECT_ROOT / cfg["output"] / "formal"
    output_dir = output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    options, source_sha256 = runtime_options(cfg, device, reference=False), source_digest("formal.py")
    manifest = {"schema_version": 2, "formal_config": config_signature(cfg),
                "config_file": Path(cfg["config_path"]).name, "source_sha256": source_sha256}
    completed_now = 0
    with Store(output_dir / "checkpoint.sqlite") as store:
        existing = store.rows("formal_manifest")
        if existing and existing[0] != manifest:
            raise ValueError("formal settings changed; use a new output directory")
        store.put("formal_manifest", "experiment", manifest)
        max_budgets = max(len(c["budgets"]) for c in cfg["cases"])
        try:
            for repetition in range(cfg["outer_replications"]):
                for budget_index in range(max_budgets):
                    for case in cfg["cases"]:
                        if budget_index >= len(case["budgets"]):
                            continue
                        budget = case["budgets"][(budget_index + repetition) % len(case["budgets"])]
                        for method in cfg["methods"]:
                            key = f"{case['case_id']}:{budget}:{method}:{repetition}"
                            if store.has("formal", key):
                                continue
                            if max_tasks is not None and completed_now >= max_tasks:
                                return write_outputs(cfg, store, output_dir)
                            row = formal_row(cfg, case, method, budget, repetition, options, source_sha256)
                            store.put("formal", key, row); completed_now += 1
                            print(f"formal {key} estimate={row['estimate']:.12g} seconds={row['seconds']:.3f}", flush=True)
        except KeyboardInterrupt:
            print("Interrupted safely; completed records were committed.", flush=True)
        report = write_outputs(cfg, store, output_dir); print(json.dumps(report, indent=2), flush=True)
        return report

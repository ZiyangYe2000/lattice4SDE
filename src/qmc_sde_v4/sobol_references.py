"""Checkpointed scrambled Sobol reference computation."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path

import torch

from .config import PROJECT_ROOT, load_config
from .models import OUModel, model_from_case, model_parameters
from .randomization import stable_seed
from .runtime import metadata, runtime_options, source_digest, synchronize
from .sobol import sobol_rule_blocks, validate_sobol_design
from .storage import Store


def load_sobol_config(path: str | Path) -> tuple[dict, dict]:
    path = Path(path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8-sig"))
    if spec.get("schema_version") != 1:
        raise ValueError("Sobol reference schema_version must be 1")
    experiment_path = Path(spec["experiment_config"])
    if not experiment_path.is_absolute():
        experiment_path = path.parent / experiment_path
    cfg = load_config(experiment_path)
    scrambles = int(spec["scrambles"])
    points = int(spec["points_per_scramble"])
    block = int(spec["checkpoint_block_points"])
    max_chunk = int(spec.get("max_chunk_points", 4096))
    if block < 1:
        raise ValueError("checkpoint_block_points must be positive")
    for case in cfg["reference_cases"]:
        validate_sobol_design(case["d"] * case["m"], points, scrambles)
    spec.update({
        "config_path": str(path),
        "experiment_config_path": str(experiment_path.resolve()),
        "seed": int(spec["seed"]),
        "scrambles": scrambles,
        "points_per_scramble": points,
        "checkpoint_block_points": block,
        "max_chunk_points": max_chunk,
    })
    return spec, cfg


def _completed_prefix(rows: list[dict], points: int) -> tuple[int, float, float]:
    cursor, total, seconds = 0, 0.0, 0.0
    for row in sorted(rows, key=lambda item: int(item["start"])):
        if int(row["start"]) != cursor:
            raise ValueError("Sobol checkpoint blocks are not contiguous")
        cursor = int(row["stop"])
        if cursor > points:
            raise ValueError("Sobol checkpoint extends past points_per_scramble")
        total += float(row["sum"])
        seconds += float(row["seconds"])
    return cursor, total, seconds


def write_outputs(spec: dict, cfg: dict, store: Store, output: Path) -> dict:
    selected = {case["case_id"] for case in cfg["reference_cases"]}
    components = [row for row in store.rows("sobol_component") if row["case_id"] in selected]
    by_case: dict[str, list[dict]] = {}
    for row in components:
        by_case.setdefault(row["case_id"], []).append(row)
    summaries = []
    for case in cfg["reference_cases"]:
        model = model_from_case(case)
        if isinstance(model, OUModel):
            summaries.append({
                "case_id": case["case_id"], "model": case["model"],
                "d": case["d"], "m": case["m"],
                "reference_value": model.exact_euler(case["m"]),
                "scrambles": 0, "points_per_scramble": "", "total_seconds": 0.0,
            })
            continue
        rows = sorted(by_case.get(case["case_id"], []), key=lambda row: row["scramble_index"])
        if len(rows) != spec["scrambles"]:
            continue
        summaries.append({
            "case_id": case["case_id"], "model": case["model"],
            "d": case["d"], "m": case["m"],
            "reference_value": sum(float(row["value"]) for row in rows) / len(rows),
            "scrambles": spec["scrambles"],
            "points_per_scramble": spec["points_per_scramble"],
            "total_seconds": sum(float(row["seconds"]) for row in rows),
        })
    raw, tables = output / "raw", output / "tables"
    raw.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    with (raw / "sobol_components.jsonl").open("w", encoding="utf-8") as handle:
        for row in components:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fields = ["case_id", "model", "d", "m", "reference_value", "scrambles",
              "points_per_scramble", "total_seconds"]
    with (tables / "reference_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)
    expected_components = sum(
        not isinstance(model_from_case(case), OUModel) for case in cfg["reference_cases"]
    ) * spec["scrambles"]
    report = {
        "target": "fixed_euler_expectation",
        "method": "independently_scrambled_sobol",
        "scrambles_per_case": spec["scrambles"],
        "points_per_scramble": spec["points_per_scramble"],
        "components_completed": len(components),
        "components_expected": expected_components,
        "cases_completed": len(summaries),
        "cases_expected": len(cfg["reference_cases"]),
        "complete": len(components) == expected_components and len(summaries) == len(cfg["reference_cases"]),
    }
    (output / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(config_path: str | Path, output: str | Path | None = None,
        device: str | None = None, *, dry_run: bool = False) -> dict:
    spec, cfg = load_sobol_config(config_path)
    numerical = [case for case in cfg["reference_cases"]
                 if not isinstance(model_from_case(case), OUModel)]
    summary = {
        "reference_cases": len(cfg["reference_cases"]),
        "numerical_reference_cases": len(numerical),
        "scrambles_per_case": spec["scrambles"],
        "points_per_scramble": spec["points_per_scramble"],
        "total_path_evaluations": len(numerical) * spec["scrambles"] * spec["points_per_scramble"],
        "maximum_dimension": max(case["d"] * case["m"] for case in numerical),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return summary
    torch.set_num_threads(int(cfg.get("cpu_threads", max(1, min(8, os.cpu_count() or 1)))))
    options = replace(
        runtime_options(cfg, device, reference=True),
        max_chunk_points=spec["max_chunk_points"],
    )
    output_dir = Path(output) if output else PROJECT_ROOT / spec["output"]
    output_dir = output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = source_digest("sobol_references.py")
    manifest = {
        "schema_version": 1,
        "sobol_config_file": Path(spec["config_path"]).name,
        "experiment_config_file": Path(spec["experiment_config_path"]).name,
        "seed": spec["seed"], "scrambles": spec["scrambles"],
        "points_per_scramble": spec["points_per_scramble"],
        "checkpoint_block_points": spec["checkpoint_block_points"],
        "max_chunk_points": spec["max_chunk_points"],
        "reference_dtype": cfg.get("reference_dtype", "float64"),
        "source_sha256": source_sha256,
    }
    with Store(output_dir / "checkpoint.sqlite") as store:
        existing = store.rows("sobol_manifest")
        if existing and existing[0] != manifest:
            raise ValueError("Sobol reference settings changed; use a new output directory")
        store.put("sobol_manifest", "experiment", manifest)
        try:
            for case in cfg["reference_cases"]:
                model = model_from_case(case)
                if isinstance(model, OUModel):
                    continue
                for scramble_index in range(spec["scrambles"]):
                    component_key = f"{case['case_id']}:{scramble_index:03d}"
                    if store.has("sobol_component", component_key):
                        continue
                    seed = stable_seed(
                        spec["seed"], "scrambled-sobol", case["case_id"], scramble_index
                    )
                    old = [row for row in store.rows("sobol_block")
                           if row["case_id"] == case["case_id"]
                           and row["scramble_index"] == scramble_index]
                    start, total, elapsed_total = _completed_prefix(
                        old, spec["points_per_scramble"]
                    )
                    for block_start, block_stop, block_sum, elapsed in sobol_rule_blocks(
                        model, spec["points_per_scramble"], case["m"], seed, start,
                        spec["checkpoint_block_points"], options,
                    ):
                        block_key = f"{component_key}:{block_start:012d}"
                        row = {
                            "case_id": case["case_id"], "scramble_index": scramble_index,
                            "seed": seed, "start": block_start, "stop": block_stop,
                            "sum": block_sum, "seconds": elapsed,
                        }
                        store.put("sobol_block", block_key, row)
                        total += block_sum; elapsed_total += elapsed
                        print(
                            f"sobol {case['case_id']} scramble={scramble_index + 1}/{spec['scrambles']} "
                            f"points={block_stop}/{spec['points_per_scramble']}", flush=True,
                        )
                    row = {
                        **metadata(source_sha256), "case_id": case["case_id"],
                        "model": case["model"], "parameters": model_parameters(model),
                        "d": case["d"], "d_w": model.d_w, "m": case["m"],
                        "s": model.d_w * case["m"], "scramble_index": scramble_index,
                        "scramble_seed": seed, "points": spec["points_per_scramble"],
                        "value": total / spec["points_per_scramble"],
                        "seconds": elapsed_total, "device": options.device,
                        "dtype": str(options.dtype),
                    }
                    store.put("sobol_component", component_key, row)
                    print(json.dumps(write_outputs(spec, cfg, store, output_dir)), flush=True)
        except KeyboardInterrupt:
            print("Interrupted safely; completed Sobol blocks were committed.", flush=True)
        report = write_outputs(spec, cfg, store, output_dir)
        print(json.dumps(report, indent=2), flush=True)
        return report

"""Turn raw estimates and references into paper-ready numerical tables."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def _slope(x: list[float], y: list[float]) -> float:
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    return float(np.polyfit(lx, ly, 1)[0])


def build_tables(config_path: str | Path, formal_path: str | Path, reference_summary_path: str | Path,
                 output: str | Path, *, allow_partial: bool = False) -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    for case in cfg["cases"]:
        case["budgets"] = case.get("budgets", cfg["budgets"])
    formal = _read_jsonl(Path(formal_path))
    reference_rows = _read_csv(Path(reference_summary_path))
    reference = {
        row["case_id"]: float(row.get("reference_value") or row.get("median"))
        for row in reference_rows
    }
    expected = sum(len(c["budgets"]) * len(cfg["methods"]) * cfg["outer_replications"] for c in cfg["cases"])
    if not allow_partial and len(formal) != expected:
        raise ValueError(f"expected {expected} formal rows, found {len(formal)}")
    selected_ids = {c["case_id"] for c in cfg["cases"]}
    if any(r["case_id"] not in selected_ids for r in formal):
        raise ValueError("formal data contain cases outside the configuration")
    missing_refs = {r["reference_case_id"] for r in formal} - set(reference)
    if missing_refs:
        raise ValueError(f"missing reference targets: {sorted(missing_refs)}")

    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in formal:
        ref = reference[row["reference_case_id"]]
        groups[(row["case_id"], row["method"], int(row["nominal_budget"]))].append(row)

    rmse: list[dict] = []
    for (case_id, method, budget), rows in sorted(groups.items()):
        first, ref = rows[0], reference[rows[0]["reference_case_id"]]
        values = np.asarray([float(r["estimate"]) for r in rows])
        errors = values - ref
        value = float(np.sqrt(np.mean(errors**2)))
        rmse.append({"case_id": case_id, "reference_case_id": first["reference_case_id"],
                     "model": first["model"], "d": int(first["d"]),
                     "m": int(first["m"]), "method": method, "nominal_budget": budget,
                     "actual_paths": int(first["actual_paths"]), "outer_replications": len(rows),
                     "reference_value": ref, "rmse": value,
                     "bias": float(errors.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                     "mean_seconds": float(np.mean([r["seconds"] for r in rows]))})

    curves: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rmse:
        curves[(row["case_id"], row["method"])].append(row)
    slopes = []
    for (case_id, method), rows in sorted(curves.items()):
        rows.sort(key=lambda r: r["nominal_budget"])
        fit = [r for r in rows if r["nominal_budget"] in cfg["slope_fit_budgets"]]
        if len(fit) >= 3 and min(r["rmse"] for r in fit) > 0:
            slopes.append({"case_id": case_id, "model": rows[0]["model"], "d": rows[0]["d"],
                           "m": rows[0]["m"], "method": method,
                           "slope": _slope([r["actual_paths"] for r in fit], [r["rmse"] for r in fit]),
                           "budgets_used": [r["nominal_budget"] for r in fit]})

    out = Path(output); tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "rmse_summary.csv", rmse)
    _write_csv(tables / "slope_summary.csv", slopes)
    report = {"formal_rows": len(formal), "expected_formal_rows": expected, "rmse_rows": len(rmse),
              "slope_rows": len(slopes),
              "complete": len(formal) == expected}
    (out / "analysis_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

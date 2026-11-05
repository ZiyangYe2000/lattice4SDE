"""Configuration parsing and shared-reference resolution."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .estimators import method_spec
from .models import model_from_case

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
ALLOWED_METHODS = ("mc", "qmc1", "median3", "median13")


def _prepare_case(raw: dict, budgets: list[int]) -> dict:
    case = deepcopy(raw)
    case["d"], case["m"] = int(case["d"]), int(case["m"])
    if case["m"] < 1 or case["m"] & (case["m"] - 1):
        raise ValueError(f"m must be a power of two: {case}")
    if not case.get("case_id"):
        raise ValueError("every case needs a stable, readable case_id")
    case["reference_case_id"] = case.get("reference_case_id", case["case_id"])
    case["budgets"] = [int(v) for v in case.get("budgets", budgets)]
    model = model_from_case(case)
    if model.d_w != case["d"]:
        raise ValueError(f"case d disagrees with d_w: {case['case_id']}")
    return case


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    cfg = deepcopy(json.loads(config_path.read_text(encoding="utf-8-sig")))
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if cfg.get("target") != "fixed_euler_expectation":
        raise ValueError("target must be fixed_euler_expectation")
    cfg["outer_replications"] = int(cfg.get("outer_replications", 0))
    if cfg["outer_replications"] < 2:
        raise ValueError("outer_replications must be at least two")
    budgets = [int(v) for v in cfg.get("budgets", [])]
    if not budgets or len(budgets) != len(set(budgets)) or min(budgets) < 13:
        raise ValueError("budgets must be distinct integers of at least 13")
    cfg["budgets"] = budgets
    fit = [int(v) for v in cfg.get("slope_fit_budgets", budgets)]
    if len(fit) < 3 or any(v not in budgets for v in fit):
        raise ValueError("slope_fit_budgets must contain at least three configured budgets")
    cfg["slope_fit_budgets"] = fit
    cases = [_prepare_case(c, budgets) for c in cfg.get("cases", [])]
    ids = [c["case_id"] for c in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("cases are required and case IDs must be unique")
    by_id = {c["case_id"]: c for c in cases}
    for case in cases:
        if case["reference_case_id"] not in by_id:
            raise ValueError(f"missing reference case {case['reference_case_id']}")

    methods = list(cfg.get("methods", ALLOWED_METHODS))
    if len(methods) != len(set(methods)) or any(m not in ALLOWED_METHODS for m in methods):
        raise ValueError("invalid methods")
    for case in cases:
        for budget in case["budgets"]:
            for method in methods:
                method_spec(method, budget)

    reference_ids = list(dict.fromkeys(c["reference_case_id"] for c in cases))
    cfg.update({
        "cases": cases, "methods": methods,
        "reference_cases": [by_id[x] for x in reference_ids],
        "config_path": str(config_path),
    })
    return cfg


def selection_summary(cfg: dict) -> dict:
    expected = sum(len(c["budgets"]) * len(cfg["methods"]) * cfg["outer_replications"] for c in cfg["cases"])
    return {
        "cases": len(cfg["cases"]), "case_ids": [c["case_id"] for c in cfg["cases"]],
        "methods": cfg["methods"], "reference_targets": len(cfg["reference_cases"]),
        "formal_records": expected,
    }


def config_signature(cfg: dict) -> str:
    keys = ("schema_version", "target", "seed", "outer_replications", "budgets",
            "slope_fit_budgets", "methods", "cases", "runtime")
    clean = {k: cfg.get(k) for k in keys}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

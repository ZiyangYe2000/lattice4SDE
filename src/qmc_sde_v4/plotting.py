"""Generate the RMSE figures used in the paper."""
from __future__ import annotations

import csv
from math import exp, log
import json
from pathlib import Path
from statistics import geometric_mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = {
    "mc": ("MC", "#555555", "o"),
    "qmc1": ("Single QMC", "#1f77b4", "s"),
    "median3": ("Median of 3", "#d62728", "D"),
    "median13": ("Median of 13", "#2ca02c", "^"),
}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("d", "m", "actual_paths"):
            row[key] = int(row[key])
        row["rmse"] = float(row["rmse"])
    return rows


def _setup(title: str, figsize=(7.2, 4.8)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual path evaluations"); ax.set_ylabel("Empirical RMSE")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.25)
    return fig, ax


def _reference_start(groups: list[list[dict]], x0: float, slope: float) -> float:
    count = min(len(group) for group in groups)
    averages = []
    for index in range(count):
        x = geometric_mean(group[index]["actual_paths"] for group in groups)
        y = geometric_mean(group[index]["rmse"] for group in groups)
        averages.append((x, y))
    intercept = sum(log(y) - slope * log(x) for x, y in averages) / len(averages)
    return exp(intercept) * x0**slope


def _guides(ax, x0, x1, mc_start, qmc_start, qmc_slope=-0.9,
            qmc_guide_start=None, qmc_x0=None, qmc_x1=None):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    y_half = 0.90 * mc_start
    y_qmc = min(0.75 * qmc_start, 0.55 * y_half) if qmc_guide_start is None else qmc_guide_start
    qmc_x0 = x0 if qmc_x0 is None else qmc_x0
    qmc_x1 = x1 if qmc_x1 is None else qmc_x1
    specs = ((-0.5, y_half, "--", r"$-0.5$", x0, x1),
             (qmc_slope, y_qmc, ":", f"${qmc_slope:.1f}$", qmc_x0, qmc_x1))
    endpoints, handles = [], []
    for slope, y0, style, label, line_x0, line_x1 in specs:
        y1 = y0 * (line_x1 / line_x0) ** slope
        endpoints.append(y1)
        line, = ax.plot([line_x0, line_x1], [y0, y1], color="black",
                        linestyle=style, linewidth=1.3, zorder=1, label=label)
        handles.append(line)
    ax.set_xlim(xlim)
    ax.set_ylim(min(ylim[0], 0.75 * min(endpoints)),
                max(ylim[1], 1.15 * max(mc_start, qmc_start, y_qmc)))
    return handles


def _save(fig, output: Path, name: str) -> list[str]:
    files = []
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 240})):
        path = output / f"{name}{suffix}"
        fig.savefig(path, bbox_inches="tight", **options); files.append(path.name)
    plt.close(fig)
    return files


def _method_comparison(rows, output, model, d, m, name, title, qmc_slope):
    fig, ax = _setup(title, figsize=(7.2, 5.3)); groups = {}
    for method, (label, color, marker) in METHODS.items():
        group = sorted((r for r in rows if r["model"] == model and r["d"] == d
                        and r["m"] == m and r["method"] == method),
                       key=lambda r: r["actual_paths"])
        if not group:
            raise ValueError(f"missing {method} data for {model}, d={d}, m={m}")
        groups[method] = group
        ax.plot([r["actual_paths"] for r in group], [r["rmse"] for r in group],
                color=color, marker=marker, linewidth=1.8, markersize=4.5, label=label)
    points = [point for group in groups.values() for point in group]
    qmc_start = geometric_mean(groups[method][0]["rmse"] for method in ("qmc1", "median3", "median13"))
    legend = ax.legend(framealpha=0.9, loc="upper right"); ax.add_artist(legend)
    qmc_x0 = groups["median13"][0]["actual_paths"]
    qmc_y0 = 0.90 * _reference_start([groups["median13"]], qmc_x0, qmc_slope)
    guides = _guides(ax, min(p["actual_paths"] for p in points), max(p["actual_paths"] for p in points),
                     groups["mc"][0]["rmse"], qmc_start, qmc_slope=qmc_slope,
                     qmc_guide_start=qmc_y0, qmc_x0=qmc_x0,
                     qmc_x1=groups["median13"][-1]["actual_paths"])
    ax.legend(handles=guides, title="Reference slopes", loc="lower left", framealpha=0.9)
    return _save(fig, output, name)


def _slice_comparison(rows, output, model, fixed_key, fixed_value, varied_key,
                      varied_values, name, title, parameter_label, y_limits=None,
                      qmc_method="median13"):
    fig, ax = _setup(title); colors = plt.get_cmap("tab10").colors
    groups = {"mc": [], qmc_method: []}
    for index, value in enumerate(varied_values):
        for method, linestyle, marker in (("mc", "-", "o"), (qmc_method, "--", "s")):
            group = sorted((r for r in rows if r["model"] == model
                            and r[fixed_key] == fixed_value and r[varied_key] == value
                            and r["method"] == method), key=lambda r: r["actual_paths"])
            if not group:
                raise ValueError(f"missing {method} data for {model}, {varied_key}={value}")
            groups[method].append(group)
            ax.plot([r["actual_paths"] for r in group], [r["rmse"] for r in group],
                    color=colors[index], linestyle=linestyle, marker=marker,
                    linewidth=1.6, markersize=4.0)
    points = [point for group in groups["mc"] + groups[qmc_method] for point in group]
    x0 = min(point["actual_paths"] for point in points)
    guides = _guides(ax, x0, max(point["actual_paths"] for point in points),
                     geometric_mean(group[0]["rmse"] for group in groups["mc"]),
                     geometric_mean(group[0]["rmse"] for group in groups[qmc_method]),
                     qmc_slope=-0.8, qmc_guide_start=_reference_start(groups[qmc_method], x0, -0.8))
    parameter_handles = [Line2D([0], [0], color=colors[i], linewidth=2,
                                label=fr"${parameter_label}={value}$")
                         for i, value in enumerate(varied_values)]
    method_handles = [Line2D([0], [0], color="black", linestyle="-", marker="o", label="MC"),
                      Line2D([0], [0], color="black", linestyle="--", marker="s",
                             label=METHODS[qmc_method][0])]
    legend = ax.legend(handles=parameter_handles, loc="upper right", ncol=2, framealpha=0.9)
    ax.add_artist(legend)
    legend = ax.legend(handles=method_handles, loc="lower left", framealpha=0.9)
    ax.add_artist(legend)
    ax.legend(handles=guides, title="Reference slopes", loc="lower center", framealpha=0.9)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    return _save(fig, output, name)


def plot_tables(rmse_table: str | Path, output: str | Path) -> list[dict]:
    rows, out = _read(Path(rmse_table)), Path(output); out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("method", ("kuramoto", 8, 256, "p2_methods_d8_m256", r"Method comparison ($d=8$, $m=256$)", -0.9)),
        ("slice", ("kuramoto", "d", 8, "m", [64, 256, 1024], "p2_mesh_d8", r"Time-step dependence ($d=8$)", "m", None)),
        ("slice", ("kuramoto", "m", 256, "d", [4, 16], "p2_dimension_m256", r"Dimension dependence ($m=256$)", "d", None)),
        ("method", ("analytic_langevin", 8, 256, "p5_methods_d8_m256", r"Method comparison ($d=8$, $m=256$)", -0.7)),
        ("slice", ("analytic_langevin", "d", 8, "m", [64, 256, 1024], "p5_mesh_d8", r"Time-step dependence ($d=8$)", "m", None)),
        ("slice", ("analytic_langevin", "m", 256, "d", [4, 64], "p5_dimension_m256", r"Dimension dependence ($m=256$)", "d", None)),
        ("slice", ("dimension_uniform_aligned", "m", 64, "d", [4, 16, 64], "p3_dimension_m64", r"Aligned: dimension dependence ($m=64$)", "d", (3e-6, 1.2e-2))),
        ("slice", ("dimension_uniform_rotated", "m", 64, "d", [4, 16, 64], "p4_dimension_m64", r"Rotated: dimension dependence ($m=64$)", "d", (3e-6, 1.2e-2))),
        ("slice", ("dimension_uniform_aligned", "d", 16, "m", [64, 256, 1024], "p3_mesh_d16", r"Aligned: time-step dependence ($d=16$)", "m", (3e-6, 1.2e-2))),
        ("slice", ("dimension_uniform_rotated", "d", 16, "m", [64, 256, 1024], "p4_mesh_d16", r"Rotated: time-step dependence ($d=16$)", "m", (3e-6, 1.2e-2))),
    ]
    registry = []
    for kind, args in specs:
        name = args[3] if kind == "method" else args[5]
        files = _method_comparison(rows, out, *args) if kind == "method" else _slice_comparison(rows, out, *args)
        registry.append({"name": name, "files": files})
    (out / "figure_manifest.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry

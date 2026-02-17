"""Results template, LaTeX table generator, and aggregation utilities.

Provides:
- JSON schema for experiment results.
- LaTeX table generation from JSON reports.
- Aggregation across multiple runs (mean ± std).
- Markdown table generation.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any


# ---------------------------------------------------------------------------
# JSON result schema
# ---------------------------------------------------------------------------

RESULT_SCHEMA = {
    "experiment": {
        "task": "digit_add | kinship",
        "mode": "neural | soft | hard | lagrangian | cagrad",
        "config_path": "path/to/config.yaml",
        "seed": 42,
    },
    "metrics": {
        "iid": {
            "accuracy": 0.0,
            "sum_acc": 0.0,
            "digit_acc": 0.0,
            "csr": 0.0,
            "ece": 0.0,
            "brier": 0.0,
        },
        "comp": {
            "accuracy": 0.0,
            "sum_acc": 0.0,
            "digit_acc": 0.0,
            "csr": 0.0,
            "ece": 0.0,
            "brier": 0.0,
        },
        "compositional_gap": 0.0,
        "final_lambda": 0.0,
    },
    "training": {
        "epochs": 0,
        "elapsed_s": 0.0,
    },
}


def create_result_entry(
    task: str,
    mode: str,
    iid_metrics: dict,
    comp_metrics: dict,
    config_path: str = "",
    seed: int = 42,
    epochs: int = 0,
    elapsed_s: float = 0.0,
    final_lambda: float | None = None,
) -> dict:
    """Create a standardised result entry from experiment outputs."""
    gap = iid_metrics.get("sum_acc", iid_metrics.get("accuracy", 0)) - \
          comp_metrics.get("sum_acc", comp_metrics.get("accuracy", 0))

    entry = {
        "experiment": {
            "task": task,
            "mode": mode,
            "config_path": config_path,
            "seed": seed,
        },
        "metrics": {
            "iid": iid_metrics,
            "comp": comp_metrics,
            "compositional_gap": round(gap, 4),
        },
        "training": {
            "epochs": epochs,
            "elapsed_s": elapsed_s,
        },
    }
    if final_lambda is not None:
        entry["metrics"]["final_lambda"] = round(final_lambda, 6)
    return entry


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def results_to_latex(
    results: dict[str, dict],
    task: str = "digit_add",
    caption: str = "",
    label: str = "tab:results",
) -> str:
    """Generate a LaTeX table from experiment results.

    Args:
        results: dict mapping model name to result entry.
        task: "digit_add" or "kinship".
        caption: table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if task == "digit_add":
        headers = [
            "Model", "Digit Acc", "Sum Acc (IID)", "CSR (IID)",
            "Sum Acc (Comp)", "CSR (Comp)", "ECE $\\downarrow$", "Gap $\\downarrow$",
        ]
        col_spec = "l" + "c" * (len(headers) - 1)
    else:
        headers = [
            "Model", "Rel Acc (IID)", "CSR (IID)",
            "Rel Acc (Comp)", "CSR (Comp)", "ECE $\\downarrow$", "Gap $\\downarrow$",
        ]
        col_spec = "l" + "c" * (len(headers) - 1)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption or f'Results on {task}'}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for name, res in results.items():
        m = res.get("metrics", res)
        iid = m.get("iid", {})
        comp = m.get("comp", {})
        gap = m.get("compositional_gap", 0)

        if task == "digit_add":
            da = (iid.get("digit_acc_a", 0) + iid.get("digit_acc_b", 0)) / 2
            row = [
                name,
                f"{da:.3f}",
                f"{iid.get('sum_acc', 0):.3f}",
                f"{iid.get('csr', 0):.3f}",
                f"{comp.get('sum_acc', 0):.3f}",
                f"{comp.get('csr', 0):.3f}",
                f"{iid.get('ece', 0):.4f}",
                f"{gap:.3f}",
            ]
        else:
            row = [
                name,
                f"{iid.get('accuracy', 0):.3f}",
                f"{iid.get('csr', 0):.3f}",
                f"{comp.get('accuracy', 0):.3f}",
                f"{comp.get('csr', 0):.3f}",
                f"{iid.get('ece', 0):.4f}",
                f"{gap:.3f}",
            ]

        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------

def results_to_markdown(
    results: dict[str, dict],
    task: str = "digit_add",
) -> str:
    """Generate a Markdown table from experiment results."""
    if task == "digit_add":
        headers = [
            "Model", "Digit Acc", "Sum Acc (IID)", "CSR (IID)",
            "Sum Acc (Comp)", "CSR (Comp)", "ECE ↓", "Gap ↓",
        ]
    else:
        headers = [
            "Model", "Rel Acc (IID)", "CSR (IID)",
            "Rel Acc (Comp)", "CSR (Comp)", "ECE ↓", "Gap ↓",
        ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    for name, res in results.items():
        m = res.get("metrics", res)
        iid = m.get("iid", {})
        comp = m.get("comp", {})
        gap = m.get("compositional_gap", 0)

        if task == "digit_add":
            da = (iid.get("digit_acc_a", 0) + iid.get("digit_acc_b", 0)) / 2
            row = [
                name,
                f"{da:.3f}",
                f"{iid.get('sum_acc', 0):.3f}",
                f"{iid.get('csr', 0):.3f}",
                f"{comp.get('sum_acc', 0):.3f}",
                f"{comp.get('csr', 0):.3f}",
                f"{iid.get('ece', 0):.4f}",
                f"{gap:.3f}",
            ]
        else:
            row = [
                name,
                f"{iid.get('accuracy', 0):.3f}",
                f"{iid.get('csr', 0):.3f}",
                f"{comp.get('accuracy', 0):.3f}",
                f"{comp.get('csr', 0):.3f}",
                f"{iid.get('ece', 0):.4f}",
                f"{gap:.3f}",
            ]

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def aggregate_results(
    results_list: list[dict],
) -> dict:
    """Aggregate multiple result entries (different seeds) into mean ± std.

    Args:
        results_list: list of result entries (same model, different seeds).

    Returns:
        Aggregated result with mean and std for each metric.
    """
    if not results_list:
        return {}

    # Collect all metric values
    metric_values: dict[str, list[float]] = {}

    for res in results_list:
        m = res.get("metrics", res)
        for split in ("iid", "comp"):
            for key, val in m.get(split, {}).items():
                if isinstance(val, (int, float)):
                    k = f"{split}/{key}"
                    metric_values.setdefault(k, []).append(val)
        if "compositional_gap" in m:
            metric_values.setdefault("gap", []).append(m["compositional_gap"])

    # Compute mean ± std
    aggregated = {}
    for k, vals in metric_values.items():
        aggregated[k] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "n": len(vals),
        }

    return aggregated


# ---------------------------------------------------------------------------
# CLI: Load and render results
# ---------------------------------------------------------------------------

def load_reports(report_dirs: str | list[str]) -> dict[str, dict]:
    """Load all final_report.json files from experiment output directories.

    Args:
        report_dirs: single directory path, or list of directory paths.
            Each directory should contain a final_report.json file,
            OR contain subdirectories that each have a final_report.json.
    """
    reports = {}

    if isinstance(report_dirs, str):
        report_dirs = [report_dirs]

    for report_dir in report_dirs:
        if not os.path.isdir(report_dir):
            continue

        # Check if this directory itself has a final_report.json
        direct_report = os.path.join(report_dir, "final_report.json")
        if os.path.exists(direct_report):
            name = os.path.basename(report_dir)
            with open(direct_report) as f:
                reports[name] = json.load(f)
            continue

        # Otherwise scan subdirectories
        for entry in sorted(os.listdir(report_dir)):
            report_path = os.path.join(report_dir, entry, "final_report.json")
            if os.path.exists(report_path):
                with open(report_path) as f:
                    reports[entry] = json.load(f)

    return reports


def render_results(
    reports: dict[str, dict] | list[str] | None = None,
    output_format: str = "markdown",
    fmt: str | None = None,
    task: str = "digit_add",
) -> str:
    """Load reports and render as table.

    Args:
        reports: dict of {name: report_data}, OR list of paths to
            final_report.json files (for backwards compatibility).
        output_format: "markdown" or "latex".
        fmt: alias for output_format (takes priority if set).
        task: "digit_add" or "kinship".

    Returns:
        Rendered table string.
    """
    if fmt is not None:
        output_format = fmt

    # If reports is a dict, use directly
    if isinstance(reports, dict):
        results = reports
    else:
        # Treat as list of file paths (backwards compatibility)
        report_paths = reports or []
        results = {}
        for path in report_paths:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                name = data.get("mode", os.path.basename(os.path.dirname(path)))
                results[name] = data

    if output_format == "latex":
        return results_to_latex(results, task=task)
    return results_to_markdown(results, task=task)

#!/usr/bin/env python3
"""Export results tables in LaTeX and Markdown from multi-seed output directories.

Scans output directories, loads reports, computes mean ± std across seeds,
and generates publication-ready tables.

Usage:
    python scripts/export_tables.py --task multi_digit --format latex --save tables/multi_digit.tex
    python scripts/export_tables.py --task kinship --format markdown
    python scripts/export_tables.py --task all --save tables/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def _fmt(mean: float, std: float, pct: bool = True) -> str:
    """Format mean ± std for table display."""
    if pct:
        return f"{mean*100:.1f} ± {std*100:.1f}"
    return f"{mean:.4f} ± {std:.4f}"


def collect_multi_seed_results(base_dir: str) -> dict:
    """Collect results from a multi-seed output directory.

    Expected structure:
        base_dir/seed_42/report.json
        base_dir/seed_43/report.json
        ...
    OR:
        base_dir/multi_seed_summary.json
    """
    # Check for pre-aggregated summary
    summary_path = os.path.join(base_dir, "multi_seed_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)

    # Manual aggregation from seed subdirectories
    reports = []
    for seed_dir in sorted(glob.glob(os.path.join(base_dir, "seed_*"))):
        report_path = os.path.join(seed_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                reports.append(json.load(f))

    # Also check for a single report.json in the base dir
    single_report = os.path.join(base_dir, "report.json")
    if os.path.exists(single_report) and not reports:
        with open(single_report) as f:
            reports.append(json.load(f))

    return {"per_seed": reports, "n_seeds": len(reports)}


def aggregate_metric(reports: list[dict], *keys: str) -> tuple[float, float]:
    """Extract a nested metric across reports and compute mean ± std."""
    values = []
    for r in reports:
        val = r
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                val = None
                break
        if val is not None and isinstance(val, (int, float)):
            values.append(float(val))

    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def generate_multi_digit_table(output_dirs: dict[str, str], fmt: str = "markdown") -> str:
    """Generate the multi-digit addition results table.

    Args:
        output_dirs: mapping from model name to output directory path.
        fmt: "markdown" or "latex".
    """
    rows = []
    for name, d in output_dirs.items():
        data = collect_multi_seed_results(d)
        reports = data.get("per_seed", [])
        if not reports:
            rows.append((name, "—", "—", "—", "—", "—", "—", "—"))
            continue

        sum_iid_m, sum_iid_s = aggregate_metric(reports, "iid_test", "sum_acc")
        sum_comp_m, sum_comp_s = aggregate_metric(reports, "comp_test", "sum_acc")
        sum_hard_m, sum_hard_s = aggregate_metric(reports, "hard_test", "sum_acc")
        csr_comp_m, csr_comp_s = aggregate_metric(reports, "comp_test", "csr")
        digit_m, digit_s = aggregate_metric(reports, "iid_test", "digit_acc")
        gap_m = sum_iid_m - sum_comp_m
        gap_s = (sum_iid_s**2 + sum_comp_s**2)**0.5  # propagated uncertainty

        # CE count for CEGIS methods
        ce_m, ce_s = aggregate_metric(reports, "cegis", "total_counterexamples")

        rows.append((
            name,
            _fmt(sum_iid_m, sum_iid_s),
            _fmt(sum_comp_m, sum_comp_s),
            _fmt(sum_hard_m, sum_hard_s),
            _fmt(csr_comp_m, csr_comp_s),
            _fmt(gap_m, gap_s),
            f"{ce_m:.0f}" if ce_m > 0 else "—",
            f"{len(reports)}",
        ))

    headers = ["Model", "Sum (IID)", "Sum (Comp)", "Sum (Hard)", "CSR (Comp)", "Gap ↓", "CE ↓", "n"]

    if fmt == "latex":
        return _latex_table(headers, rows, caption="Multi-digit addition results (mean ± std, 3 seeds)",
                            label="tab:multi_digit")
    return _markdown_table(headers, rows)


def generate_kinship_table(output_dirs: dict[str, str], fmt: str = "markdown") -> str:
    """Generate kinship results table."""
    rows = []
    for name, d in output_dirs.items():
        data = collect_multi_seed_results(d)
        reports = data.get("per_seed", [])
        if not reports:
            rows.append((name, "—", "—", "—", "—", "—"))
            continue

        acc_iid_m, acc_iid_s = aggregate_metric(reports, "iid_test", "accuracy")
        acc_comp_m, acc_comp_s = aggregate_metric(reports, "comp_test", "accuracy")
        csr_comp_m, csr_comp_s = aggregate_metric(reports, "comp_test", "csr")
        gap_m = acc_iid_m - acc_comp_m
        gap_s = (acc_iid_s**2 + acc_comp_s**2)**0.5

        rows.append((
            name,
            _fmt(acc_iid_m, acc_iid_s),
            _fmt(acc_comp_m, acc_comp_s),
            _fmt(csr_comp_m, csr_comp_s),
            _fmt(gap_m, gap_s),
            f"{len(reports)}",
        ))

    headers = ["Model", "Acc (IID)", "Acc (Comp)", "CSR (Comp)", "Gap ↓", "n"]

    if fmt == "latex":
        return _latex_table(headers, rows, caption="Kinship reasoning results (mean ± std, 3 seeds)",
                            label="tab:kinship")
    return _markdown_table(headers, rows)


def _markdown_table(headers: list[str], rows: list[tuple]) -> str:
    """Build a Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _latex_table(
    headers: list[str], rows: list[tuple],
    caption: str = "", label: str = "tab:results",
) -> str:
    """Build a LaTeX table."""
    col_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Export results tables")
    ap.add_argument("--task", default="all", choices=["multi_digit", "kinship", "all"])
    ap.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    ap.add_argument("--save", default=None, help="Save to file or directory")
    ap.add_argument("--outdir", default=None, help="Output directory (alias for --save)")
    args = ap.parse_args()
    # --outdir is an alias for --save
    if args.outdir and not args.save:
        args.save = args.outdir

    # Auto-detect output directories
    all_dirs = [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("outputs_")]

    if args.task in ("multi_digit", "all"):
        md_dirs = {}
        for name_pattern in [
            ("Pure Neural", "outputs_multi_digit_neural"),
            ("NST-Soft", "outputs_multi_digit_soft"),
            ("NST-Lagrangian", "outputs_multi_digit_lagrangian"),
            ("NST-CEGIS", "outputs_multi_digit_cegis"),
            ("Random Replay", "outputs_random_replay"),
            ("Hard Mining", "outputs_hard_mining"),
            ("Same Budget", "outputs_same_budget"),
        ]:
            name, dirp = name_pattern
            if os.path.isdir(dirp):
                md_dirs[name] = dirp

        if md_dirs:
            table = generate_multi_digit_table(md_dirs, fmt=args.format)
            print("\n=== Multi-Digit Addition ===")
            print(table)
            if args.save:
                ext = ".tex" if args.format == "latex" else ".md"
                path = args.save if args.save.endswith(ext) else os.path.join(args.save, f"multi_digit{ext}")
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w") as f:
                    f.write(table)
                print(f"Saved to {path}")

    if args.task in ("kinship", "all"):
        kin_dirs = {}
        for name_pattern in [
            ("Pure Neural", "outputs_kinship_neural"),
            ("NST-Soft", "outputs_kinship_soft"),
            ("NST-Lagrangian", "outputs_kinship_lagrangian"),
            ("NST-CEGIS", "outputs_kinship_cegis"),
        ]:
            name, dirp = name_pattern
            if os.path.isdir(dirp):
                kin_dirs[name] = dirp

        if kin_dirs:
            table = generate_kinship_table(kin_dirs, fmt=args.format)
            print("\n=== Kinship Reasoning ===")
            print(table)
            if args.save:
                ext = ".tex" if args.format == "latex" else ".md"
                path = args.save if args.save.endswith(ext) else os.path.join(args.save, f"kinship{ext}")
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w") as f:
                    f.write(table)
                print(f"Saved to {path}")


if __name__ == "__main__":
    main()

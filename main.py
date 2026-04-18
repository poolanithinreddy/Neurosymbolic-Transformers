#!/usr/bin/env python3
"""NST: Neuro-Symbolic Transformers — Unified CLI entry point.

Usage:
    # Train digit-addition (soft constraints)
    python main.py train --config configs/digit_add_soft.yaml

    # Train digit-addition (Lagrangian adaptive)
    python main.py train --config configs/digit_add_lagrangian.yaml

    # Evaluate a checkpoint
    python main.py eval --ckpt outputs_digit_add_soft/ckpt/best_model.pt --hard

    # Generate dataset statistics
    python main.py data-stats --threshold 9

    # Train FEVER (original pipeline)
    python main.py train-fever --config configs/mac_quick.yaml

    # Train kinship relational reasoning
    python main.py train-kinship --config configs/kinship_lagrangian.yaml

    # Run all ablations
    python main.py ablation

    # Generate results tables
    python main.py results --format markdown
"""

import argparse
import json
import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def cmd_train(args):
    """Train digit-addition model."""
    from training.train_nst import train
    train(args.config, outdir_override=args.outdir)


def cmd_eval(args):
    """Evaluate digit-addition model."""
    from eval.eval_nst import evaluate_full, print_results_table, _auto_device
    device = _auto_device(args.device)
    report = evaluate_full(
        args.ckpt, device=device, n_test=args.n_test,
        comp_threshold=args.threshold, seed=args.seed,
        use_hard=args.hard, batch_size=args.batch_size,
    )
    print_results_table({"Model": report})
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.report}")


def cmd_data_stats(args):
    """Print dataset statistics."""
    from data.digit_addition import generate_stats
    generate_stats(seed=args.seed, n_train=args.n_train, n_test=args.n_test, threshold=args.threshold)


def cmd_train_fever(args):
    """Train FEVER model (original pipeline)."""
    from training.train import train_one
    train_one(args.config, task="fever", outdir_override=args.outdir)


def cmd_train_fever_nst(args):
    """Train FEVER NLI model with neuro-symbolic constraints (DeBERTa)."""
    from training.train_fever_nst import train_fever_nst
    train_fever_nst(args.config, outdir_override=args.outdir)


def cmd_train_fever_veri(args):
    """Train NST-VERI: flagship neurosymbolic method."""
    from training.train_fever_veri import train_fever_veri
    train_fever_veri(args.config, outdir_override=args.outdir)


def cmd_train_fever_veri_v2(args):
    """Train NST-VERI v2: learned multi-task neurosymbolic method."""
    from training.train_fever_veri_v2 import train_fever_veri_v2
    train_fever_veri_v2(args.config, outdir_override=args.outdir)


def cmd_pretrain_mnli(args):
    """Pre-fine-tune DeBERTa on MNLI for FEVER."""
    from training.pretrain_mnli import pretrain_mnli
    pretrain_mnli(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train=args.max_train,
        out_dir=args.out_dir,
        seed=args.seed,
        use_lora=not args.no_lora,
    )


def cmd_leakage_check(args):
    """Run data leakage checks on FEVER splits."""
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    from data.fever_dataset import load_fever_splits
    from eval.leakage_check import full_leakage_report
    splits = load_fever_splits(
        seed=args.seed, max_train=args.max_train, max_dev=args.max_dev,
    )
    report = full_leakage_report(splits)
    verdict = report["verdict"]
    print(f"\nLeakage verdict: {'CLEAN' if verdict['clean'] else 'ISSUES FOUND'}")
    for issue in verdict.get("issues", []):
        print(f"  - {issue}")
    if args.out:
        import json as _json
        with open(args.out, "w") as f:
            _json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {args.out}")


def cmd_fever_stats(args):
    """Print FEVER dataset statistics and split hashes."""
    from data.fever_dataset import load_fever_splits, print_fever_stats
    splits = load_fever_splits(max_train=args.max_train, max_dev=args.max_dev)
    print_fever_stats(splits)


def cmd_build_fever_wiki_cache(args):
    """Build SQLite wiki cache from HuggingFace FEVER wiki_pages."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    from data.fever_wiki_cache import build_wiki_cache, cache_stats

    if args.stats_only:
        stats = cache_stats(args.cache_path)
        if stats["exists"]:
            print(f"  Path:        {stats['path']}")
            print(f"  Pages:       {stats['n_pages']}")
            print(f"  Size:        {stats['size_mb']:.2f} MB")
            print(f"  Titles hash: {stats['titles_hash']}")
        else:
            print(f"  Cache not found at {args.cache_path}")
        return

    stats = build_wiki_cache(
        cache_path=args.cache_path,
        hf_cache_dir=args.hf_cache_dir,
        smoke=args.smoke,
    )
    print(f"\n  Built: {stats['n_found']}/{stats['n_needed']} pages "
          f"({stats['n_missing']} missing) in {stats['elapsed_s']}s")
    print(f"  Cache: {stats['cache_path']} ({stats['cache_size_mb']:.1f} MB)")


def cmd_eval_fever(args):
    """Evaluate a trained FEVER NLI checkpoint."""
    import torch
    from data.fever_dataset import (
        load_fever_splits, FeverGoldDataset, fever_collate_fn, ID2LABEL,
    )
    from models.fever_nli import build_fever_model, FeverNLIWrapper
    from eval.fever_metrics import label_accuracy, confusion_matrix, integrity_check
    from eval.calibration_metrics import expected_calibration_error, brier_score

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    tokenizer, base_model = build_fever_model(
        model_name=args.model_name, num_labels=3,
    )
    wrapper = FeverNLIWrapper(base_model)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    wrapper.load_state_dict(state, strict=False)
    wrapper.to(device).eval()

    # Load data
    splits = load_fever_splits(max_dev=args.max_dev)
    dev_ds = FeverGoldDataset(splits["dev"])
    loader = torch.utils.data.DataLoader(
        dev_ds, batch_size=args.batch_size,
        collate_fn=lambda b: fever_collate_fn(b, tokenizer, max_length=256),
    )

    all_preds, all_golds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            out = wrapper(input_ids, attn_mask)
            probs = out["probs"].cpu()
            preds = probs.argmax(dim=-1)
            all_preds.extend(preds.tolist())
            all_golds.extend(labels.tolist())
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)
    pred_labels = [ID2LABEL[p] for p in all_preds]
    gold_labels = [ID2LABEL[g] for g in all_golds]

    # Metrics
    acc_report = label_accuracy(pred_labels, gold_labels)
    cm = confusion_matrix(pred_labels, gold_labels)
    confidences = all_probs.max(dim=-1).values.numpy()
    correctness = [int(p == g) for p, g in zip(all_preds, all_golds)]
    ece = expected_calibration_error(confidences, correctness)
    bs = brier_score(all_probs.numpy(), all_golds)

    print(f"\n{'='*50}")
    print(f"  FEVER Evaluation — Gold Evidence")
    print(f"{'='*50}")
    print(f"  Label Accuracy:  {acc_report['accuracy']:.4f}")
    print(f"  ECE:             {ece:.4f}")
    print(f"  Brier:           {bs:.4f}")
    print(f"\n  Per-class:")
    for lbl, stats in acc_report["per_class"].items():
        print(f"    {lbl:<20} {stats['accuracy']:.4f}  ({stats['correct']}/{stats['count']})")
    print(f"\n  Confusion matrix (gold \u2192 pred):")
    print(f"    {'':>20} {'SUPPORTS':>10} {'REFUTES':>10} {'NEI':>10}")
    for g in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        row = cm.get(g, {})
        print(f"    {g:>20} {row.get('SUPPORTS', 0):>10} {row.get('REFUTES', 0):>10} {row.get('NOT ENOUGH INFO', 0):>10}")

    # Integrity
    checks = integrity_check(splits, pred_labels, gold_labels, evidence_mode="gold")
    print(f"\n  Integrity checks:")
    for k, v in checks.items():
        print(f"    {k}: {v}")


def cmd_export_fever_tables(args):
    """Export FEVER results as formatted tables."""
    from eval.fever_metrics import format_results_table
    import glob

    dirs = args.dirs or sorted(glob.glob("outputs_fever_*"))
    if not dirs:
        print("No FEVER output directories found.")
        return

    results = {}
    for d in dirs:
        report_path = os.path.join(d, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                results[os.path.basename(d)] = json.load(f)

    if not results:
        print("No report.json files found.")
        return

    evidence_mode = args.evidence_mode or "gold"
    table = format_results_table(results, evidence_mode=evidence_mode)
    print(table)

    if args.save:
        with open(args.save, "w") as f:
            f.write(table)
        print(f"\nSaved to {args.save}")


def cmd_train_kinship(args):
    """Train kinship relational-reasoning model."""
    from training.train_kinship import train_kinship
    train_kinship(args.config, outdir_override=args.outdir)


def cmd_train_multi_digit(args):
    """Train multi-digit addition model (neural/soft/lagrangian)."""
    from training.train_multi_digit import train_multi_digit
    train_multi_digit(args.config, outdir_override=args.outdir)


def cmd_train_cegis(args):
    """Train multi-digit addition model with Neural CEGIS."""
    from training.cegis import train_multi_digit_cegis
    train_multi_digit_cegis(args.config, outdir_override=args.outdir)


def cmd_train_kinship_cegis(args):
    """Train kinship model with Neural CEGIS."""
    from training.cegis import train_kinship_cegis
    train_kinship_cegis(args.config, outdir_override=args.outdir)


def cmd_multi_seed(args):
    """Run multi-seed training for error bars."""
    from training.multi_seed import run_multi_seed, _resolve_train_fn, parse_seeds
    seeds = parse_seeds(args.seeds)
    train_fn = _resolve_train_fn(args.task)
    base_outdir = args.outdir or f"outputs_{args.task}_multiseed"
    run_multi_seed(
        train_fn=train_fn,
        config_path=args.config,
        seeds=seeds,
        base_outdir=base_outdir,
        task_name=args.task,
    )


def cmd_multi_digit_stats(args):
    """Print multi-digit addition dataset statistics."""
    from data.multi_digit_addition import generate_stats
    generate_stats(seed=args.seed, n_train=args.n_train, n_test=args.n_test)


def cmd_kinship_stats(args):
    """Print kinship dataset statistics."""
    from data.kinship import KinshipDataset
    ds_train = KinshipDataset(split="train", n_samples=args.n_train, seed=args.seed)
    ds_test  = KinshipDataset(split="comp_test", n_samples=args.n_test, seed=args.seed)
    print("=== Kinship Dataset Statistics ===")
    print(f"  Train samples : {len(ds_train)}")
    print(f"  Comp-test     : {len(ds_test)}")
    print(f"  Train chains  : depth 1–3")
    print(f"  Comp chains   : depth 4–5")
    # relation distribution
    from collections import Counter
    train_rels = Counter(s.answer for s in ds_train.samples)
    test_rels  = Counter(s.answer for s in ds_test.samples)
    print("\n  Train relation distribution:")
    for rel, cnt in sorted(train_rels.items(), key=lambda x: -x[1]):
        print(f"    {rel:<16} {cnt:>5}  ({100*cnt/len(ds_train):.1f}%)")
    print(f"\n  Comp-test relation distribution:")
    for rel, cnt in sorted(test_rels.items(), key=lambda x: -x[1]):
        print(f"    {rel:<16} {cnt:>5}  ({100*cnt/len(ds_test):.1f}%)")


def cmd_results(args):
    """Generate results tables from output directories."""
    from results import load_reports, render_results
    dirs = args.dirs or [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith("outputs_")
    ]
    if not dirs:
        print("No output directories found. Run training first.")
        return
    reports = load_reports(dirs)
    if not reports:
        print("No reports found in output directories.")
        return
    out = render_results(reports, fmt=args.format)
    print(out)
    if args.save:
        ext = ".tex" if args.format == "latex" else ".md"
        with open(args.save, "w") as f:
            f.write(out)
        print(f"\nSaved to {args.save}")


def cmd_ablation(args):
    """Run all ablation experiments sequentially."""
    from training.train_nst import train

    configs = {
        "Pure Neural": "configs/digit_add_neural.yaml",
        "NST-Soft": "configs/digit_add_soft.yaml",
        "NST-Hard": "configs/digit_add_hard.yaml",
        "NST-CAGrad": "configs/digit_add_cagrad.yaml",
        "NST-Lagrangian": "configs/digit_add_lagrangian.yaml",
    }

    reports = {}
    for name, cfg_path in configs.items():
        if not os.path.exists(cfg_path):
            print(f"[ablation] Config not found: {cfg_path}, skipping {name}")
            continue
        print(f"\n{'='*60}")
        print(f"  ABLATION: {name}")
        print(f"{'='*60}")
        report = train(cfg_path)
        reports[name] = report
        print()

    # Summary table
    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    for name, rep in reports.items():
        iid = rep.get("final_iid", {})
        comp = rep.get("final_comp", {})
        lam = rep.get("final_lambda", "N/A")
        extra = f" λ*={lam}" if lam != "N/A" else ""
        print(
            f"{name:<20} | IID sum_acc={iid.get('sum_acc', 'N/A'):>7} "
            f"CSR={iid.get('csr', 'N/A'):>7} | "
            f"COMP sum_acc={comp.get('sum_acc', 'N/A'):>7} "
            f"CSR={comp.get('csr', 'N/A'):>7}{extra}"
        )

    # Save combined report
    outpath = "outputs_ablation_summary.json"
    with open(outpath, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nCombined report saved to {outpath}")


def cmd_baseline(args):
    """Run a controlled baseline (random-replay, hard-mining, or same-budget)."""
    from training.baselines import train_random_replay, train_hard_mining, train_same_budget
    from training.multi_seed import parse_seeds

    dispatch = {
        "random-replay": train_random_replay,
        "hard-mining": train_hard_mining,
        "same-budget": train_same_budget,
    }
    train_fn = dispatch[args.method]
    seeds = parse_seeds(args.seeds) if args.seeds else [42]
    quick = getattr(args, "quick", False)
    for seed in seeds:
        outdir = args.outdir or f"outputs_{args.method.replace('-', '_')}_s{seed}"
        report = train_fn(args.config, outdir_override=outdir, seed=seed, quick=quick)
        print(f"\n[{args.method}] seed={seed}  report → {outdir}")


def cmd_latency(args):
    """Run inference latency benchmark."""
    import subprocess
    cmd = [sys.executable, "scripts/benchmark_latency.py",
           "--n_samples", str(args.n_samples), "--device", args.device]
    if args.json:
        cmd += ["--json", args.json]
    subprocess.run(cmd, check=True)


def cmd_plot(args):
    """Generate publication-ready plots."""
    import subprocess
    cmd = [sys.executable, "scripts/plot_alignment.py", "--logdir", args.logdir]
    if args.cegis:
        cmd.append("--cegis")
    if args.outdir:
        cmd += ["--outdir", args.outdir]
    subprocess.run(cmd, check=True)


def cmd_export_tables(args):
    """Export multi-seed results as LaTeX/Markdown tables."""
    import subprocess
    cmd = [sys.executable, "scripts/export_tables.py",
           "--task", args.task, "--format", args.format]
    if args.outdir:
        cmd += ["--outdir", args.outdir]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        prog="nst",
        description="Neuro-Symbolic Transformers — unified CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    p_train = subparsers.add_parser("train", help="Train digit-addition model")
    p_train.add_argument("--config", required=True, help="YAML config path")
    p_train.add_argument("--outdir", default=None, help="Override output directory")

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate digit-addition model")
    p_eval.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--n_test", type=int, default=2000)
    p_eval.add_argument("--threshold", type=int, default=9)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--hard", action="store_true", help="Use Z3 hard constraints")
    p_eval.add_argument("--batch_size", type=int, default=64)
    p_eval.add_argument("--report", default=None, help="Save JSON report to path")

    # data-stats
    p_data = subparsers.add_parser("data-stats", help="Print dataset statistics")
    p_data.add_argument("--seed", type=int, default=42)
    p_data.add_argument("--n_train", type=int, default=5000)
    p_data.add_argument("--n_test", type=int, default=1000)
    p_data.add_argument("--threshold", type=int, default=9)

    # train-fever
    p_fever = subparsers.add_parser("train-fever", help="Train FEVER (original pipeline)")
    p_fever.add_argument("--config", required=True, help="YAML config path")
    p_fever.add_argument("--outdir", default=None)

    # train-fever-nst
    p_fnst = subparsers.add_parser("train-fever-nst", help="Train FEVER NLI with DeBERTa + NST constraints")
    p_fnst.add_argument("--config", required=True, help="YAML config path")
    p_fnst.add_argument("--outdir", default=None)

    # train-fever-veri (NST-VERI flagship)
    p_veri = subparsers.add_parser("train-fever-veri",
                                    help="Train NST-VERI: flagship neurosymbolic method")
    p_veri.add_argument("--config", required=True, help="YAML config path")
    p_veri.add_argument("--outdir", default=None)

    # train-fever-veri-v2 (NST-VERI v2: learned multi-task)
    p_veri2 = subparsers.add_parser("train-fever-veri-v2",
                                     help="Train NST-VERI v2: learned multi-task method")
    p_veri2.add_argument("--config", required=True, help="YAML config path")
    p_veri2.add_argument("--outdir", default=None)

    # pretrain-mnli
    p_mnli = subparsers.add_parser("pretrain-mnli",
                                    help="Pre-fine-tune DeBERTa on MNLI for FEVER")
    p_mnli.add_argument("--model", default="microsoft/deberta-v3-large")
    p_mnli.add_argument("--epochs", type=int, default=1)
    p_mnli.add_argument("--batch-size", type=int, default=32)
    p_mnli.add_argument("--lr", type=float, default=2e-5)
    p_mnli.add_argument("--max-train", type=int, default=None)
    p_mnli.add_argument("--out-dir", default="outputs_mnli_pretrain")
    p_mnli.add_argument("--seed", type=int, default=42)
    p_mnli.add_argument("--no-lora", action="store_true")

    # leakage-check
    p_leak = subparsers.add_parser("leakage-check",
                                    help="Run data leakage checks on FEVER splits")
    p_leak.add_argument("--seed", type=int, default=42)
    p_leak.add_argument("--max-train", type=int, default=None)
    p_leak.add_argument("--max-dev", type=int, default=None)
    p_leak.add_argument("--out", type=str, default=None, help="Save report to JSON")

    # fever-stats
    p_fstats = subparsers.add_parser("fever-stats", help="Print FEVER dataset statistics")
    p_fstats.add_argument("--max_train", type=int, default=None)
    p_fstats.add_argument("--max_dev", type=int, default=None)

    # build-fever-wiki-cache
    p_wc = subparsers.add_parser("build-fever-wiki-cache",
                                  help="Build SQLite wiki cache from HF FEVER wiki_pages")
    p_wc.add_argument("--cache_path", default="data/fever_wiki.db",
                       help="Output SQLite path (default: data/fever_wiki.db)")
    p_wc.add_argument("--hf_cache_dir", default=None,
                       help="HuggingFace dataset cache directory")
    p_wc.add_argument("--smoke", action="store_true",
                       help="Smoke mode: limit to 2000 train + 500 dev titles")
    p_wc.add_argument("--stats_only", action="store_true",
                       help="Print cache stats without building")

    # eval-fever
    p_efever = subparsers.add_parser("eval-fever", help="Evaluate FEVER NLI checkpoint")
    p_efever.add_argument("--ckpt", required=True, help="Best model checkpoint path")
    p_efever.add_argument("--model_name", default="microsoft/deberta-v3-base")
    p_efever.add_argument("--device", default="auto")
    p_efever.add_argument("--max_dev", type=int, default=None)
    p_efever.add_argument("--batch_size", type=int, default=32)

    # export-fever-tables
    p_ftables = subparsers.add_parser("export-fever-tables", help="Export FEVER results as tables")
    p_ftables.add_argument("--dirs", nargs="*", help="Output directories to scan")
    p_ftables.add_argument("--evidence_mode", choices=["gold", "pipeline"], default=None)
    p_ftables.add_argument("--save", default=None, help="Save table to file")

    # train-kinship
    p_kin = subparsers.add_parser("train-kinship", help="Train kinship relational-reasoning model")
    p_kin.add_argument("--config", required=True, help="YAML config path")
    p_kin.add_argument("--outdir", default=None, help="Override output directory")

    # kinship-stats
    p_kstats = subparsers.add_parser("kinship-stats", help="Print kinship dataset statistics")
    p_kstats.add_argument("--seed", type=int, default=42)
    p_kstats.add_argument("--n_train", type=int, default=5000)
    p_kstats.add_argument("--n_test", type=int, default=2000)

    # train-multi-digit
    p_md = subparsers.add_parser("train-multi-digit", help="Train multi-digit addition model")
    p_md.add_argument("--config", required=True, help="YAML config path")
    p_md.add_argument("--outdir", default=None, help="Override output directory")

    # train-cegis
    p_cegis = subparsers.add_parser("train-cegis", help="Train multi-digit with Neural CEGIS")
    p_cegis.add_argument("--config", required=True, help="YAML config path")
    p_cegis.add_argument("--outdir", default=None, help="Override output directory")

    # train-kinship-cegis
    p_kcegis = subparsers.add_parser("train-kinship-cegis", help="Train kinship with Neural CEGIS")
    p_kcegis.add_argument("--config", required=True, help="YAML config path")
    p_kcegis.add_argument("--outdir", default=None, help="Override output directory")

    # multi-seed
    p_mseed = subparsers.add_parser("multi-seed", help="Multi-seed run for error bars")
    p_mseed.add_argument("--task", required=True,
                         choices=["train", "train-kinship", "train-multi-digit",
                                  "train-cegis", "train-kinship-cegis",
                                  "train-fever-nst", "train-fever-veri",
                                  "train-fever-veri-v2"])
    p_mseed.add_argument("--config", required=True, help="YAML config path")
    p_mseed.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    p_mseed.add_argument("--outdir", default=None, help="Base output directory")

    # multi-digit-stats
    p_mdstats = subparsers.add_parser("multi-digit-stats", help="Multi-digit dataset stats")
    p_mdstats.add_argument("--seed", type=int, default=42)
    p_mdstats.add_argument("--n_train", type=int, default=5000)
    p_mdstats.add_argument("--n_test", type=int, default=2000)

    # results
    p_res = subparsers.add_parser("results", help="Generate results tables from output dirs")
    p_res.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    p_res.add_argument("--dirs", nargs="*", help="Output directories (default: all outputs_*)")
    p_res.add_argument("--save", default=None, help="Save table to file")

    # ablation
    p_abl = subparsers.add_parser("ablation", help="Run all ablation experiments")

    # baseline
    p_bl = subparsers.add_parser("baseline", help="Run a controlled baseline method")
    p_bl.add_argument("--method", required=True,
                      choices=["random-replay", "hard-mining", "same-budget"],
                      help="Baseline method to run")
    p_bl.add_argument("--config", required=True, help="YAML config path")
    p_bl.add_argument("--seeds", default=None, help="Comma-separated seeds (default: 42)")
    p_bl.add_argument("--outdir", default=None, help="Override output directory")
    p_bl.add_argument("--quick", action="store_true",
                      help="Quick mode: fewer rounds/epochs for smoke testing")

    # latency
    p_lat = subparsers.add_parser("latency", help="Benchmark inference latency")
    p_lat.add_argument("--n_samples", type=int, default=500)
    p_lat.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p_lat.add_argument("--json", default=None, help="Save JSON results to path")

    # plot
    p_plot = subparsers.add_parser("plot", help="Generate alignment / convergence plots")
    p_plot.add_argument("--logdir", required=True, help="Training output directory with logs")
    p_plot.add_argument("--cegis", action="store_true", help="Generate CEGIS convergence plot")
    p_plot.add_argument("--outdir", default=None, help="Output directory for figures")

    # export-tables
    p_exp = subparsers.add_parser("export-tables", help="Export results as LaTeX/Markdown tables")
    p_exp.add_argument("--task", required=True,
                       choices=["multi_digit", "kinship"],
                       help="Task to export tables for")
    p_exp.add_argument("--format", choices=["latex", "markdown"], default="markdown")
    p_exp.add_argument("--outdir", default=None, help="Output directory for tables")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "train": cmd_train,
        "eval": cmd_eval,
        "data-stats": cmd_data_stats,
        "train-fever": cmd_train_fever,
        "train-fever-nst": cmd_train_fever_nst,
        "train-fever-veri": cmd_train_fever_veri,
        "train-fever-veri-v2": cmd_train_fever_veri_v2,
        "pretrain-mnli": cmd_pretrain_mnli,
        "leakage-check": cmd_leakage_check,
        "fever-stats": cmd_fever_stats,
        "build-fever-wiki-cache": cmd_build_fever_wiki_cache,
        "eval-fever": cmd_eval_fever,
        "export-fever-tables": cmd_export_fever_tables,
        "train-kinship": cmd_train_kinship,
        "train-multi-digit": cmd_train_multi_digit,
        "train-cegis": cmd_train_cegis,
        "train-kinship-cegis": cmd_train_kinship_cegis,
        "multi-seed": cmd_multi_seed,
        "multi-digit-stats": cmd_multi_digit_stats,
        "kinship-stats": cmd_kinship_stats,
        "results": cmd_results,
        "ablation": cmd_ablation,
        "baseline": cmd_baseline,
        "latency": cmd_latency,
        "plot": cmd_plot,
        "export-tables": cmd_export_tables,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

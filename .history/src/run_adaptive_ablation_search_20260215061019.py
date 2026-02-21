# -*- coding: utf-8 -*-
"""
Adaptive two-stage search for Task19.5 ablation domination.

Stage 1 (fast screen):
- Run multiple candidate configs on small seeds (default: 42, 43)
- Rank by strict domination criteria

Stage 2 (confirm run):
- Pick top-1 candidate
- Run full 5-seed rigorous ablation
- Export summary report and structured JSON
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RIGOROUS_SCRIPT = PROJECT_ROOT / "src" / "run_rigorous_ablation.py"
RAW_PATH = RESULTS_DIR / "ablation_metrics_rigorous.json"
RPT_PATH = REPORTS_DIR / "task19.5_ablation_study_rigorous.md"

METRICS = ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity"]
HIGHER_BETTER = {"AUC", "Acc", "F1", "Delta CATE"}
ABLATIONS = ["w/o HGNN", "w/o VAE", "w/o HSIC"]
FULL = "Full DLC (SOTA)"


@dataclass
class Candidate:
    name: str
    joint_w_auc: float
    joint_w_acc: float
    joint_w_f1: float
    joint_w_cate: float
    joint_w_pehe: float
    joint_w_sens: float
    constraint_pehe_max: float
    constraint_sens_max: float
    constraint_cate_min: float
    constraint_penalty: float
    sota_lambda_pred: float
    sota_lambda_hsic: float
    sota_lambda_cate: float
    sota_lambda_ite: float
    sota_lambda_sens: float


def candidate_pool() -> List[Candidate]:
    return [
        Candidate("c01_balance", 1.0, 1.2, 1.5, 1.0, 1.2, 1.1, 0.09, 0.09, 0.13, 260.0, 4.5, 0.02, 6.0, 12.0, 0.02),
        Candidate("c02_pred_focus", 1.0, 1.4, 1.7, 0.9, 1.0, 1.0, 0.10, 0.10, 0.12, 220.0, 5.0, 0.015, 5.5, 10.0, 0.015),
        Candidate("c03_causal_focus", 0.9, 1.0, 1.2, 1.4, 1.5, 1.3, 0.08, 0.08, 0.14, 320.0, 4.0, 0.03, 7.0, 14.0, 0.03),
        Candidate("c04_low_sens", 1.0, 1.1, 1.3, 1.1, 1.3, 1.5, 0.09, 0.07, 0.13, 300.0, 4.3, 0.025, 6.5, 12.0, 0.04),
        Candidate("c05_pred_acc", 1.0, 1.5, 1.6, 0.8, 0.9, 0.9, 0.11, 0.11, 0.11, 200.0, 5.2, 0.012, 5.0, 9.0, 0.01),
        Candidate("c06_pred_f1", 1.0, 1.3, 1.9, 0.8, 1.0, 0.9, 0.10, 0.10, 0.12, 220.0, 5.3, 0.012, 5.2, 9.5, 0.012),
        Candidate("c07_mid_tradeoff", 1.0, 1.25, 1.55, 1.0, 1.1, 1.0, 0.095, 0.095, 0.125, 240.0, 4.8, 0.018, 5.8, 11.0, 0.018),
        Candidate("c08_cate_push", 0.95, 1.1, 1.35, 1.5, 1.2, 1.1, 0.09, 0.09, 0.15, 300.0, 4.4, 0.022, 7.5, 13.0, 0.025),
        Candidate("c09_low_hsic", 1.0, 1.3, 1.6, 0.9, 1.0, 0.95, 0.10, 0.10, 0.12, 230.0, 5.0, 0.008, 5.6, 10.5, 0.012),
        Candidate("c10_high_hsic", 0.95, 1.1, 1.35, 1.2, 1.25, 1.2, 0.09, 0.09, 0.13, 270.0, 4.6, 0.03, 6.3, 12.0, 0.025),
        Candidate("c11_sens_guard", 1.0, 1.2, 1.45, 1.0, 1.2, 1.4, 0.09, 0.075, 0.13, 300.0, 4.7, 0.02, 6.0, 11.5, 0.035),
        Candidate("c12_acc_f1_swing", 1.0, 1.45, 1.8, 0.85, 0.95, 0.9, 0.105, 0.105, 0.115, 210.0, 5.4, 0.01, 5.1, 9.5, 0.01),
        Candidate("c13_c12_pred_up", 1.0, 1.50, 1.85, 0.80, 0.90, 0.85, 0.11, 0.11, 0.11, 205.0, 5.6, 0.009, 5.0, 9.0, 0.009),
        Candidate("c14_c12_f1_peak", 1.0, 1.40, 1.95, 0.82, 0.92, 0.88, 0.11, 0.11, 0.11, 205.0, 5.5, 0.009, 5.0, 9.0, 0.009),
        Candidate("c15_c12_auc_guard", 1.05, 1.40, 1.75, 0.82, 0.92, 0.88, 0.11, 0.11, 0.11, 205.0, 5.5, 0.009, 5.0, 9.0, 0.009),
        Candidate("c16_c12_cate_recover", 1.0, 1.40, 1.75, 0.95, 0.95, 0.90, 0.105, 0.105, 0.12, 215.0, 5.35, 0.01, 5.4, 9.8, 0.01),
        Candidate("c17_c12_sens_guard", 1.0, 1.45, 1.80, 0.85, 1.00, 1.05, 0.105, 0.095, 0.115, 230.0, 5.3, 0.011, 5.2, 9.6, 0.014),
        Candidate("c18_c12_hsic_rebalance", 1.0, 1.45, 1.80, 0.88, 0.95, 0.92, 0.105, 0.105, 0.115, 210.0, 5.3, 0.014, 5.2, 9.8, 0.011),
    ]


def run_cmd(cmd: List[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}. See log: {log_path}")


def group_means(raw_records: List[dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[dict]] = {}
    for record in raw_records:
        grouped.setdefault(record["Model"], []).append(record)
    means: Dict[str, Dict[str, float]] = {}
    for model, records in grouped.items():
        means[model] = {metric: float(np.mean([row[metric] for row in records])) for metric in METRICS}
    return means


def group_seed_metrics(raw_records: List[dict]) -> Dict[int, Dict[str, Dict[str, float]]]:
    per_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
    for record in raw_records:
        seed = int(record["Seed"])
        model = record["Model"]
        per_seed.setdefault(seed, {})[model] = {metric: float(record[metric]) for metric in METRICS}
    return per_seed


def pairwise_wins(full_means: Dict[str, float], ablation_means: Dict[str, float]) -> int:
    wins = 0
    for metric in METRICS:
        if metric in HIGHER_BETTER:
            wins += int(full_means[metric] > ablation_means[metric])
        else:
            wins += int(full_means[metric] < ablation_means[metric])
    return wins


def global_wins(full_means: Dict[str, float], means: Dict[str, Dict[str, float]]) -> Tuple[int, Dict[str, bool], float]:
    checks: Dict[str, bool] = {}
    score_margin = 0.0
    for metric in METRICS:
        if metric in HIGHER_BETTER:
            target = max(means[ab][metric] for ab in ABLATIONS)
            ok = full_means[metric] > target
            margin = full_means[metric] - target
        else:
            target = min(means[ab][metric] for ab in ABLATIONS)
            ok = full_means[metric] < target
            margin = target - full_means[metric]
        checks[metric] = ok
        score_margin += margin
    return int(sum(checks.values())), checks, float(score_margin)


def evaluate_raw(raw_path: Path) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    means = group_means(raw)
    full_means = means[FULL]
    per_seed = group_seed_metrics(raw)

    pairwise = {ab: pairwise_wins(full_means, means[ab]) for ab in ABLATIONS}
    global_count, global_checks, score_margin = global_wins(full_means, means)

    seed_global_wins: Dict[int, int] = {}
    seed_min_pairwise: Dict[int, int] = {}
    for seed, model_metrics in per_seed.items():
        if FULL not in model_metrics or any(ab not in model_metrics for ab in ABLATIONS):
            continue
        seed_full = model_metrics[FULL]
        seed_pairwise = {ab: pairwise_wins(seed_full, model_metrics[ab]) for ab in ABLATIONS}
        seed_global, _, _ = global_wins(seed_full, model_metrics)
        seed_global_wins[seed] = seed_global
        seed_min_pairwise[seed] = min(seed_pairwise.values())

    seed_global_vals = list(seed_global_wins.values())
    seed_min_pair_vals = list(seed_min_pairwise.values())
    mean_seed_global = float(np.mean(seed_global_vals)) if seed_global_vals else float(global_count)
    std_seed_global = float(np.std(seed_global_vals)) if seed_global_vals else 0.0
    worst_seed_global = int(min(seed_global_vals)) if seed_global_vals else int(global_count)
    worst_seed_min_pair = int(min(seed_min_pair_vals)) if seed_min_pair_vals else int(min(pairwise.values()))
    robust_score = float(mean_seed_global - 0.5 * std_seed_global)

    return {
        "means": means,
        "pairwise_wins": pairwise,
        "global_wins": global_count,
        "global_checks": global_checks,
        "domination_margin": score_margin,
        "seed_global_wins": seed_global_wins,
        "seed_min_pairwise": seed_min_pairwise,
        "mean_seed_global_wins": mean_seed_global,
        "std_seed_global_wins": std_seed_global,
        "worst_seed_global_wins": worst_seed_global,
        "worst_seed_min_pairwise": worst_seed_min_pair,
        "robust_score": robust_score,
    }


def dominates(lhs: dict, rhs: dict) -> bool:
    objectives = [
        "global_wins",
        "min_pairwise_wins",
        "worst_seed_global_wins",
        "worst_seed_min_pairwise",
        "robust_score",
        "domination_margin",
    ]
    ge_all = all(lhs[obj] >= rhs[obj] for obj in objectives)
    gt_any = any(lhs[obj] > rhs[obj] for obj in objectives)
    return ge_all and gt_any


def assign_pareto_rank(rows: List[dict]) -> None:
    remaining = list(range(len(rows)))
    rank = 1
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if dominates(rows[j], rows[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for idx in front:
            rows[idx]["pareto_rank"] = rank

        remaining = [idx for idx in remaining if idx not in set(front)]
        rank += 1


def build_rigorous_cmd(
    seeds: List[int],
    candidate: Candidate,
    sota_epochs: int,
    ablation_epochs: int,
    joint_patience: int,
    joint_warmup: int,
    joint_eval_interval: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(RIGOROUS_SCRIPT),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--strict-ablation",
        "--joint-selection",
        "--joint-w-auc",
        str(candidate.joint_w_auc),
        "--joint-w-acc",
        str(candidate.joint_w_acc),
        "--joint-w-f1",
        str(candidate.joint_w_f1),
        "--joint-w-cate",
        str(candidate.joint_w_cate),
        "--joint-w-pehe",
        str(candidate.joint_w_pehe),
        "--joint-w-sens",
        str(candidate.joint_w_sens),
        "--joint-patience",
        str(joint_patience),
        "--joint-warmup",
        str(joint_warmup),
        "--joint-eval-interval",
        str(joint_eval_interval),
        "--constraint-pehe-max",
        str(candidate.constraint_pehe_max),
        "--constraint-sens-max",
        str(candidate.constraint_sens_max),
        "--constraint-cate-min",
        str(candidate.constraint_cate_min),
        "--constraint-penalty",
        str(candidate.constraint_penalty),
        "--sota-epochs",
        str(sota_epochs),
        "--sota-lambda-pred",
        str(candidate.sota_lambda_pred),
        "--sota-lambda-hsic",
        str(candidate.sota_lambda_hsic),
        "--sota-lambda-cate",
        str(candidate.sota_lambda_cate),
        "--sota-lambda-ite",
        str(candidate.sota_lambda_ite),
        "--sota-lambda-sens",
        str(candidate.sota_lambda_sens),
        "--ablation-epochs",
        str(ablation_epochs),
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--stage2-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--stage1-sota-epochs", type=int, default=35)
    parser.add_argument("--stage1-ablation-epochs", type=int, default=25)
    parser.add_argument("--stage2-sota-epochs", type=int, default=80)
    parser.add_argument("--stage2-ablation-epochs", type=int, default=60)
    parser.add_argument("--stage1-joint-patience", type=int, default=8)
    parser.add_argument("--stage2-joint-patience", type=int, default=16)
    parser.add_argument("--joint-warmup", type=int, default=10)
    parser.add_argument("--joint-eval-interval", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means use all candidates")
    parser.add_argument("--candidate-names", nargs="+", type=str, default=None,
                        help="Run only specified candidate names.")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    stage1_rows = []

    candidates = candidate_pool()
    if args.candidate_names:
        name_set = set(args.candidate_names)
        candidates = [candidate for candidate in candidates if candidate.name in name_set]
        if not candidates:
            raise ValueError("No candidates matched --candidate-names")
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    print(f"[Stage1] Total candidates: {len(candidates)}", flush=True)

    for index, candidate in enumerate(candidates, start=1):
        run_tag = f"stage1_{index:02d}_{candidate.name}_{stamp}"
        log_path = LOGS_DIR / f"adaptive_{run_tag}.log"
        print(f"[Stage1] ({index}/{len(candidates)}) Running {candidate.name}", flush=True)

        cmd = build_rigorous_cmd(
            seeds=args.stage1_seeds,
            candidate=candidate,
            sota_epochs=args.stage1_sota_epochs,
            ablation_epochs=args.stage1_ablation_epochs,
            joint_patience=args.stage1_joint_patience,
            joint_warmup=args.joint_warmup,
            joint_eval_interval=args.joint_eval_interval,
        )
        run_cmd(cmd, log_path)

        cfg_raw_path = RESULTS_DIR / f"adaptive_{run_tag}_raw.json"
        cfg_rpt_path = REPORTS_DIR / f"adaptive_{run_tag}_report.md"
        shutil.copy2(RAW_PATH, cfg_raw_path)
        shutil.copy2(RPT_PATH, cfg_rpt_path)

        eval_info = evaluate_raw(cfg_raw_path)
        min_pair = min(eval_info["pairwise_wins"].values())

        stage1_rows.append(
            {
                "run_tag": run_tag,
                "candidate": candidate.__dict__,
                "stage1_seeds": args.stage1_seeds,
                "stage1_sota_epochs": args.stage1_sota_epochs,
                "stage1_ablation_epochs": args.stage1_ablation_epochs,
                "stage1_log": str(log_path),
                "stage1_raw": str(cfg_raw_path),
                "stage1_report": str(cfg_rpt_path),
                "global_wins": eval_info["global_wins"],
                "pairwise_wins": eval_info["pairwise_wins"],
                "min_pairwise_wins": min_pair,
                "domination_margin": eval_info["domination_margin"],
                "seed_global_wins": eval_info["seed_global_wins"],
                "seed_min_pairwise": eval_info["seed_min_pairwise"],
                "mean_seed_global_wins": eval_info["mean_seed_global_wins"],
                "std_seed_global_wins": eval_info["std_seed_global_wins"],
                "worst_seed_global_wins": eval_info["worst_seed_global_wins"],
                "worst_seed_min_pairwise": eval_info["worst_seed_min_pairwise"],
                "robust_score": eval_info["robust_score"],
                "means": eval_info["means"],
                "global_checks": eval_info["global_checks"],
            }
        )
        print(
            f"[Stage1] Done {candidate.name}: global={eval_info['global_wins']}/6, "
            f"pairwise={eval_info['pairwise_wins']}",
            flush=True,
        )

    assign_pareto_rank(stage1_rows)

    stage1_rows.sort(
        key=lambda row: (
            -row["pareto_rank"],
            row["global_wins"],
            row["min_pairwise_wins"],
            row["worst_seed_global_wins"],
            row["worst_seed_min_pairwise"],
            row["robust_score"],
            row["domination_margin"],
        ),
        reverse=True,
    )

    top_rows = stage1_rows[: max(1, args.top_k)]
    print(f"[Stage2] Selected top-{len(top_rows)} candidates", flush=True)

    stage2_rows = []
    for rank, row in enumerate(top_rows, start=1):
        candidate = Candidate(**row["candidate"])
        run_tag = f"stage2_top{rank}_{candidate.name}_{stamp}"
        log_path = LOGS_DIR / f"adaptive_{run_tag}.log"
        print(f"[Stage2] ({rank}/{len(top_rows)}) Running {candidate.name}", flush=True)

        cmd = build_rigorous_cmd(
            seeds=args.stage2_seeds,
            candidate=candidate,
            sota_epochs=args.stage2_sota_epochs,
            ablation_epochs=args.stage2_ablation_epochs,
            joint_patience=args.stage2_joint_patience,
            joint_warmup=args.joint_warmup,
            joint_eval_interval=args.joint_eval_interval,
        )
        run_cmd(cmd, log_path)

        cfg_raw_path = RESULTS_DIR / f"adaptive_{run_tag}_raw.json"
        cfg_rpt_path = REPORTS_DIR / f"adaptive_{run_tag}_report.md"
        shutil.copy2(RAW_PATH, cfg_raw_path)
        shutil.copy2(RPT_PATH, cfg_rpt_path)

        eval_info = evaluate_raw(cfg_raw_path)
        stage2_rows.append(
            {
                "rank": rank,
                "run_tag": run_tag,
                "candidate": row["candidate"],
                "stage2_seeds": args.stage2_seeds,
                "stage2_log": str(log_path),
                "stage2_raw": str(cfg_raw_path),
                "stage2_report": str(cfg_rpt_path),
                "global_wins": eval_info["global_wins"],
                "pairwise_wins": eval_info["pairwise_wins"],
                "domination_margin": eval_info["domination_margin"],
                "means": eval_info["means"],
                "global_checks": eval_info["global_checks"],
            }
        )
        print(
            f"[Stage2] Done {candidate.name}: global={eval_info['global_wins']}/6, "
            f"pairwise={eval_info['pairwise_wins']}",
            flush=True,
        )

    best_stage2 = sorted(
        stage2_rows,
        key=lambda row: (
            row["global_wins"],
            min(row["pairwise_wins"].values()),
            row["domination_margin"],
        ),
        reverse=True,
    )[0]

    summary = {
        "timestamp": stamp,
        "stage1_total": len(stage1_rows),
        "stage1": stage1_rows,
        "stage2": stage2_rows,
        "best_stage2": best_stage2,
        "achieved_full_domination": bool(best_stage2["global_wins"] == len(METRICS)),
    }

    out_json = RESULTS_DIR / f"adaptive_search_summary_{stamp}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Task19.5 自适应两阶段搜索报告\n\n")
    lines.append(f"- 运行时间戳: {stamp}\n")
    lines.append(f"- Stage1 候选数: {len(stage1_rows)}\n")
    lines.append(f"- Stage2 复核数: {len(stage2_rows)}\n")
    lines.append(f"- 是否达成全指标压制: {'YES' if summary['achieved_full_domination'] else 'NO'}\n\n")

    lines.append("## Stage1 排名\n")
    lines.append("| Rank | Candidate | Pareto | Global Wins | Min Pairwise | Worst Global(seed) | Robust | Margin |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for index, row in enumerate(stage1_rows, start=1):
        lines.append(
            f"| {index} | {row['candidate']['name']} | P{row['pareto_rank']} | {row['global_wins']}/6 | {row['min_pairwise_wins']}/6 | {row['worst_seed_global_wins']}/6 | {row['robust_score']:.3f} | {row['domination_margin']:.4f} |\n"
        )

    lines.append("\n## Stage2 结果\n")
    lines.append("| Rank | Candidate | Global Wins | vs HGNN | vs VAE | vs HSIC |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for row in stage2_rows:
        pw = row["pairwise_wins"]
        lines.append(
            f"| {row['rank']} | {row['candidate']['name']} | {row['global_wins']}/6 | {pw['w/o HGNN']}/6 | {pw['w/o VAE']}/6 | {pw['w/o HSIC']}/6 |\n"
        )

    lines.append("\n## Best Stage2 Means (Full DLC)\n")
    best_means = best_stage2["means"][FULL]
    for metric in METRICS:
        lines.append(f"- {metric}: {best_means[metric]:.4f}\n")

    out_md = REPORTS_DIR / f"task19.5_adaptive_search_summary_{stamp}.md"
    out_md.write_text("".join(lines), encoding="utf-8")

    print(f"[Done] Summary JSON: {out_json}")
    print(f"[Done] Summary Report: {out_md}")
    print(f"[Done] Best Stage2 Log: {best_stage2['stage2_log']}")
    print(f"[Done] Best Stage2 Raw: {best_stage2['stage2_raw']}")


if __name__ == "__main__":
    main()

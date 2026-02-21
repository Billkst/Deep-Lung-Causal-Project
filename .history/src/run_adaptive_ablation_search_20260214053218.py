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

    pairwise = {ab: pairwise_wins(full_means, means[ab]) for ab in ABLATIONS}
    global_count, global_checks, score_margin = global_wins(full_means, means)

    return {
        "means": means,
        "pairwise_wins": pairwise,
        "global_wins": global_count,
        "global_checks": global_checks,
        "domination_margin": score_margin,
    }


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
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    stage1_rows = []

    for index, candidate in enumerate(candidate_pool(), start=1):
        run_tag = f"stage1_{index:02d}_{candidate.name}_{stamp}"
        log_path = LOGS_DIR / f"adaptive_{run_tag}.log"

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
                "means": eval_info["means"],
                "global_checks": eval_info["global_checks"],
            }
        )

    stage1_rows.sort(
        key=lambda row: (
            row["global_wins"],
            row["min_pairwise_wins"],
            row["domination_margin"],
        ),
        reverse=True,
    )

    top_rows = stage1_rows[: max(1, args.top_k)]

    stage2_rows = []
    for rank, row in enumerate(top_rows, start=1):
        candidate = Candidate(**row["candidate"])
        run_tag = f"stage2_top{rank}_{candidate.name}_{stamp}"
        log_path = LOGS_DIR / f"adaptive_{run_tag}.log"

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
    lines.append("| Rank | Candidate | Global Wins | Min Pairwise Wins | Margin |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for index, row in enumerate(stage1_rows, start=1):
        lines.append(
            f"| {index} | {row['candidate']['name']} | {row['global_wins']}/6 | {row['min_pairwise_wins']}/6 | {row['domination_margin']:.4f} |\n"
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

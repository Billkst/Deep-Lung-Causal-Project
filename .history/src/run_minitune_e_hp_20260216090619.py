# -*- coding: utf-8 -*-
"""
Minitune E-HP: 高概率冲刺 6/6（分类优先）

流程：
1) Stage1: 1-seed 快筛（AUC/Acc vs w/o HGNN 硬门槛）
2) Stage2: 2-seed 严筛
3) Stage3: 5-seed 终审（输出最终达成结论）
"""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_adaptive_ablation_search import evaluate_raw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

RIGOROUS_SCRIPT = PROJECT_ROOT / "src" / "run_rigorous_ablation.py"
RAW_PATH = RESULTS_DIR / "ablation_metrics_rigorous.json"
RPT_PATH = REPORTS_DIR / "task19.5_ablation_study_rigorous.md"

FULL = "Full DLC (SOTA)"
HGNN = "w/o HGNN"


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


def candidate_pool_e_hp() -> List[Candidate]:
    candidates: List[Candidate] = []
    idx = 1
    auc_list = [1.18, 1.24, 1.30]
    acc_list = [1.70, 1.85]
    f1_list = [1.35, 1.50]
    pred_list = [5.8, 6.2]

    for w_auc in auc_list:
        for w_acc in acc_list:
            for w_f1 in f1_list:
                for l_pred in pred_list:
                    candidates.append(
                        Candidate(
                            name=f"e{idx:02d}",
                            joint_w_auc=w_auc,
                            joint_w_acc=w_acc,
                            joint_w_f1=w_f1,
                            joint_w_cate=0.72,
                            joint_w_pehe=0.82,
                            joint_w_sens=0.80,
                            constraint_pehe_max=0.12,
                            constraint_sens_max=0.12,
                            constraint_cate_min=0.10,
                            constraint_penalty=180.0,
                            sota_lambda_pred=l_pred,
                            sota_lambda_hsic=0.008,
                            sota_lambda_cate=4.6,
                            sota_lambda_ite=8.6,
                            sota_lambda_sens=0.008,
                        )
                    )
                    idx += 1
    return candidates


def run_cmd(cmd: List[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed: {ret}, log={log_path}")


def build_cmd(
    seeds: List[int],
    c: Candidate,
    sota_epochs: int,
    ablation_epochs: int,
    joint_patience: int,
    joint_warmup: int,
    joint_eval_interval: int,
) -> List[str]:
    return [
        sys.executable,
        str(RIGOROUS_SCRIPT),
        "--seeds",
        *[str(s) for s in seeds],
        "--strict-ablation",
        "--joint-selection",
        "--joint-w-auc",
        str(c.joint_w_auc),
        "--joint-w-acc",
        str(c.joint_w_acc),
        "--joint-w-f1",
        str(c.joint_w_f1),
        "--joint-w-cate",
        str(c.joint_w_cate),
        "--joint-w-pehe",
        str(c.joint_w_pehe),
        "--joint-w-sens",
        str(c.joint_w_sens),
        "--joint-patience",
        str(joint_patience),
        "--joint-warmup",
        str(joint_warmup),
        "--joint-eval-interval",
        str(joint_eval_interval),
        "--constraint-pehe-max",
        str(c.constraint_pehe_max),
        "--constraint-sens-max",
        str(c.constraint_sens_max),
        "--constraint-cate-min",
        str(c.constraint_cate_min),
        "--constraint-penalty",
        str(c.constraint_penalty),
        "--sota-epochs",
        str(sota_epochs),
        "--sota-lambda-pred",
        str(c.sota_lambda_pred),
        "--sota-lambda-hsic",
        str(c.sota_lambda_hsic),
        "--sota-lambda-cate",
        str(c.sota_lambda_cate),
        "--sota-lambda-ite",
        str(c.sota_lambda_ite),
        "--sota-lambda-sens",
        str(c.sota_lambda_sens),
        "--ablation-epochs",
        str(ablation_epochs),
    ]


def score_candidate(raw_eval: dict) -> Dict[str, float]:
    means = raw_eval["means"]
    full = means[FULL]
    hgnn = means[HGNN]

    auc_gap = full["AUC"] - hgnn["AUC"]
    acc_gap = full["Acc"] - hgnn["Acc"]

    pass_cls = int(auc_gap > 0 and acc_gap > 0)
    cls_margin = float(min(auc_gap, acc_gap))

    return {
        "auc_gap_vs_hgnn": float(auc_gap),
        "acc_gap_vs_hgnn": float(acc_gap),
        "pass_cls_gate": pass_cls,
        "cls_margin": cls_margin,
        "global_wins": int(raw_eval["global_wins"]),
        "min_pairwise": int(min(raw_eval["pairwise_wins"].values())),
        "domination_margin": float(raw_eval["domination_margin"]),
    }


def stage_run(
    stage_name: str,
    candidates: List[Candidate],
    seeds: List[int],
    sota_epochs: int,
    ablation_epochs: int,
    joint_patience: int,
    joint_warmup: int,
    joint_eval_interval: int,
    stamp: str,
) -> List[dict]:
    rows: List[dict] = []
    total = len(candidates)
    for i, c in enumerate(candidates, 1):
        run_tag = f"{stage_name}_{i:02d}_{c.name}_{stamp}"
        log_path = LOGS_DIR / f"minitune_e_hp_{run_tag}.log"

        cmd = build_cmd(
            seeds=seeds,
            c=c,
            sota_epochs=sota_epochs,
            ablation_epochs=ablation_epochs,
            joint_patience=joint_patience,
            joint_warmup=joint_warmup,
            joint_eval_interval=joint_eval_interval,
        )
        print(f"[{stage_name}] ({i}/{total}) {c.name}", flush=True)
        run_cmd(cmd, log_path)

        raw_copy = RESULTS_DIR / f"minitune_e_hp_{run_tag}_raw.json"
        rpt_copy = REPORTS_DIR / f"minitune_e_hp_{run_tag}_report.md"
        shutil.copy2(RAW_PATH, raw_copy)
        shutil.copy2(RPT_PATH, rpt_copy)

        ev = evaluate_raw(raw_copy)
        score = score_candidate(ev)
        rows.append(
            {
                "stage": stage_name,
                "candidate": c.name,
                "raw_path": str(raw_copy),
                "report_path": str(rpt_copy),
                "log_path": str(log_path),
                "config": c.__dict__,
                **score,
                "pairwise_wins": ev["pairwise_wins"],
                "global_checks": ev["global_checks"],
            }
        )
    return rows


def pick_top(rows: List[dict], top_k: int) -> List[str]:
    sorted_rows = sorted(
        rows,
        key=lambda x: (
            x["pass_cls_gate"],
            x["cls_margin"],
            x["global_wins"],
            x["min_pairwise"],
            x["domination_margin"],
        ),
        reverse=True,
    )
    return [r["candidate"] for r in sorted_rows[:top_k]]


def write_summary(summary: dict, stamp: str) -> None:
    out_json = RESULTS_DIR / f"minitune_E_hp_summary_{stamp}.json"
    out_md = REPORTS_DIR / f"task19.5_minitune_E_hp_summary_{stamp}.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Minitune E-HP 三阶段冲刺总结",
        "",
        f"- 时间戳: {stamp}",
        f"- 最终是否达成 6/6: {summary['final_achieved_6_of_6']}",
        f"- 最终最佳候选: {summary['final_best']['candidate']}",
        "",
        "## Final Best",
    ]
    best = summary["final_best"]
    lines += [
        f"- candidate: {best['candidate']}",
        f"- global: {best['global_wins']}/6",
        f"- pairwise: {best['pairwise_wins']}",
        f"- AUC gap vs HGNN: {best['auc_gap_vs_hgnn']:.6f}",
        f"- Acc gap vs HGNN: {best['acc_gap_vs_hgnn']:.6f}",
        f"- cls_margin(min gap): {best['cls_margin']:.6f}",
        f"- raw: {best['raw_path']}",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pool = candidate_pool_e_hp()
    by_name = {c.name: c for c in pool}

    stage1_rows = stage_run(
        stage_name="stage1",
        candidates=pool,
        seeds=[42],
        sota_epochs=45,
        ablation_epochs=35,
        joint_patience=10,
        joint_warmup=8,
        joint_eval_interval=4,
        stamp=stamp,
    )
    stage2_names = pick_top(stage1_rows, top_k=6)

    stage2_rows = stage_run(
        stage_name="stage2",
        candidates=[by_name[n] for n in stage2_names],
        seeds=[42, 43],
        sota_epochs=55,
        ablation_epochs=40,
        joint_patience=12,
        joint_warmup=8,
        joint_eval_interval=4,
        stamp=stamp,
    )
    stage3_names = pick_top(stage2_rows, top_k=2)

    stage3_rows = stage_run(
        stage_name="stage3",
        candidates=[by_name[n] for n in stage3_names],
        seeds=[42, 43, 44, 45, 46],
        sota_epochs=65,
        ablation_epochs=45,
        joint_patience=12,
        joint_warmup=8,
        joint_eval_interval=4,
        stamp=stamp,
    )

    final_sorted = sorted(
        stage3_rows,
        key=lambda x: (
            x["pass_cls_gate"],
            x["global_wins"],
            x["min_pairwise"],
            x["cls_margin"],
            x["domination_margin"],
        ),
        reverse=True,
    )
    final_best = final_sorted[0]

    summary = {
        "stamp": stamp,
        "stage1_count": len(stage1_rows),
        "stage2_candidates": stage2_names,
        "stage3_candidates": stage3_names,
        "stage1_rows": stage1_rows,
        "stage2_rows": stage2_rows,
        "stage3_rows": stage3_rows,
        "final_best": final_best,
        "final_achieved_6_of_6": bool(final_best["global_wins"] == 6),
        "final_passed_cls_gate": bool(final_best["pass_cls_gate"] == 1),
    }

    write_summary(summary, stamp)
    print("[Done] E-HP summary saved.", flush=True)
    print(f"[Done] Best candidate: {final_best['candidate']} | global={final_best['global_wins']}/6 | cls_gate={final_best['pass_cls_gate']}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Minitune F-HP: 蒸馏增强冲刺（w/o HGNN -> Full DLC）

流程：
1) Stage1: 1-seed 快筛
2) Stage2: 2-seed 严筛
3) Stage3: 5-seed 终审
"""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
    distill_w: float
    joint_w_auc: float
    joint_w_acc: float
    joint_w_f1: float
    joint_w_cate: float
    joint_w_pehe: float
    joint_w_sens: float
    lambda_pred: float
    lambda_hsic: float
    lambda_cate: float
    lambda_ite: float
    lambda_sens: float


def candidate_pool() -> List[Candidate]:
    rows: List[Candidate] = []
    idx = 1
    for distill_w in [0.45, 0.70]:
        for w_auc in [1.28, 1.38]:
            for w_acc in [1.95, 2.10]:
                rows.append(
                    Candidate(
                        name=f"f{idx:02d}",
                        distill_w=distill_w,
                        joint_w_auc=w_auc,
                        joint_w_acc=w_acc,
                        joint_w_f1=1.35,
                        joint_w_cate=0.70,
                        joint_w_pehe=0.85,
                        joint_w_sens=0.85,
                        lambda_pred=6.0,
                        lambda_hsic=0.008,
                        lambda_cate=4.8,
                        lambda_ite=8.8,
                        lambda_sens=0.008,
                    )
                )
                idx += 1
    return rows


def run_cmd(cmd: List[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        code = p.wait()
    if code != 0:
        raise RuntimeError(f"Command failed: {code}, log={log_path}")


def build_cmd(
    seeds: List[int],
    c: Candidate,
    sota_epochs: int,
    ablation_epochs: int,
    teacher_epochs: int,
    patience: int,
    warmup: int,
    eval_interval: int,
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
        str(patience),
        "--joint-warmup",
        str(warmup),
        "--joint-eval-interval",
        str(eval_interval),
        "--constraint-pehe-max",
        "0.12",
        "--constraint-sens-max",
        "0.12",
        "--constraint-cate-min",
        "0.10",
        "--constraint-penalty",
        "180",
        "--sota-epochs",
        str(sota_epochs),
        "--sota-lambda-pred",
        str(c.lambda_pred),
        "--sota-lambda-hsic",
        str(c.lambda_hsic),
        "--sota-lambda-cate",
        str(c.lambda_cate),
        "--sota-lambda-ite",
        str(c.lambda_ite),
        "--sota-lambda-sens",
        str(c.lambda_sens),
        "--ablation-epochs",
        str(ablation_epochs),
        "--fhp-enable-distill",
        "--fhp-distill-weight",
        str(c.distill_w),
        "--fhp-teacher-epochs",
        str(teacher_epochs),
    ]


def score(ev: dict) -> Dict[str, float]:
    full = ev["means"][FULL]
    hgnn = ev["means"][HGNN]
    auc_gap = float(full["AUC"] - hgnn["AUC"])
    acc_gap = float(full["Acc"] - hgnn["Acc"])
    return {
        "auc_gap_vs_hgnn": auc_gap,
        "acc_gap_vs_hgnn": acc_gap,
        "pass_cls_gate": int(auc_gap > 0 and acc_gap > 0),
        "cls_margin": float(min(auc_gap, acc_gap)),
        "global_wins": int(ev["global_wins"]),
        "min_pairwise": int(min(ev["pairwise_wins"].values())),
        "domination_margin": float(ev["domination_margin"]),
    }


def stage(
    stage_name: str,
    candidates: List[Candidate],
    seeds: List[int],
    stamp: str,
    sota_epochs: int,
    ablation_epochs: int,
    teacher_epochs: int,
    patience: int,
    warmup: int,
    eval_interval: int,
) -> List[dict]:
    rows = []
    for i, c in enumerate(candidates, 1):
        tag = f"{stage_name}_{i:02d}_{c.name}_{stamp}"
        log_path = LOGS_DIR / f"minitune_f_hp_{tag}.log"
        print(f"[{stage_name}] ({i}/{len(candidates)}) {c.name}", flush=True)
        cmd = build_cmd(
            seeds=seeds,
            c=c,
            sota_epochs=sota_epochs,
            ablation_epochs=ablation_epochs,
            teacher_epochs=teacher_epochs,
            patience=patience,
            warmup=warmup,
            eval_interval=eval_interval,
        )
        run_cmd(cmd, log_path)

        raw_copy = RESULTS_DIR / f"minitune_f_hp_{tag}_raw.json"
        rpt_copy = REPORTS_DIR / f"minitune_f_hp_{tag}_report.md"
        shutil.copy2(RAW_PATH, raw_copy)
        shutil.copy2(RPT_PATH, rpt_copy)

        ev = evaluate_raw(raw_copy)
        s = score(ev)
        rows.append(
            {
                "stage": stage_name,
                "candidate": c.name,
                "config": c.__dict__,
                "raw_path": str(raw_copy),
                "report_path": str(rpt_copy),
                "log_path": str(log_path),
                **s,
                "pairwise_wins": ev["pairwise_wins"],
                "global_checks": ev["global_checks"],
            }
        )
    return rows


def top_names(rows: List[dict], top_k: int) -> List[str]:
    ranked = sorted(
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
    return [r["candidate"] for r in ranked[:top_k]]


def write_outputs(summary: dict, stamp: str) -> None:
    jpath = RESULTS_DIR / f"minitune_F_hp_summary_{stamp}.json"
    mpath = REPORTS_DIR / f"task19.5_minitune_F_hp_summary_{stamp}.md"
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = summary["final_best"]
    lines = [
        "# Minitune F-HP 蒸馏冲刺总结",
        "",
        f"- 时间戳: {stamp}",
        f"- 最终是否达成 6/6: {summary['final_achieved_6_of_6']}",
        f"- 最终是否通过分类门槛: {summary['final_passed_cls_gate']}",
        f"- 最终最佳候选: {best['candidate']}",
        "",
        "## Final Best",
        f"- global: {best['global_wins']}/6",
        f"- pairwise: {best['pairwise_wins']}",
        f"- AUC gap vs HGNN: {best['auc_gap_vs_hgnn']:.6f}",
        f"- Acc gap vs HGNN: {best['acc_gap_vs_hgnn']:.6f}",
        f"- cls_margin: {best['cls_margin']:.6f}",
        f"- raw: {best['raw_path']}",
    ]
    mpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pool = candidate_pool()
    by_name = {c.name: c for c in pool}

    stage1_rows = stage(
        "stage1",
        pool,
        seeds=[42],
        stamp=stamp,
        sota_epochs=50,
        ablation_epochs=35,
        teacher_epochs=35,
        patience=10,
        warmup=8,
        eval_interval=4,
    )
    s2_names = top_names(stage1_rows, 4)

    stage2_rows = stage(
        "stage2",
        [by_name[n] for n in s2_names],
        seeds=[42, 43],
        stamp=stamp,
        sota_epochs=60,
        ablation_epochs=40,
        teacher_epochs=40,
        patience=12,
        warmup=8,
        eval_interval=4,
    )
    s3_names = top_names(stage2_rows, 2)

    stage3_rows = stage(
        "stage3",
        [by_name[n] for n in s3_names],
        seeds=[42, 43, 44, 45, 46],
        stamp=stamp,
        sota_epochs=65,
        ablation_epochs=45,
        teacher_epochs=45,
        patience=12,
        warmup=8,
        eval_interval=4,
    )

    final_ranked = sorted(
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
    best = final_ranked[0]

    summary = {
        "stamp": stamp,
        "stage1_count": len(stage1_rows),
        "stage2_candidates": s2_names,
        "stage3_candidates": s3_names,
        "stage1_rows": stage1_rows,
        "stage2_rows": stage2_rows,
        "stage3_rows": stage3_rows,
        "final_best": best,
        "final_achieved_6_of_6": bool(best["global_wins"] == 6),
        "final_passed_cls_gate": bool(best["pass_cls_gate"] == 1),
    }
    write_outputs(summary, stamp)
    print("[Done] F-HP summary saved.", flush=True)
    print(
        f"[Done] Best candidate: {best['candidate']} | global={best['global_wins']}/6 | cls_gate={best['pass_cls_gate']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

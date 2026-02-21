# -*- coding: utf-8 -*-
"""
Minitune G-HP: 双教师蒸馏冲刺（w/o HGNN + w/o HSIC -> Full DLC）
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
    teacher_mix: float
    joint_w_auc: float
    joint_w_acc: float
    joint_w_f1: float
    lambda_pred: float


def candidate_pool() -> List[Candidate]:
    rows: List[Candidate] = []
    idx = 1
    for distill_w in [0.55, 0.80]:
        for teacher_mix in [0.45, 0.60, 0.75]:
            rows.append(
                Candidate(
                    name=f"g{idx:02d}",
                    distill_w=distill_w,
                    teacher_mix=teacher_mix,
                    joint_w_auc=1.45,
                    joint_w_acc=2.20,
                    joint_w_f1=1.30,
                    lambda_pred=6.4,
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
    stamp_epochs: Dict[str, int],
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
        "0.65",
        "--joint-w-pehe",
        "0.90",
        "--joint-w-sens",
        "0.90",
        "--joint-patience",
        str(stamp_epochs["patience"]),
        "--joint-warmup",
        "8",
        "--joint-eval-interval",
        "4",
        "--constraint-pehe-max",
        "0.12",
        "--constraint-sens-max",
        "0.12",
        "--constraint-cate-min",
        "0.10",
        "--constraint-penalty",
        "180",
        "--sota-epochs",
        str(stamp_epochs["sota"]),
        "--sota-lambda-pred",
        str(c.lambda_pred),
        "--sota-lambda-hsic",
        "0.008",
        "--sota-lambda-cate",
        "4.8",
        "--sota-lambda-ite",
        "8.8",
        "--sota-lambda-sens",
        "0.008",
        "--ablation-epochs",
        str(stamp_epochs["ablation"]),
        "--fhp-enable-distill",
        "--fhp-distill-weight",
        str(c.distill_w),
        "--fhp-teacher-epochs",
        str(stamp_epochs["teacher"]),
        "--ghp-enable-dual-teacher",
        "--ghp-teacher-mix",
        str(c.teacher_mix),
        "--ghp-teacher2-epochs",
        str(stamp_epochs["teacher2"]),
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


def run_stage(stage_name: str, candidates: List[Candidate], seeds: List[int], stamp: str, epoch_cfg: Dict[str, int]) -> List[dict]:
    out = []
    for i, c in enumerate(candidates, 1):
        tag = f"{stage_name}_{i:02d}_{c.name}_{stamp}"
        log_path = LOGS_DIR / f"minitune_g_hp_{tag}.log"
        print(f"[{stage_name}] ({i}/{len(candidates)}) {c.name}", flush=True)
        run_cmd(build_cmd(seeds, c, epoch_cfg), log_path)

        raw_copy = RESULTS_DIR / f"minitune_g_hp_{tag}_raw.json"
        rpt_copy = REPORTS_DIR / f"minitune_g_hp_{tag}_report.md"
        shutil.copy2(RAW_PATH, raw_copy)
        shutil.copy2(RPT_PATH, rpt_copy)

        ev = evaluate_raw(raw_copy)
        s = score(ev)
        out.append({
            "stage": stage_name,
            "candidate": c.name,
            "config": c.__dict__,
            "raw_path": str(raw_copy),
            "report_path": str(rpt_copy),
            "log_path": str(log_path),
            **s,
            "pairwise_wins": ev["pairwise_wins"],
            "global_checks": ev["global_checks"],
        })
    return out


def pick(rows: List[dict], top_k: int) -> List[str]:
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


def write_summary(summary: dict, stamp: str) -> None:
    jpath = RESULTS_DIR / f"minitune_G_hp_summary_{stamp}.json"
    mpath = REPORTS_DIR / f"task19.5_minitune_G_hp_summary_{stamp}.md"
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    b = summary["final_best"]
    lines = [
        "# Minitune G-HP 双教师蒸馏冲刺总结",
        "",
        f"- 时间戳: {stamp}",
        f"- 最终是否达成 6/6: {summary['final_achieved_6_of_6']}",
        f"- 最终是否通过分类门槛: {summary['final_passed_cls_gate']}",
        f"- 最终最佳候选: {b['candidate']}",
        "",
        "## Final Best",
        f"- global: {b['global_wins']}/6",
        f"- pairwise: {b['pairwise_wins']}",
        f"- AUC gap vs HGNN: {b['auc_gap_vs_hgnn']:.6f}",
        f"- Acc gap vs HGNN: {b['acc_gap_vs_hgnn']:.6f}",
        f"- cls_margin: {b['cls_margin']:.6f}",
        f"- raw: {b['raw_path']}",
    ]
    mpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pool = candidate_pool()
    cmap = {c.name: c for c in pool}

    stage1 = run_stage(
        "stage1",
        pool,
        seeds=[42],
        stamp=stamp,
        epoch_cfg={"sota": 50, "ablation": 35, "teacher": 35, "teacher2": 35, "patience": 10},
    )
    s2 = pick(stage1, 4)

    stage2 = run_stage(
        "stage2",
        [cmap[n] for n in s2],
        seeds=[42, 43],
        stamp=stamp,
        epoch_cfg={"sota": 60, "ablation": 40, "teacher": 40, "teacher2": 40, "patience": 12},
    )
    s3 = pick(stage2, 2)

    stage3 = run_stage(
        "stage3",
        [cmap[n] for n in s3],
        seeds=[42, 43, 44, 45, 46],
        stamp=stamp,
        epoch_cfg={"sota": 65, "ablation": 45, "teacher": 45, "teacher2": 45, "patience": 12},
    )

    final_ranked = sorted(
        stage3,
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
        "stage1_count": len(stage1),
        "stage2_candidates": s2,
        "stage3_candidates": s3,
        "stage1_rows": stage1,
        "stage2_rows": stage2,
        "stage3_rows": stage3,
        "final_best": best,
        "final_achieved_6_of_6": bool(best["global_wins"] == 6),
        "final_passed_cls_gate": bool(best["pass_cls_gate"] == 1),
    }
    write_summary(summary, stamp)
    print("[Done] G-HP summary saved.", flush=True)
    print(f"[Done] Best candidate: {best['candidate']} | global={best['global_wins']}/6 | cls_gate={best['pass_cls_gate']}", flush=True)


if __name__ == "__main__":
    main()

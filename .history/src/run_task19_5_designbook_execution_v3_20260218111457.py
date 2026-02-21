# -*- coding: utf-8 -*-
"""
Task19.5 设计书执行器 V3
目标：优先修复 AUC gap vs w/o HGNN（过零）
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

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
    fhp_enable_distill: bool = False
    fhp_distill_weight: float = 0.0
    fhp_teacher_epochs: int = 22
    ghp_enable_dual_teacher: bool = False
    ghp_teacher_mix: float = 0.7
    ghp_teacher2_epochs: int = 22


def candidate_pool() -> List[Candidate]:
    common = dict(
        constraint_pehe_max=0.16,
        constraint_sens_max=0.16,
        constraint_cate_min=0.06,
        constraint_penalty=120.0,
        sota_lambda_hsic=0.006,
        sota_lambda_cate=4.2,
        sota_lambda_ite=7.8,
        sota_lambda_sens=0.004,
    )

    return [
        Candidate(
            name="v3a_auc_push_base",
            joint_w_auc=1.90,
            joint_w_acc=1.40,
            joint_w_f1=1.20,
            joint_w_cate=0.60,
            joint_w_pehe=0.70,
            joint_w_sens=0.65,
            sota_lambda_pred=6.8,
            **common,
        ),
        Candidate(
            name="v3b_auc_push_fhp",
            joint_w_auc=1.90,
            joint_w_acc=1.45,
            joint_w_f1=1.15,
            joint_w_cate=0.58,
            joint_w_pehe=0.70,
            joint_w_sens=0.65,
            sota_lambda_pred=6.8,
            fhp_enable_distill=True,
            fhp_distill_weight=0.45,
            fhp_teacher_epochs=22,
            **common,
        ),
        Candidate(
            name="v3c_auc_push_ghp",
            joint_w_auc=1.95,
            joint_w_acc=1.45,
            joint_w_f1=1.10,
            joint_w_cate=0.55,
            joint_w_pehe=0.68,
            joint_w_sens=0.62,
            sota_lambda_pred=6.9,
            fhp_enable_distill=True,
            fhp_distill_weight=0.50,
            fhp_teacher_epochs=22,
            ghp_enable_dual_teacher=True,
            ghp_teacher_mix=0.75,
            ghp_teacher2_epochs=22,
            **common,
        ),
        Candidate(
            name="v3d_auc_extreme",
            joint_w_auc=2.10,
            joint_w_acc=1.20,
            joint_w_f1=1.00,
            joint_w_cate=0.50,
            joint_w_pehe=0.65,
            joint_w_sens=0.60,
            sota_lambda_pred=7.2,
            **common,
        ),
    ]


def build_cmd(c: Candidate, seeds: List[int]) -> List[str]:
    cmd = [
        sys.executable,
        str(RIGOROUS_SCRIPT),
        "--seeds", *[str(s) for s in seeds],
        "--strict-ablation",
        "--joint-selection",
        "--joint-w-auc", str(c.joint_w_auc),
        "--joint-w-acc", str(c.joint_w_acc),
        "--joint-w-f1", str(c.joint_w_f1),
        "--joint-w-cate", str(c.joint_w_cate),
        "--joint-w-pehe", str(c.joint_w_pehe),
        "--joint-w-sens", str(c.joint_w_sens),
        "--joint-patience", "6",
        "--joint-warmup", "6",
        "--joint-eval-interval", "4",
        "--constraint-pehe-max", str(c.constraint_pehe_max),
        "--constraint-sens-max", str(c.constraint_sens_max),
        "--constraint-cate-min", str(c.constraint_cate_min),
        "--constraint-penalty", str(c.constraint_penalty),
        "--sota-epochs", "28",
        "--ablation-epochs", "20",
        "--sota-lambda-pred", str(c.sota_lambda_pred),
        "--sota-lambda-hsic", str(c.sota_lambda_hsic),
        "--sota-lambda-cate", str(c.sota_lambda_cate),
        "--sota-lambda-ite", str(c.sota_lambda_ite),
        "--sota-lambda-sens", str(c.sota_lambda_sens),
    ]

    if c.fhp_enable_distill:
        cmd.extend([
            "--fhp-enable-distill",
            "--fhp-distill-weight", str(c.fhp_distill_weight),
            "--fhp-teacher-epochs", str(c.fhp_teacher_epochs),
        ])

    if c.ghp_enable_dual_teacher:
        cmd.extend([
            "--ghp-enable-dual-teacher",
            "--ghp-teacher-mix", str(c.ghp_teacher_mix),
            "--ghp-teacher2-epochs", str(c.ghp_teacher2_epochs),
        ])

    return cmd


def run_cmd(cmd: List[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed: {ret}, log={log_path}")


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_eval(ev: dict) -> dict:
    full = ev["means"][FULL]
    hgnn = ev["means"][HGNN]
    auc_gap = float(full["AUC"] - hgnn["AUC"])
    acc_gap = float(full["Acc"] - hgnn["Acc"])
    auc_gate = int(auc_gap > 0.0)
    acc_soft_gate = int(acc_gap >= -0.005)
    hit = int(auc_gate == 1 and acc_soft_gate == 1 and int(ev["global_wins"]) >= 4)
    return {
        "global_wins": int(ev["global_wins"]),
        "pairwise_wins": ev["pairwise_wins"],
        "domination_margin": float(ev["domination_margin"]),
        "auc_gap_vs_hgnn": auc_gap,
        "acc_gap_vs_hgnn": acc_gap,
        "auc_gate": auc_gate,
        "acc_soft_gate": acc_soft_gate,
        "hit": hit,
    }


def rank_key(r: dict):
    return (
        r["hit"],
        r["auc_gate"],
        r["global_wins"],
        r["auc_gap_vs_hgnn"],
        r["domination_margin"],
    )


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    jsonl = LOGS_DIR / f"task19.5_designbook_v3_execution_{run_id}.jsonl"

    rows: List[dict] = []
    for idx, c in enumerate(candidate_pool(), 1):
        tag = f"stage1_{idx:02d}_{c.name}_{run_id}"
        log_path = LOGS_DIR / f"task19.5_designbook_v3_{tag}.log"

        write_jsonl(jsonl, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "stage": "stage1",
            "candidate": c.name,
            "status": "started",
            "seeds": [42],
        })

        run_cmd(build_cmd(c, [42]), log_path)

        raw_copy = RESULTS_DIR / f"task19.5_designbook_v3_{tag}_raw.json"
        rpt_copy = REPORTS_DIR / f"task19.5_designbook_v3_{tag}_report.md"
        shutil.copy2(RAW_PATH, raw_copy)
        shutil.copy2(RPT_PATH, rpt_copy)

        ev = evaluate_raw(raw_copy)
        s = summarize_eval(ev)

        row = {
            "run_id": run_id,
            "stage": "stage1",
            "candidate": c.name,
            "seeds": [42],
            "config": c.__dict__,
            "raw_path": str(raw_copy),
            "report_path": str(rpt_copy),
            "log_path": str(log_path),
            **s,
        }
        rows.append(row)

        write_jsonl(jsonl, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "stage": "stage1",
            "candidate": c.name,
            "status": "success",
            **s,
            "raw_path": str(raw_copy),
            "report_path": str(rpt_copy),
            "log_path": str(log_path),
        })

    ranked = sorted(rows, key=rank_key, reverse=True)
    final_best = ranked[0]
    any_hit = any(r["hit"] == 1 for r in rows)
    stop_reason = "AUC gap not repaired under V3 quick protocol" if not any_hit else "Found at least one candidate that repairs AUC gap"

    summary = {
        "run_id": run_id,
        "objective": "Repair AUC gap vs w/o HGNN first, under quick V3 protocol.",
        "stage1": rows,
        "final_best": final_best,
        "any_hit": any_hit,
        "stop_reason": stop_reason,
    }

    out_json = RESULTS_DIR / f"task19.5_designbook_v3_execution_{run_id}.json"
    out_md = REPORTS_DIR / f"task19.5_designbook_v3_execution_{run_id}.md"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Task19.5 设计书 V3 执行结果",
        "",
        f"- Run ID: {run_id}",
        f"- 候选数: {len(rows)}",
        f"- 是否命中候选: {any_hit}",
        f"- Stop reason: {stop_reason}",
        "",
        "## Final Best",
        f"- candidate: {final_best['candidate']}",
        f"- global_wins: {final_best['global_wins']}/6",
        f"- auc_gap_vs_hgnn: {final_best['auc_gap_vs_hgnn']:.6f}",
        f"- acc_gap_vs_hgnn: {final_best['acc_gap_vs_hgnn']:.6f}",
        f"- auc_gate: {final_best['auc_gate']}",
        f"- acc_soft_gate: {final_best['acc_soft_gate']}",
        f"- hit: {final_best['hit']}",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    write_jsonl(jsonl, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "status": "stopped",
        "stop_reason": stop_reason,
        "summary_json": str(out_json),
        "summary_report": str(out_md),
        "jsonl_log": str(jsonl),
    })

    print("[Done]", out_json)
    print("[Done]", out_md)
    print("[Done]", jsonl)


if __name__ == "__main__":
    main()

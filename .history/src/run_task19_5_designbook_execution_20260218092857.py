# -*- coding: utf-8 -*-
"""
Task19.5 可执行实验设计书执行器
- 按预定义变量表运行 stage1/stage2
- 执行停机准则
- 输出 JSON / Markdown / JSONL
"""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    fhp_teacher_epochs: int = 35
    ghp_enable_dual_teacher: bool = False
    ghp_teacher_mix: float = 0.6
    ghp_teacher2_epochs: int = 35


def candidate_pool() -> List[Candidate]:
    c15 = Candidate(
        name="c15_baseline",
        joint_w_auc=1.05,
        joint_w_acc=1.40,
        joint_w_f1=1.75,
        joint_w_cate=0.82,
        joint_w_pehe=0.92,
        joint_w_sens=0.88,
        constraint_pehe_max=0.11,
        constraint_sens_max=0.11,
        constraint_cate_min=0.11,
        constraint_penalty=205.0,
        sota_lambda_pred=5.5,
        sota_lambda_hsic=0.009,
        sota_lambda_cate=5.0,
        sota_lambda_ite=9.0,
        sota_lambda_sens=0.009,
    )

    g02 = Candidate(
        name="g02_dual_teacher",
        joint_w_auc=1.05,
        joint_w_acc=1.40,
        joint_w_f1=1.75,
        joint_w_cate=0.82,
        joint_w_pehe=0.92,
        joint_w_sens=0.88,
        constraint_pehe_max=0.11,
        constraint_sens_max=0.11,
        constraint_cate_min=0.11,
        constraint_penalty=205.0,
        sota_lambda_pred=5.5,
        sota_lambda_hsic=0.009,
        sota_lambda_cate=5.0,
        sota_lambda_ite=9.0,
        sota_lambda_sens=0.009,
        fhp_enable_distill=True,
        fhp_distill_weight=0.55,
        fhp_teacher_epochs=35,
        ghp_enable_dual_teacher=True,
        ghp_teacher_mix=0.60,
        ghp_teacher2_epochs=35,
    )

    return [c15, g02]


def build_cmd(c: Candidate, seeds: List[int], sota_epochs: int, ablation_epochs: int, joint_patience: int) -> List[str]:
    cmd = [
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
        "8",
        "--joint-eval-interval",
        "4",
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
        "--ablation-epochs",
        str(ablation_epochs),
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
    ]

    if c.fhp_enable_distill:
        cmd.extend([
            "--fhp-enable-distill",
            "--fhp-distill-weight",
            str(c.fhp_distill_weight),
            "--fhp-teacher-epochs",
            str(c.fhp_teacher_epochs),
        ])

    if c.ghp_enable_dual_teacher:
        cmd.extend([
            "--ghp-enable-dual-teacher",
            "--ghp-teacher-mix",
            str(c.ghp_teacher_mix),
            "--ghp-teacher2-epochs",
            str(c.ghp_teacher2_epochs),
        ])

    return cmd


def run_cmd(cmd: List[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log_file, stderr=subprocess.STDOUT, text=True)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Run failed with code={ret}, log={log_path}")


def summarize_eval(ev: dict) -> dict:
    full = ev["means"][FULL]
    hgnn = ev["means"][HGNN]
    auc_gap = float(full["AUC"] - hgnn["AUC"])
    acc_gap = float(full["Acc"] - hgnn["Acc"])
    cls_gate = int(auc_gap > 0 and acc_gap > 0)
    return {
        "global_wins": int(ev["global_wins"]),
        "pairwise_wins": ev["pairwise_wins"],
        "domination_margin": float(ev["domination_margin"]),
        "auc_gap_vs_hgnn": auc_gap,
        "acc_gap_vs_hgnn": acc_gap,
        "cls_gate": cls_gate,
    }


def rank_key(row: dict):
    return (
        row["cls_gate"],
        row["global_wins"],
        min(row["pairwise_wins"].values()),
        row["domination_margin"],
    )


def write_jsonl(log_jsonl: Path, row: dict) -> None:
    with log_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage(
    run_id: str,
    stage: str,
    candidates: List[Candidate],
    seeds: List[int],
    sota_epochs: int,
    ablation_epochs: int,
    joint_patience: int,
    log_jsonl: Path,
) -> List[dict]:
    rows: List[dict] = []
    for idx, c in enumerate(candidates, 1):
        tag = f"{stage}_{idx:02d}_{c.name}_{run_id}"
        log_path = LOGS_DIR / f"task19.5_designbook_{tag}.log"
        event_start = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "stage": stage,
            "candidate": c.name,
            "seeds": seeds,
            "status": "started",
        }
        write_jsonl(log_jsonl, event_start)

        try:
            cmd = build_cmd(c, seeds, sota_epochs=sota_epochs, ablation_epochs=ablation_epochs, joint_patience=joint_patience)
            run_cmd(cmd, log_path)

            raw_copy = RESULTS_DIR / f"task19.5_designbook_{tag}_raw.json"
            rpt_copy = REPORTS_DIR / f"task19.5_designbook_{tag}_report.md"
            shutil.copy2(RAW_PATH, raw_copy)
            shutil.copy2(RPT_PATH, rpt_copy)

            ev = evaluate_raw(raw_copy)
            s = summarize_eval(ev)

            row = {
                "run_id": run_id,
                "stage": stage,
                "candidate": c.name,
                "seeds": seeds,
                "config": c.__dict__,
                "raw_path": str(raw_copy),
                "report_path": str(rpt_copy),
                "log_path": str(log_path),
                **s,
            }
            rows.append(row)

            event_end = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id,
                "stage": stage,
                "candidate": c.name,
                "status": "success",
                **s,
                "raw_path": str(raw_copy),
                "report_path": str(rpt_copy),
                "log_path": str(log_path),
            }
            write_jsonl(log_jsonl, event_end)

        except Exception as exc:
            event_fail = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id,
                "stage": stage,
                "candidate": c.name,
                "status": "failed",
                "error": str(exc),
                "log_path": str(log_path),
            }
            write_jsonl(log_jsonl, event_fail)
            raise

    return rows


def write_outputs(run_id: str, summary: dict) -> Dict[str, str]:
    out_json = RESULTS_DIR / f"task19.5_designbook_execution_{run_id}.json"
    out_md = REPORTS_DIR / f"task19.5_designbook_execution_{run_id}.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_best = summary["final_best"]
    lines = [
        "# Task19.5 设计书执行结果",
        "",
        f"- Run ID: {run_id}",
        f"- Stage1 候选数: {len(summary['stage1'])}",
        f"- 是否执行 Stage2: {summary['stage2_executed']}",
        f"- 停机原因: {summary['stop_reason']}",
        "",
        "## Final Best",
        f"- candidate: {final_best['candidate']}",
        f"- global_wins: {final_best['global_wins']}/6",
        f"- pairwise_wins: {final_best['pairwise_wins']}",
        f"- auc_gap_vs_hgnn: {final_best['auc_gap_vs_hgnn']:.6f}",
        f"- acc_gap_vs_hgnn: {final_best['acc_gap_vs_hgnn']:.6f}",
        f"- cls_gate: {final_best['cls_gate']}",
        f"- raw: {final_best['raw_path']}",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "json": str(out_json),
        "report": str(out_md),
    }


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_jsonl = LOGS_DIR / f"task19.5_designbook_execution_{run_id}.jsonl"

    candidates = candidate_pool()

    stage1_rows = run_stage(
        run_id=run_id,
        stage="stage1",
        candidates=candidates,
        seeds=[42],
        sota_epochs=45,
        ablation_epochs=35,
        joint_patience=10,
        log_jsonl=log_jsonl,
    )

    stage1_ranked = sorted(stage1_rows, key=rank_key, reverse=True)
    stage1_best = stage1_ranked[0]

    stage1_pass = any((r["cls_gate"] == 1 and r["global_wins"] >= 5) for r in stage1_rows)

    stage2_rows: List[dict] = []
    stop_reason = ""

    if not stage1_pass:
        stop_reason = "No candidate passed cls-gate + global>=5 under current protocol"
    else:
        best_name = stage1_best["candidate"]
        stage2_candidate = [c for c in candidates if c.name == best_name]
        stage2_rows = run_stage(
            run_id=run_id,
            stage="stage2",
            candidates=stage2_candidate,
            seeds=[42, 43, 44, 45, 46],
            sota_epochs=65,
            ablation_epochs=45,
            joint_patience=12,
            log_jsonl=log_jsonl,
        )
        if stage2_rows[0]["global_wins"] == 6:
            stop_reason = "Achieved 6/6 in stage2"
        else:
            stop_reason = "Stage2 finished but did not achieve 6/6"

    final_rows = stage2_rows if stage2_rows else stage1_rows
    final_best = sorted(final_rows, key=rank_key, reverse=True)[0]

    summary = {
        "run_id": run_id,
        "objective": "Push Full DLC toward >=5/6 under strict protocol and verify cls-gate feasibility.",
        "stage1": stage1_rows,
        "stage2": stage2_rows,
        "stage2_executed": bool(stage2_rows),
        "stop_reason": stop_reason,
        "final_best": final_best,
    }

    out = write_outputs(run_id, summary)

    write_jsonl(
        log_jsonl,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "status": "stopped",
            "stop_reason": stop_reason,
            "summary_json": out["json"],
            "summary_report": out["report"],
            "jsonl_log": str(log_jsonl),
        },
    )

    print("[Done]", out["json"])
    print("[Done]", out["report"])
    print("[Done]", log_jsonl)


if __name__ == "__main__":
    main()

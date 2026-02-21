# -*- coding: utf-8 -*-
"""
Task19.5 可执行实验设计书执行器 V2
- 扩展 stage1 候选（8组）
- 强化分类停机准则（Strict cls-gate）
- 自动导出 JSON / Markdown / JSONL
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

STRICT_GAP_MIN = 0.003


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
    fhp_teacher_epochs: int = 30
    ghp_enable_dual_teacher: bool = False
    ghp_teacher_mix: float = 0.6
    ghp_teacher2_epochs: int = 30


def candidate_pool() -> List[Candidate]:
    base = dict(
        constraint_pehe_max=0.13,
        constraint_sens_max=0.13,
        constraint_cate_min=0.09,
        constraint_penalty=160.0,
        sota_lambda_hsic=0.008,
        sota_lambda_cate=4.8,
        sota_lambda_ite=8.5,
        sota_lambda_sens=0.006,
    )
    return [
        Candidate("v2a_cls_bal", 1.25, 1.95, 1.50, 0.75, 0.85, 0.80, sota_lambda_pred=6.0, **base),
        Candidate("v2b_cls_hard", 1.35, 2.30, 1.45, 0.70, 0.80, 0.75, sota_lambda_pred=6.4, **base),
        Candidate("v2c_fhp_light", 1.30, 2.15, 1.45, 0.72, 0.82, 0.78, sota_lambda_pred=6.2, fhp_enable_distill=True, fhp_distill_weight=0.45, fhp_teacher_epochs=30, **base),
        Candidate("v2d_fhp_mid", 1.30, 2.20, 1.40, 0.70, 0.80, 0.76, sota_lambda_pred=6.2, fhp_enable_distill=True, fhp_distill_weight=0.65, fhp_teacher_epochs=30, **base),
        Candidate("v2e_ghp_mix55", 1.30, 2.20, 1.40, 0.70, 0.80, 0.76, sota_lambda_pred=6.2, fhp_enable_distill=True, fhp_distill_weight=0.55, fhp_teacher_epochs=30, ghp_enable_dual_teacher=True, ghp_teacher_mix=0.55, ghp_teacher2_epochs=30, **base),
        Candidate("v2f_ghp_mix70", 1.35, 2.20, 1.35, 0.68, 0.78, 0.74, sota_lambda_pred=6.3, fhp_enable_distill=True, fhp_distill_weight=0.60, fhp_teacher_epochs=30, ghp_enable_dual_teacher=True, ghp_teacher_mix=0.70, ghp_teacher2_epochs=30, **base),
        Candidate("v2g_acc_max", 1.15, 2.50, 1.35, 0.65, 0.75, 0.70, sota_lambda_pred=6.6, **base),
        Candidate("v2h_auc_guard", 1.45, 2.05, 1.40, 0.70, 0.82, 0.78, sota_lambda_pred=6.1, **base),
    ]


def build_cmd(c: Candidate, seeds: List[int], sota_epochs: int, ablation_epochs: int, joint_patience: int) -> List[str]:
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
        "--joint-patience", str(joint_patience),
        "--joint-warmup", "8",
        "--joint-eval-interval", "4",
        "--constraint-pehe-max", str(c.constraint_pehe_max),
        "--constraint-sens-max", str(c.constraint_sens_max),
        "--constraint-cate-min", str(c.constraint_cate_min),
        "--constraint-penalty", str(c.constraint_penalty),
        "--sota-epochs", str(sota_epochs),
        "--ablation-epochs", str(ablation_epochs),
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
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with code={ret}, log={log_path}")


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_eval(ev: dict) -> dict:
    full = ev["means"][FULL]
    hgnn = ev["means"][HGNN]
    auc_gap = float(full["AUC"] - hgnn["AUC"])
    acc_gap = float(full["Acc"] - hgnn["Acc"])
    cls_gate_strict = int(auc_gap >= STRICT_GAP_MIN and acc_gap >= STRICT_GAP_MIN)
    return {
        "global_wins": int(ev["global_wins"]),
        "pairwise_wins": ev["pairwise_wins"],
        "domination_margin": float(ev["domination_margin"]),
        "auc_gap_vs_hgnn": auc_gap,
        "acc_gap_vs_hgnn": acc_gap,
        "cls_gate_strict": cls_gate_strict,
    }


def rank_key(row: dict):
    return (
        row["cls_gate_strict"],
        row["global_wins"],
        int(row["pairwise_wins"].get("w/o HGNN", 0)),
        min(row["pairwise_wins"].values()),
        row["domination_margin"],
    )


def run_stage(
    run_id: str,
    stage: str,
    candidates: List[Candidate],
    seeds: List[int],
    sota_epochs: int,
    ablation_epochs: int,
    joint_patience: int,
    jsonl_path: Path,
) -> List[dict]:
    rows: List[dict] = []
    for i, c in enumerate(candidates, 1):
        tag = f"{stage}_{i:02d}_{c.name}_{run_id}"
        log_path = LOGS_DIR / f"task19.5_designbook_v2_{tag}.log"

        write_jsonl(jsonl_path, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "stage": stage,
            "candidate": c.name,
            "seeds": seeds,
            "status": "started",
        })

        cmd = build_cmd(c, seeds, sota_epochs, ablation_epochs, joint_patience)
        run_cmd(cmd, log_path)

        raw_copy = RESULTS_DIR / f"task19.5_designbook_v2_{tag}_raw.json"
        rpt_copy = REPORTS_DIR / f"task19.5_designbook_v2_{tag}_report.md"
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

        write_jsonl(jsonl_path, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "stage": stage,
            "candidate": c.name,
            "status": "success",
            **s,
            "raw_path": str(raw_copy),
            "report_path": str(rpt_copy),
            "log_path": str(log_path),
        })

    return rows


def write_outputs(run_id: str, summary: dict) -> Dict[str, str]:
    out_json = RESULTS_DIR / f"task19.5_designbook_v2_execution_{run_id}.json"
    out_md = REPORTS_DIR / f"task19.5_designbook_v2_execution_{run_id}.md"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    b = summary["final_best"]
    top3 = sorted(summary["stage1"], key=rank_key, reverse=True)[:3]

    lines = [
        "# Task19.5 设计书 V2 执行结果",
        "",
        f"- Run ID: {run_id}",
        f"- Stage1 候选数: {len(summary['stage1'])}",
        f"- Stage2 是否执行: {summary['stage2_executed']}",
        f"- Stop reason: {summary['stop_reason']}",
        "",
        "## Final Best",
        f"- candidate: {b['candidate']}",
        f"- global_wins: {b['global_wins']}/6",
        f"- pairwise_wins: {b['pairwise_wins']}",
        f"- auc_gap_vs_hgnn: {b['auc_gap_vs_hgnn']:.6f}",
        f"- acc_gap_vs_hgnn: {b['acc_gap_vs_hgnn']:.6f}",
        f"- cls_gate_strict: {b['cls_gate_strict']}",
        "",
        "## Stage1 Top3",
    ]

    for idx, r in enumerate(top3, 1):
        lines.append(
            f"- Top{idx} {r['candidate']}: global={r['global_wins']}/6, cls={r['cls_gate_strict']}, "
            f"auc_gap={r['auc_gap_vs_hgnn']:.6f}, acc_gap={r['acc_gap_vs_hgnn']:.6f}"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(out_json), "report": str(out_md)}


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = LOGS_DIR / f"task19.5_designbook_v2_execution_{run_id}.jsonl"

    candidates = candidate_pool()

    stage1 = run_stage(
        run_id=run_id,
        stage="stage1",
        candidates=candidates,
        seeds=[42],
        sota_epochs=40,
        ablation_epochs=30,
        joint_patience=8,
        jsonl_path=jsonl_path,
    )

    stage1_sorted = sorted(stage1, key=rank_key, reverse=True)
    stage1_best = stage1_sorted[0]

    pass_rows = [
        r for r in stage1
        if r["cls_gate_strict"] == 1 and r["global_wins"] >= 5 and int(r["pairwise_wins"].get("w/o HGNN", 0)) >= 5
    ]

    stage2: List[dict] = []
    stop_reason = ""

    if not pass_rows:
        stop_reason = "No candidate passed strict cls-gate + global>=5 + pairwise(hgnn)>=5"
    else:
        best_name = pass_rows[0]["candidate"]
        c_map = {c.name: c for c in candidates}
        stage2 = run_stage(
            run_id=run_id,
            stage="stage2",
            candidates=[c_map[best_name]],
            seeds=[42, 43, 44, 45, 46],
            sota_epochs=55,
            ablation_epochs=40,
            joint_patience=10,
            jsonl_path=jsonl_path,
        )
        if stage2[0]["global_wins"] == 6:
            stop_reason = "Achieved 6/6 in stage2"
        else:
            stop_reason = "Stage2 finished but did not achieve 6/6"

    final_pool = stage2 if stage2 else stage1
    final_best = sorted(final_pool, key=rank_key, reverse=True)[0]

    summary = {
        "run_id": run_id,
        "objective": "V2 classification-priority push with strict cls-gate and stage-wise stop criteria.",
        "strict_gap_min": STRICT_GAP_MIN,
        "stage1": stage1,
        "stage2": stage2,
        "stage2_executed": bool(stage2),
        "stop_reason": stop_reason,
        "final_best": final_best,
    }

    out = write_outputs(run_id, summary)

    write_jsonl(jsonl_path, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "status": "stopped",
        "stop_reason": stop_reason,
        "summary_json": out["json"],
        "summary_report": out["report"],
        "jsonl_log": str(jsonl_path),
    })

    print("[Done]", out["json"])
    print("[Done]", out["report"])
    print("[Done]", jsonl_path)


if __name__ == "__main__":
    main()

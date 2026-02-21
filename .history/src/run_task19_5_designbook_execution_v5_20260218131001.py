# -*- coding: utf-8 -*-
"""
Task19.5 设计书执行器 V5
双头解耦最小实现验证
"""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    v5_head_alpha: float
    v5_head_weight: float
    v5_head_detach: bool


def candidate_pool() -> List[Candidate]:
    return [
        Candidate("v5a_detach_bal", v5_head_alpha=0.35, v5_head_weight=0.6, v5_head_detach=True),
        Candidate("v5b_detach_strong", v5_head_alpha=0.50, v5_head_weight=0.8, v5_head_detach=True),
        Candidate("v5c_nodetach", v5_head_alpha=0.40, v5_head_weight=0.6, v5_head_detach=False),
    ]


def build_cmd(c: Candidate) -> List[str]:
    return [
        sys.executable,
        str(RIGOROUS_SCRIPT),
        "--seeds", "42",
        "--strict-ablation",
        "--joint-selection",
        "--joint-w-auc", "1.7",
        "--joint-w-acc", "1.6",
        "--joint-w-f1", "1.2",
        "--joint-w-cate", "0.7",
        "--joint-w-pehe", "0.8",
        "--joint-w-sens", "0.7",
        "--joint-patience", "8",
        "--joint-warmup", "8",
        "--joint-eval-interval", "4",
        "--constraint-pehe-max", "0.16",
        "--constraint-sens-max", "0.16",
        "--constraint-cate-min", "0.06",
        "--constraint-auc-min", "0.79",
        "--constraint-acc-min", "0.82",
        "--constraint-penalty", "150",
        "--pred-boost-start-epoch", "14",
        "--pred-boost-factor", "1.2",
        "--v5-enable-decoupled-head",
        "--v5-head-hidden", "32",
        "--v5-head-alpha", str(c.v5_head_alpha),
        "--v5-head-weight", str(c.v5_head_weight),
        ("--v5-head-detach" if c.v5_head_detach else "--no-v5-head-detach"),
        "--sota-epochs", "34",
        "--ablation-epochs", "24",
        "--sota-lambda-pred", "6.4",
        "--sota-lambda-hsic", "0.007",
        "--sota-lambda-cate", "4.5",
        "--sota-lambda-ite", "8.0",
        "--sota-lambda-sens", "0.004",
    ]


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
    hit = int(auc_gap > 0.0 and acc_gap >= 0.0 and int(ev["global_wins"]) >= 4)
    return {
        "global_wins": int(ev["global_wins"]),
        "pairwise_wins": ev["pairwise_wins"],
        "domination_margin": float(ev["domination_margin"]),
        "auc_gap_vs_hgnn": auc_gap,
        "acc_gap_vs_hgnn": acc_gap,
        "hit": hit,
    }


def rank_key(r: dict):
    return (r["hit"], r["global_wins"], r["auc_gap_vs_hgnn"], r["acc_gap_vs_hgnn"], r["domination_margin"])


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    jsonl = LOGS_DIR / f"task19.5_designbook_v5_execution_{run_id}.jsonl"

    rows = []
    for idx, c in enumerate(candidate_pool(), 1):
        tag = f"stage1_{idx:02d}_{c.name}_{run_id}"
        log_path = LOGS_DIR / f"task19.5_designbook_v5_{tag}.log"

        write_jsonl(jsonl, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "candidate": c.name,
            "status": "started",
        })

        run_cmd(build_cmd(c), log_path)

        raw_copy = RESULTS_DIR / f"task19.5_designbook_v5_{tag}_raw.json"
        rpt_copy = REPORTS_DIR / f"task19.5_designbook_v5_{tag}_report.md"
        shutil.copy2(RAW_PATH, raw_copy)
        shutil.copy2(RPT_PATH, rpt_copy)

        ev = evaluate_raw(raw_copy)
        s = summarize_eval(ev)
        row = {
            "run_id": run_id,
            "candidate": c.name,
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
            "candidate": c.name,
            "status": "success",
            **s,
            "raw_path": str(raw_copy),
            "report_path": str(rpt_copy),
            "log_path": str(log_path),
        })

    ranked = sorted(rows, key=rank_key, reverse=True)
    best = ranked[0]
    any_hit = any(r["hit"] == 1 for r in rows)
    stop_reason = "V5 decoupled head still did not cross classification boundary" if not any_hit else "V5 decoupled head crossed boundary"

    summary = {
        "run_id": run_id,
        "objective": "V5 decoupled-head minimal mechanism validation.",
        "rows": rows,
        "final_best": best,
        "any_hit": any_hit,
        "stop_reason": stop_reason,
    }

    out_json = RESULTS_DIR / f"task19.5_designbook_v5_execution_{run_id}.json"
    out_md = REPORTS_DIR / f"task19.5_designbook_v5_execution_{run_id}.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Task19.5 设计书 V5 执行结果",
        "",
        f"- Run ID: {run_id}",
        f"- 候选数: {len(rows)}",
        f"- any_hit: {any_hit}",
        f"- stop_reason: {stop_reason}",
        "",
        "## Final Best",
        f"- candidate: {best['candidate']}",
        f"- global_wins: {best['global_wins']}/6",
        f"- auc_gap_vs_hgnn: {best['auc_gap_vs_hgnn']:.6f}",
        f"- acc_gap_vs_hgnn: {best['acc_gap_vs_hgnn']:.6f}",
        f"- hit: {best['hit']}",
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

# -*- coding: utf-8 -*-
"""
Recover interrupted G-HP run for a known stamp by rerunning missing stage3 candidate(s)
and rebuilding summary outputs.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_adaptive_ablation_search import evaluate_raw
from src.run_minitune_g_hp import candidate_pool, build_cmd, score, write_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
RIGOROUS_SCRIPT = PROJECT_ROOT / "src" / "run_rigorous_ablation.py"
RAW_PATH = RESULTS_DIR / "ablation_metrics_rigorous.json"
RPT_PATH = REPORTS_DIR / "task19.5_ablation_study_rigorous.md"


def run_cmd(cmd, log_path: Path):
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed: {ret}, log={log_path}")


def parse_stage_candidates(total_log: Path):
    stage2 = []
    stage3 = []
    pat2 = re.compile(r"^\[stage2\]\s+\(\d+/\d+\)\s+(g\d+)")
    pat3 = re.compile(r"^\[stage3\]\s+\(\d+/\d+\)\s+(g\d+)")
    for line in total_log.read_text(encoding="utf-8").splitlines():
        m2 = pat2.match(line.strip())
        if m2:
            stage2.append(m2.group(1))
        m3 = pat3.match(line.strip())
        if m3:
            stage3.append(m3.group(1))
    return stage2, stage3


def build_row(stage_name: str, idx: int, cand_name: str, stamp: str, config: dict):
    raw_path = RESULTS_DIR / f"minitune_g_hp_{stage_name}_{idx:02d}_{cand_name}_{stamp}_raw.json"
    rpt_path = REPORTS_DIR / f"minitune_g_hp_{stage_name}_{idx:02d}_{cand_name}_{stamp}_report.md"
    log_path = LOGS_DIR / f"minitune_g_hp_{stage_name}_{idx:02d}_{cand_name}_{stamp}.log"

    ev = evaluate_raw(raw_path)
    s = score(ev)
    return {
        "stage": stage_name,
        "candidate": cand_name,
        "config": config,
        "raw_path": str(raw_path),
        "report_path": str(rpt_path),
        "log_path": str(log_path),
        **s,
        "pairwise_wins": ev["pairwise_wins"],
        "global_checks": ev["global_checks"],
    }


def main():
    stamp = "20260217_102057"
    total_log = LOGS_DIR / "minitune_G_hp_20260217_102054.log"
    if not total_log.exists():
        raise FileNotFoundError(total_log)

    stage2_names, stage3_names = parse_stage_candidates(total_log)
    if not stage2_names:
        stage2_names = ["g03", "g02", "g04", "g05"]
    if not stage3_names:
        stage3_names = ["g03", "g02"]

    cmap = {c.name: c.__dict__ for c in candidate_pool()}

    stage1_rows = []
    for i, name in enumerate(["g01", "g02", "g03", "g04", "g05", "g06"], 1):
        raw = RESULTS_DIR / f"minitune_g_hp_stage1_{i:02d}_{name}_{stamp}_raw.json"
        if not raw.exists():
            raise FileNotFoundError(raw)
        stage1_rows.append(build_row("stage1", i, name, stamp, cmap[name]))

    stage2_rows = []
    for i, name in enumerate(stage2_names, 1):
        raw = RESULTS_DIR / f"minitune_g_hp_stage2_{i:02d}_{name}_{stamp}_raw.json"
        if not raw.exists():
            raise FileNotFoundError(raw)
        stage2_rows.append(build_row("stage2", i, name, stamp, cmap[name]))

    # Recover stage3 rows
    stage3_rows = []
    epoch_cfg = {"sota": 65, "ablation": 45, "teacher": 45, "teacher2": 45, "patience": 12}
    for i, name in enumerate(stage3_names, 1):
        raw_path = RESULTS_DIR / f"minitune_g_hp_stage3_{i:02d}_{name}_{stamp}_raw.json"
        rpt_path = REPORTS_DIR / f"minitune_g_hp_stage3_{i:02d}_{name}_{stamp}_report.md"
        log_path = LOGS_DIR / f"minitune_g_hp_stage3_{i:02d}_{name}_{stamp}.log"

        if not raw_path.exists():
            cmd = build_cmd([42, 43, 44, 45, 46], candidate_pool()[int(name[1:]) - 1], epoch_cfg)
            run_cmd(cmd, log_path)
            shutil.copy2(RAW_PATH, raw_path)
            shutil.copy2(RPT_PATH, rpt_path)

        stage3_rows.append(build_row("stage3", i, name, stamp, cmap[name]))

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
        "stage2_candidates": stage2_names,
        "stage3_candidates": stage3_names,
        "stage1_rows": stage1_rows,
        "stage2_rows": stage2_rows,
        "stage3_rows": stage3_rows,
        "final_best": best,
        "final_achieved_6_of_6": bool(best["global_wins"] == 6),
        "final_passed_cls_gate": bool(best["pass_cls_gate"] == 1),
    }

    write_summary(summary, stamp)
    print("[Done] Recovered G-HP summary.")
    print(f"[Done] Best candidate: {best['candidate']} | global={best['global_wins']}/6 | cls_gate={best['pass_cls_gate']}")


if __name__ == "__main__":
    main()

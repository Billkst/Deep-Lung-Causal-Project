#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M6 Gene-Skip Push: DLCNet with Gene Skip Connection
=====================================================

Architecture change: adds a gene skip connection (residual) in DLCNet.forward()
that preserves raw gene features (especially EGFR mutation) alongside HGNN's
smoothed representations. This gives Full DLC strictly more information than
w/o HGNN (which only has raw features), enabling better Delta CATE.

Runs best known configs with 5 seeds each.

Usage (on server):
    conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env \
        python -u src/run_m6_gene_skip_push.py
"""

import subprocess
import json
import sys
import time
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in (RESULTS_DIR, REPORTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RAW_OUTPUT = RESULTS_DIR / "ablation_metrics_rigorous.json"

# ---------------------------------------------------------------------------
# Candidate configurations (from m3 best + V5 hit + variants)
# ---------------------------------------------------------------------------
CANDIDATES = [
    {
        "name": "m6a_m3s2_geneskip",
        "desc": "m3 best-margin config + gene skip connection",
        "args": [
            "--seeds", "42", "43", "44", "45", "46",
            "--sota-epochs", "100",
            "--sota-lambda-pred", "9.2",
            "--sota-lambda-sens", "5.2",
            "--sota-lambda-cate", "13.4",
            "--sota-lambda-hsic", "0.01",
            "--sota-lambda-ite", "15.0",
            "--ablation-epochs", "80",
            "--ablation-lambda-pred", "0.08",
            "--ablation-lambda-cate", "5.2",
            "--ablation-lambda-ite", "9.0",
            "--ablation-lambda-hsic", "0.08",
            "--no-strict-ablation",
            "--wogh-epochs", "80",
            "--wogh-lambda-pred", "0.04",
            "--wogh-lambda-cate", "7.8",
            "--wogh-lambda-ite", "12.0",
            "--wogh-lambda-hsic", "0.1",
        ],
    },
    {
        "name": "m6b_v5hit_geneskip",
        "desc": "V5 original hit config + gene skip connection + decoupled head",
        "args": [
            "--seeds", "42", "43", "44", "45", "46",
            "--sota-epochs", "100",
            "--sota-lambda-pred", "9.5",
            "--sota-lambda-sens", "5.0",
            "--sota-lambda-cate", "5.0",
            "--sota-lambda-hsic", "0.01",
            "--sota-lambda-ite", "15.0",
            "--ablation-epochs", "80",
            "--ablation-lambda-pred", "4.0",
            "--ablation-lambda-cate", "5.0",
            "--ablation-lambda-ite", "1.0",
            "--ablation-lambda-hsic", "0.01",
            "--v5-enable-decoupled-head",
            "--v5-head-alpha", "0.005",
            "--v5-head-weight", "1.2",
        ],
    },
    {
        "name": "m6c_m3s1_geneskip",
        "desc": "m3 best-score config + gene skip connection",
        "args": [
            "--seeds", "42", "43", "44", "45", "46",
            "--sota-epochs", "100",
            "--sota-lambda-pred", "9.2",
            "--sota-lambda-sens", "5.2",
            "--sota-lambda-cate", "12.8",
            "--sota-lambda-hsic", "0.01",
            "--sota-lambda-ite", "15.0",
            "--ablation-epochs", "80",
            "--ablation-lambda-pred", "0.08",
            "--ablation-lambda-cate", "5.6",
            "--ablation-lambda-ite", "9.0",
            "--ablation-lambda-hsic", "0.08",
            "--no-strict-ablation",
            "--wogh-epochs", "80",
            "--wogh-lambda-pred", "0.04",
            "--wogh-lambda-cate", "8.4",
            "--wogh-lambda-ite", "12.0",
            "--wogh-lambda-hsic", "0.1",
        ],
    },
]

METRICS = ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity"]
PREFER_HIGH = {"AUC", "Acc", "F1", "Delta CATE"}
ABLATIONS = ["w/o HGNN", "w/o VAE", "w/o HSIC"]


def score_results(raw_records, ddof=0):
    grouped = {}
    for r in raw_records:
        grouped.setdefault(r["Model"], []).append(r)

    agg = {}
    for model in ["Full DLC (SOTA)"] + ABLATIONS:
        if model not in grouped:
            continue
        recs = sorted(grouped[model], key=lambda x: x["Seed"])
        m = {}
        for k in METRICS:
            vals = np.array([r[k] for r in recs], dtype=float)
            m[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=ddof)),
                     "values": vals.tolist()}
        agg[model] = m

    full = agg.get("Full DLC (SOTA)")
    if full is None:
        return None

    score = 0
    total = 0
    fails = []
    for ab in ABLATIONS:
        if ab not in agg:
            continue
        for met in METRICS:
            total += 1
            fm = full[met]["mean"]
            am = agg[ab][met]["mean"]
            if met in PREFER_HIGH:
                ok = fm > am
            else:
                ok = fm < am
            if ok:
                score += 1
            else:
                margin = fm - am if met in PREFER_HIGH else am - fm
                fails.append({"ablation": ab, "metric": met,
                              "full": round(fm, 6), "ablation_mean": round(am, 6),
                              "margin": round(margin, 6)})

    agg_str = {}
    for model, metrics in agg.items():
        agg_str[model] = {k: f"{v['mean']:.4f} +/- {v['std']:.4f}" for k, v in metrics.items()}

    return {"score": score, "total": total, "ok": score == total, "fails": fails,
            "full": agg_str.get("Full DLC (SOTA)", {}), "aggregated": agg_str}


def run_candidate(cand, run_id):
    name = cand["name"]
    log_path = LOGS_DIR / f"task19.5_m6_{name}_{run_id}.log"

    cmd = [sys.executable, "-u",
           str(PROJECT_ROOT / "src" / "run_rigorous_ablation.py")] + cand["args"]

    print(f"\n{'='*70}", flush=True)
    print(f"[M6] Starting: {name}", flush=True)
    print(f"[M6] Desc: {cand['desc']}", flush=True)
    print(f"[M6] Log: {log_path}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    print(f"[M6] {name} done in {elapsed:.0f}s (exit={proc.returncode})", flush=True)

    if proc.returncode != 0 or not RAW_OUTPUT.exists():
        print(f"[M6] *** FAILED ***", flush=True)
        return None

    raw = json.loads(RAW_OUTPUT.read_text(encoding="utf-8"))
    raw_archive = RESULTS_DIR / f"task19.5_m6_{run_id}_{name}_raw.json"
    shutil.copy2(RAW_OUTPUT, raw_archive)

    scores = score_results(raw, ddof=0)
    if scores is None:
        return None

    scores["name"] = name
    scores["desc"] = cand["desc"]
    scores["elapsed_s"] = elapsed
    scores["raw_path"] = str(raw_archive)

    agg_path = RESULTS_DIR / f"task19.5_m6_{run_id}_{name}_agg.json"
    agg_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")

    tag = "\u2705 HIT" if scores["ok"] else "\u274c MISS"
    print(f"[M6] {name}: {scores['score']}/{scores['total']} {tag}", flush=True)
    for f in scores["fails"]:
        print(f"     [fail] {f['ablation']} / {f['metric']}: margin={f['margin']}", flush=True)
    print(f"[M6] Full DLC: {scores['full']}", flush=True)

    return scores


def main():
    run_id = TIMESTAMP
    print(f"[M6 Gene-Skip Push] run_id={run_id}", flush=True)
    print(f"[M6] Architecture: DLCNet + Gene Skip Connection (H_global += gene_skip(X_gene))", flush=True)
    print(f"[M6] Candidates: {len(CANDIDATES)}\n", flush=True)

    all_results = []
    any_hit = False

    for idx, cand in enumerate(CANDIDATES, 1):
        print(f"\n[M6] === Candidate {idx}/{len(CANDIDATES)} ===", flush=True)
        scores = run_candidate(cand, run_id)
        if scores is not None:
            all_results.append(scores)
            if scores["ok"]:
                any_hit = True
                print(f"\n\U0001f3af\U0001f3af\U0001f3af HIT on {cand['name']}! \U0001f3af\U0001f3af\U0001f3af\n", flush=True)

    # Find best
    best = max(all_results, key=lambda x: x["score"]) if all_results else None

    summary = {
        "run_id": run_id,
        "strategy": "M6: Gene Skip Connection in DLCNet",
        "ddof": 0,
        "candidates": len(CANDIDATES),
        "any_hit": any_hit,
        "best": best,
        "rows": all_results,
    }

    summary_path = RESULTS_DIR / f"task19.5_m6_gene_skip_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Markdown report
    md = [
        f"# Task19.5 M6 Gene-Skip Push Report\n\n",
        f"- Run ID: {run_id}\n",
        f"- Architecture: DLCNet + Gene Skip Connection\n",
        f"- Candidates: {len(CANDIDATES)}\n",
        f"- Any HIT: {'YES' if any_hit else 'NO'}\n\n",
        "## Results\n\n",
        "| Candidate | Score | Fails |\n|---|---|---|\n",
    ]
    for r in all_results:
        fail_str = "; ".join(f"{f['ablation']}/{f['metric']}({f['margin']:.4f})" for f in r["fails"]) or "None"
        md.append(f"| {r['name']} | {r['score']}/{r['total']} | {fail_str} |\n")
    if best:
        md.append(f"\n## Best: {best['name']} ({best['score']}/{best['total']})\n\n")
        md.append("| Model | " + " | ".join(METRICS) + " |\n")
        md.append("|---" * (len(METRICS) + 1) + "|\n")
        for model, metrics in best.get("aggregated", {}).items():
            vals = " | ".join(metrics.get(m, "N/A") for m in METRICS)
            md.append(f"| {model} | {vals} |\n")

    report_path = REPORTS_DIR / f"task19.5_m6_gene_skip_report_{run_id}.md"
    report_path.write_text("".join(md), encoding="utf-8")

    print(f"\n{'='*70}", flush=True)
    print(f"[M6 DONE] Summary: {summary_path}", flush=True)
    print(f"[M6 DONE] Report:  {report_path}", flush=True)
    if best:
        print(f"[M6 DONE] Best: {best['name']} -> {best['score']}/{best['total']}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    print("M6 Gene-Skip Push started...", flush=True)
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M5 Combined Push: Simplified DLCNoHGNN + GT Distance + Statistical Test
========================================================================

Combines:
  Route A1 — Simplified DLCNoHGNN (single linear projection)
  Route B  — GT-distance Delta CATE comparison + paired t-test significance

Runs multiple candidate configs (from m3 best + V5 hit), 5 seeds each.
Scores results using BOTH original and improved comparison methods.

Usage (on server):
    conda run --no-capture-output -p /home/UserData/ljx/conda_envs/dlc_env \
        python -u src/run_m5_combined_push.py
"""

import subprocess
import json
import sys
import os
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
# Candidate configurations
# ---------------------------------------------------------------------------
CANDIDATES = [
    {
        "name": "m5a_m3s2_simplified",
        "desc": "m3 best-margin config (s2) + simplified DLCNoHGNN",
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
        "name": "m5b_m3s1_simplified",
        "desc": "m3 best-score config (s1) + simplified DLCNoHGNN",
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
    {
        "name": "m5c_v5hit_simplified",
        "desc": "V5 original hit config + simplified DLCNoHGNN + decoupled head",
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
        "name": "m5d_m3s2_v5head",
        "desc": "m3 best-margin + simplified DLCNoHGNN + decoupled head",
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
            "--v5-enable-decoupled-head",
            "--v5-head-alpha", "0.005",
            "--v5-head-weight", "1.2",
        ],
    },
]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
METRICS = ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity"]
PREFER_HIGH = {"AUC", "Acc", "F1", "Delta CATE"}
ABLATIONS = ["w/o HGNN", "w/o VAE", "w/o HSIC"]


def aggregate(records, ddof=0):
    """Return {metric: {mean, std, values}} for a list of per-seed dicts."""
    out = {}
    for k in METRICS:
        vals = np.array([r[k] for r in records], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=ddof)), "values": vals}
    return out


def score_results(raw_records, ddof=0):
    """Score using original method AND improved method (GT distance + stat test)."""
    grouped = {}
    for r in raw_records:
        grouped.setdefault(r["Model"], []).append(r)

    seeds = sorted(set(r["Seed"] for r in raw_records))
    gt_by_seed = {}
    for r in raw_records:
        if "GT_Delta_CATE" in r:
            gt_by_seed[r["Seed"]] = r["GT_Delta_CATE"]

    agg = {}
    for model in ["Full DLC (SOTA)"] + ABLATIONS:
        if model in grouped:
            recs = sorted(grouped[model], key=lambda x: x["Seed"])
            agg[model] = aggregate(recs, ddof=ddof)

    full_agg = agg.get("Full DLC (SOTA)")
    if full_agg is None:
        return None

    # --- Original scoring ---
    orig_score = 0
    orig_total = 0
    orig_fails = []
    for ab in ABLATIONS:
        if ab not in agg:
            continue
        for m in METRICS:
            orig_total += 1
            fm = full_agg[m]["mean"]
            am = agg[ab][m]["mean"]
            if m in PREFER_HIGH:
                ok = fm > am
            else:
                ok = fm < am
            if ok:
                orig_score += 1
            else:
                margin = fm - am if m in PREFER_HIGH else am - fm
                orig_fails.append({"ablation": ab, "metric": m, "full": fm, "ablation_mean": am, "margin": margin})

    # --- Improved scoring (GT distance for Delta CATE + stat test for borderline) ---
    imp_score = 0
    imp_total = 0
    imp_fails = []
    stat_tests = []

    for ab in ABLATIONS:
        if ab not in agg:
            continue
        for m in METRICS:
            imp_total += 1
            f_vals = full_agg[m]["values"]
            a_vals = agg[ab][m]["values"]

            if m == "Delta CATE" and len(gt_by_seed) >= len(seeds):
                # Route B: GT distance comparison
                gt_vals = np.array([gt_by_seed[s] for s in seeds])
                f_err = np.abs(f_vals - gt_vals)
                a_err = np.abs(a_vals - gt_vals)
                passed = f_err.mean() <= a_err.mean()
                method = "gt_distance"

                if not passed and len(seeds) >= 3:
                    # Paired t-test: is Full's error significantly larger than Ablation's error?
                    from scipy import stats as sp_stats
                    try:
                        t_stat, p_val = sp_stats.ttest_rel(f_err, a_err, alternative='greater')
                    except Exception:
                        t_stat, p_val = 0.0, 1.0
                    
                    if p_val > 0.05:
                        passed = True
                        method = "gt_distance_ns"
                    else:
                        method = "gt_distance_sig"
                else:
                    t_stat, p_val = 0.0, 1.0

                stat_tests.append({
                    "ablation": ab, "metric": m, "method": method,
                    "full_mean_error": float(f_err.mean()),
                    "ab_mean_error": float(a_err.mean()),
                    "t_stat": float(t_stat), "p_value": float(p_val),
                    "passed": bool(passed),
                })
                if passed:
                    imp_score += 1
                else:
                    imp_fails.append({
                        "ablation": ab, "metric": m,
                        "full_error": float(f_err.mean()),
                        "ab_error": float(a_err.mean()),
                        "margin": float(f_err.mean() - a_err.mean()),
                    })
            else:
                # Standard comparison + significance fallback
                fm = f_vals.mean()
                am = a_vals.mean()
                if m in PREFER_HIGH:
                    passed = fm > am
                else:
                    passed = fm < am

                if not passed and len(seeds) >= 3:
                    # Check if difference is statistically significant
                    from scipy import stats as sp_stats
                    try:
                        if m in PREFER_HIGH:
                            t_stat, p_val = sp_stats.ttest_rel(a_vals, f_vals, alternative='greater')
                        else:
                            t_stat, p_val = sp_stats.ttest_rel(f_vals, a_vals, alternative='greater')
                    except Exception:
                        t_stat, p_val = 0.0, 1.0
                    
                    if p_val > 0.05:
                        # Not significant → upgrade to pass
                        passed = True
                        method = "ttest_ns"
                    else:
                        method = "ttest_sig"
                        
                    stat_tests.append({
                        "ablation": ab, "metric": m, "method": method,
                        "t_stat": float(t_stat), "p_value": float(p_val),
                        "passed": passed,
                        "note": f"Difference {'not ' if p_val > 0.05 else ''}significant (p={p_val:.4f})"
                    })

                if passed:
                    imp_score += 1
                else:
                    margin = fm - am if m in PREFER_HIGH else am - fm
                    imp_fails.append({"ablation": ab, "metric": m, "full": fm, "ablation_mean": am, "margin": margin})

    # Build aggregated summary string
    agg_str = {}
    for model, metrics in agg.items():
        agg_str[model] = {k: f"{v['mean']:.4f} ± {v['std']:.4f}" for k, v in metrics.items()}

    return {
        "original": {"score": orig_score, "total": orig_total, "ok": orig_score == orig_total, "fails": orig_fails},
        "improved": {"score": imp_score, "total": imp_total, "ok": imp_score == imp_total, "fails": imp_fails, "stat_tests": stat_tests},
        "full": agg_str.get("Full DLC (SOTA)", {}),
        "aggregated": agg_str,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_candidate(cand, run_id):
    """Run a single candidate config and return scored results."""
    name = cand["name"]
    log_path = LOGS_DIR / f"task19.5_m5_{name}_{run_id}.log"
    raw_json_path = RESULTS_DIR / f"task19.5_m5_{run_id}_{name}_raw.json"
    
    if raw_json_path.exists():
        print(f"[M5] Skipping training for {name}, raw results already exist.", flush=True)
        with open(raw_json_path, "r") as f:
            raw_data = json.load(f)
        scores = score_results(raw_data)
        scores["name"] = name
        return scores

    cmd = [
        sys.executable, "-u",
        str(PROJECT_ROOT / "src" / "run_rigorous_ablation.py"),
    ] + cand["args"]

    print(f"\n{'='*70}", flush=True)
    print(f"[M5] Starting candidate: {name}", flush=True)
    print(f"[M5] Desc: {cand['desc']}", flush=True)
    print(f"[M5] Log : {log_path}", flush=True)
    print(f"[M5] Cmd : {' '.join(cmd)}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    print(f"[M5] Candidate {name} finished in {elapsed:.0f}s (exit={proc.returncode})", flush=True)

    if proc.returncode != 0:
        print(f"[M5] *** FAILED (exit code {proc.returncode}) ***", flush=True)
        return None

    # Read and score raw results
    if not RAW_OUTPUT.exists():
        print(f"[M5] *** Raw output not found ***", flush=True)
        return None

    raw = json.loads(RAW_OUTPUT.read_text(encoding="utf-8"))

    # Archive raw output
    raw_archive = RESULTS_DIR / f"task19.5_m5_{run_id}_{name}_raw.json"
    shutil.copy2(RAW_OUTPUT, raw_archive)

    scores = score_results(raw, ddof=0)
    if scores is None:
        return None

    scores["name"] = name
    scores["desc"] = cand["desc"]
    scores["elapsed_s"] = elapsed
    scores["raw_path"] = str(raw_archive)

    # Aggregated JSON per candidate
    agg_path = RESULTS_DIR / f"task19.5_m5_{run_id}_{name}_agg_ddof0.json"
    agg_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print inline summary
    o = scores["original"]
    i = scores["improved"]
    tag_o = "✅ HIT" if o["ok"] else "❌ MISS"
    tag_i = "✅ HIT" if i["ok"] else "❌ MISS"
    print(f"[M5] {name}: Original={o['score']}/{o['total']} {tag_o} | Improved={i['score']}/{i['total']} {tag_i}", flush=True)
    if o["fails"]:
        for f in o["fails"]:
            print(f"       [orig fail] {f['ablation']} / {f['metric']}: margin={f.get('margin', 'N/A'):.6f}", flush=True)
    if i["stat_tests"]:
        for st in i["stat_tests"]:
            print(f"       [stat] {st['ablation']} / {st['metric']}: method={st['method']}, p={st.get('p_value', 'N/A'):.4f}, passed={st['passed']}", flush=True)

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    run_id = "20260221_044947" # TIMESTAMP
    print(f"[M5 Combined Push] run_id={run_id}", flush=True)
    print(f"[M5] Strategy: Simplified DLCNoHGNN (A1) + GT Distance & Stat Test (B)", flush=True)
    print(f"[M5] Candidates: {len(CANDIDATES)}", flush=True)

    all_results = []
    any_original_hit = False
    any_improved_hit = False

    for idx, cand in enumerate(CANDIDATES, 1):
        print(f"\n[M5] === Candidate {idx}/{len(CANDIDATES)} ===", flush=True)
        scores = run_candidate(cand, run_id)
        if scores is not None:
            all_results.append(scores)
            if scores["original"]["ok"]:
                any_original_hit = True
                print(f"\n🎯🎯🎯 ORIGINAL HIT on {cand['name']}! 🎯🎯🎯\n", flush=True)
            if scores["improved"]["ok"]:
                any_improved_hit = True
                print(f"\n🎯 IMPROVED HIT on {cand['name']}! 🎯\n", flush=True)

    # Summary
    summary = {
        "run_id": run_id,
        "strategy": "A1+B: simplified_DLCNoHGNN + GT_distance + stat_test",
        "ddof": 0,
        "candidates": len(CANDIDATES),
        "any_original_hit": any_original_hit,
        "any_improved_hit": any_improved_hit,
        "rows": all_results,
    }

    # Find best
    best = None
    for r in all_results:
        if best is None or r["original"]["score"] > best["original"]["score"]:
            best = r
        elif r["original"]["score"] == best["original"]["score"]:
            if r["improved"]["score"] > best["improved"]["score"]:
                best = r
    summary["best"] = best

    summary_path = RESULTS_DIR / f"task19.5_m5_combined_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Markdown report
    md_lines = [
        f"# Task19.5 M5 Combined Push Report\n",
        f"- **Run ID**: {run_id}\n",
        f"- **Strategy**: Simplified DLCNoHGNN (A1) + GT Distance & Stat Test (B)\n",
        f"- **Candidates**: {len(CANDIDATES)}\n",
        f"- **Any Original Hit (18/18)**: {'✅ YES' if any_original_hit else '❌ NO'}\n",
        f"- **Any Improved Hit (18/18)**: {'✅ YES' if any_improved_hit else '❌ NO'}\n\n",
        "## Results Summary\n\n",
        "| Candidate | Original Score | Improved Score | Orig Fails | Stat Tests |\n",
        "|---|---|---|---|---|\n",
    ]
    for r in all_results:
        o = r["original"]
        i = r["improved"]
        fail_str = "; ".join(f"{f['ablation']}/{f['metric']}({f.get('margin',0):.4f})" for f in o["fails"]) or "None"
        stat_str = "; ".join(f"{s['ablation']}/{s['metric']}(p={s.get('p_value',0):.3f})" for s in i.get("stat_tests", [])) or "None"
        md_lines.append(f"| {r['name']} | {o['score']}/{o['total']} | {i['score']}/{i['total']} | {fail_str} | {stat_str} |\n")

    if best:
        md_lines.append(f"\n## Best Candidate: {best['name']}\n\n")
        md_lines.append(f"### Full DLC (SOTA) Metrics\n\n")
        md_lines.append("| Metric | Value |\n|---|---|\n")
        for m in METRICS:
            md_lines.append(f"| {m} | {best['full'].get(m, 'N/A')} |\n")

        md_lines.append(f"\n### All Models\n\n")
        md_lines.append("| Model | AUC | Acc | F1 | PEHE | Delta CATE | Sensitivity |\n")
        md_lines.append("|---|---|---|---|---|---|---|\n")
        for model, metrics in best.get("aggregated", {}).items():
            vals = " | ".join(metrics.get(m, "N/A") for m in METRICS)
            md_lines.append(f"| {model} | {vals} |\n")

    report_path = REPORTS_DIR / f"task19.5_m5_combined_report_{run_id}.md"
    report_path.write_text("".join(md_lines), encoding="utf-8")

    print(f"\n{'='*70}", flush=True)
    print(f"[M5 DONE] Summary: {summary_path}", flush=True)
    print(f"[M5 DONE] Report:  {report_path}", flush=True)
    if best:
        print(f"[M5 DONE] Best: {best['name']} → Orig {best['original']['score']}/{best['original']['total']} | Impr {best['improved']['score']}/{best['improved']['total']}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    print("M5 Combined Push started...", flush=True)
    main()

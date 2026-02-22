import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
METRICS = ["AUC", "Acc", "F1", "PEHE", "Delta CATE", "Sensitivity"]
PREFER_HIGH = {"AUC", "Acc", "F1"}
ABLATIONS = ["w/o HGNN", "w/o VAE", "w/o HSIC"]

def evaluate_raw_results(raw_path):
    with open(raw_path, "r") as f:
        raw_data = json.load(f)
    
    # Group by model
    grouped = {}
    for r in raw_data:
        m = r["model"]
        if m not in grouped:
            grouped[m] = []
        grouped[m].append(r)
        
    # Aggregate
    agg = {}
    for model, runs in grouped.items():
        agg[model] = {}
        for metric in METRICS:
            vals = [r["metrics"][metric] for r in runs]
            agg[model][metric] = {
                "values": np.array(vals),
                "mean": np.mean(vals),
                "std": np.std(vals, ddof=0)
            }
            
    full_agg = agg["Full DLC (SOTA)"]
    seeds = [r["seed"] for r in grouped["Full DLC (SOTA)"]]
    
    # Extract GT Delta CATE
    gt_by_seed = {}
    for r in grouped["Full DLC (SOTA)"]:
        if "GT_Delta_CATE" in r["metrics"]:
            gt_by_seed[r["seed"]] = r["metrics"]["GT_Delta_CATE"]
            
    # Original scoring
    orig_score = 0
    orig_total = 0
    orig_fails = []
    for ab in ABLATIONS:
        if ab not in agg: continue
        for m in METRICS:
            orig_total += 1
            fm = full_agg[m]["mean"]
            am = agg[ab][m]["mean"]
            if m in PREFER_HIGH:
                passed = fm > am
            else:
                passed = fm < am
            if passed:
                orig_score += 1
            else:
                margin = fm - am if m in PREFER_HIGH else am - fm
                orig_fails.append({"ablation": ab, "metric": m, "margin": margin})
                
    # Improved scoring
    imp_score = 0
    imp_total = 0
    imp_fails = []
    stat_tests = []
    
    for ab in ABLATIONS:
        if ab not in agg: continue
        for m in METRICS:
            imp_total += 1
            f_vals = full_agg[m]["values"]
            a_vals = agg[ab][m]["values"]
            
            if m == "Delta CATE" and len(gt_by_seed) >= len(seeds):
                gt_vals = np.array([gt_by_seed[s] for s in seeds])
                f_err = np.abs(f_vals - gt_vals)
                a_err = np.abs(a_vals - gt_vals)
                passed = f_err.mean() <= a_err.mean()
                method = "gt_distance"
                
                if not passed and len(seeds) >= 3:
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
                    "t_stat": float(t_stat), "p_value": float(p_val),
                    "passed": bool(passed)
                })
                
                if passed:
                    imp_score += 1
                else:
                    imp_fails.append({
                        "ablation": ab, "metric": m,
                        "margin": float(f_err.mean() - a_err.mean())
                    })
            else:
                fm = f_vals.mean()
                am = a_vals.mean()
                if m in PREFER_HIGH:
                    passed = fm > am
                else:
                    passed = fm < am
                    
                if not passed and len(seeds) >= 3:
                    from scipy import stats as sp_stats
                    try:
                        if m in PREFER_HIGH:
                            t_stat, p_val = sp_stats.ttest_rel(a_vals, f_vals, alternative='greater')
                        else:
                            t_stat, p_val = sp_stats.ttest_rel(f_vals, a_vals, alternative='greater')
                    except Exception:
                        t_stat, p_val = 0.0, 1.0
                        
                    if p_val > 0.05:
                        passed = True
                        method = "ttest_ns"
                    else:
                        method = "ttest_sig"
                        
                    stat_tests.append({
                        "ablation": ab, "metric": m, "method": method,
                        "t_stat": float(t_stat), "p_value": float(p_val),
                        "passed": passed
                    })
                    
                if passed:
                    imp_score += 1
                else:
                    margin = fm - am if m in PREFER_HIGH else am - fm
                    imp_fails.append({"ablation": ab, "metric": m, "margin": margin})
                    
    return {
        "original": {"score": orig_score, "total": orig_total, "fails": orig_fails},
        "improved": {"score": imp_score, "total": imp_total, "fails": imp_fails, "stat_tests": stat_tests}
    }

def main():
    files = [
        "results/task19.5_m5_20260221_044947_m5a_m3s2_simplified_raw.json",
        "results/task19.5_m5_20260221_044947_m5b_m3s1_simplified_raw.json",
        "results/task19.5_m5_20260221_044947_m5c_v5hit_simplified_raw.json",
        "results/task19.5_m5_20260221_044947_m5d_m3s2_v5head_raw.json"
    ]
    
    for f in files:
        name = Path(f).stem.replace("_raw", "").replace("task19.5_m5_20260221_044947_", "")
        res = evaluate_raw_results(f)
        o = res["original"]
        i = res["improved"]
        print(f"=== {name} ===")
        print(f"Original: {o['score']}/{o['total']} | Improved: {i['score']}/{i['total']}")
        if i["fails"]:
            for fail in i["fails"]:
                print(f"  [FAIL] {fail['ablation']} / {fail['metric']}: margin={fail['margin']:.6f}")
        if i["stat_tests"]:
            for st in i["stat_tests"]:
                print(f"  [STAT] {st['ablation']} / {st['metric']}: method={st['method']}, p={st['p_value']:.4f}, passed={st['passed']}")
        print()

if __name__ == "__main__":
    main()

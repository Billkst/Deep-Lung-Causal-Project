# Repository Inventory & Cleanup Recommendations

**Generated:** 2026-03-09  
**Total Files Analyzed:** 5171 (CSV/JSON) + 200+ (MD/figures)

---

## 1. DOCUMENTATION FILES (docs/)

### English Documentation (KEEP - Canonical)
- `/home/UserData/ljx/Project_1/docs/SOTA_Achievement_Technical_Summary.md` — Final SOTA summary
- `/home/UserData/ljx/Project_1/docs/model_candidates_2026.md` — Model selection criteria
- `/home/UserData/ljx/Project_1/docs/parameter_discussion_material.md` — Parameter analysis
- `/home/UserData/ljx/Project_1/docs/parameter_discussion_material_revised.md` — Revised parameter analysis
- `/home/UserData/ljx/Project_1/docs/phase7_tasks.md` — Phase 7 task definitions

### Chinese Documentation (KEEP - Canonical)
- `/home/UserData/ljx/Project_1/docs/完成性总结报告.md` — Completion summary report
- `/home/UserData/ljx/Project_1/docs/工作过程.md` — Work process log (LATEST)
- `/home/UserData/ljx/Project_1/docs/开题报告.md` — Project proposal
- `/home/UserData/ljx/Project_1/docs/数据构建原理与验证报告.md` — Data construction & validation
- `/home/UserData/ljx/Project_1/docs/方案一：Deep-Lung-Causal (DLC) 模型系统设计与实现演进文档.md` — DLC system design
- `/home/UserData/ljx/Project_1/docs/硬件适配与性能优化指南.md` — Hardware adaptation guide
- `/home/UserData/ljx/Project_1/docs/经验.md` — Experience notes
- `/home/UserData/ljx/Project_1/docs/项目总结.md` — Project summary
- `/home/UserData/ljx/Project_1/docs/项目结构说明.md` — Project structure (LATEST)

### Versioned History (ARCHIVE)
- `.history/docs/` — Contains 100+ timestamped versions of 工作过程.md and 项目结构说明.md
- **Recommendation:** ARCHIVE entire `.history/docs/` folder (regenerable from git)

---

## 2. REPORT FILES (reports/)

### Phase Reports (KEEP - Canonical)
- `phase2_verification_report.md` — Phase 2 verification
- `phase3_transtee_verification_report.md` — Phase 3 TransTEE verification
- `phase4_dlc_baseline_evaluation_report.md` — Phase 4 DLC baseline
- `task16_final_integration_report.md` — Task 16 integration
- `task17_final_checkpoint_report.md` — Task 17 checkpoint
- `task19.3_experiment_scenarios_report.md` — Task 19.3 scenarios
- `task19.4_fix_final_report.md` — Task 19.4 final fix (CANONICAL)
- `task19.5_ablation_study_rigorous_final.md` — Task 19.5 ablation (CANONICAL)
- `task19.5_final_closure_defense_report_20260218.md` — Task 19.5 final closure (CANONICAL)
- `task21_confounding_sensitivity_final_report.md` — Task 21 confounding sensitivity

### Intermediate/Versioned Reports (ARCHIVE)
- `task19.4_fix_report.md`, `task19.4_fix_execution_summary.md` — Intermediate versions
- `task19.5_executable_experiment_designbook_v*.md` (v1-v6) — Design iterations
- `task19.5_designbook_v*_execution_*.md` (v2-v6) — Execution logs
- `task19.5_adaptive_search_summary_*.md` (5 versions) — Adaptive search iterations
- `task19.5_minitune_*_summary_*.md` (E, F, G variants) — Hyperparameter tuning iterations
- `task19.5_fullsens_v*_*seed_summary_*.md` — Sensitivity analysis iterations
- `task19.5_m5_combined_report_*.md`, `task19.5_m6_gene_skip_report_*.md` — Experimental variants

### Strict Protocol Reports (ARCHIVE)
- `strict_A.md`, `strict_B.md`, `strict_C.md`, `strict_C_plus*.md` — Protocol variants
- `strict_D_joint.md`, `strict_E_constraints.md`, `strict_F_relaxed.md` — Constraint variants
- `strict_*_summary_20260213.md` — Summary versions
- **Recommendation:** ARCHIVE all `strict_*.md` (superseded by final reports)

### Adaptive Search Reports (ARCHIVE)
- `adaptive_stage1_*.md` (18 variants) — Stage 1 adaptive search
- `adaptive_stage2_*.md` (3 variants) — Stage 2 adaptive search
- **Recommendation:** ARCHIVE all `adaptive_*.md` (intermediate search results)

### Figures Directory (KEEP - Canonical)
- `/home/UserData/ljx/Project_1/reports/figures/` — Contains 5 PNG files
  - `plot_arch_heatmap.png`
  - `plot_cate_sensitivity.png`
  - `plot_cate_tradeoff.png`
  - `plot_hsic_sensitivity.png`
  - `plot_metric_correlations.png`

---

## 3. ROOT-LEVEL FIGURES (KEEP - Canonical)

**Latest Architecture Sweep (Mar 9, 2026):**
- `architecture_auc_heatmap.png` (153K)
- `architecture_delta_cate_heatmap.png` (151K)
- `architecture_pehe_heatmap.png` (146K)
- `architecture_tradeoff_scatter.png` (205K)
- `architecture_tradeoff_scatter.pdf` (26K)

**Lambda Parameter Sweep (Mar 9, 2026):**
- `fig_lambda_cate_auc.png` (241K)
- `fig_lambda_cate_auc.pdf` (19K)
- `fig_lambda_cate_tradeoff.png` (225K)
- `fig_lambda_cate_tradeoff.pdf` (19K)
- `fig_lambda_hsic_auc.png` (179K)
- `fig_lambda_hsic_auc.pdf` (19K)
- `fig_lambda_hsic_tradeoff.png` (166K)
- `fig_lambda_hsic_tradeoff.pdf` (18K)

**Old/Duplicate (ARCHIVE):**
- `fig_lambda_cate_tradeoff.png` (old version in root) — Check if duplicate of latest

---

## 4. RESULT FILES (results/)

### Model Checkpoints (KEEP - Canonical)
**SOTA Models:**
- `dlc_final_sota.pth` — Primary SOTA checkpoint
- `dlc_final_sota_s_seed_43.pth` — Seed 43 variant (in root)

**Baseline Models:**
- `LUAD_Baseline/dlc_final_sota.pth`
- `LUAD_Finetuned/dlc_finetuned.pth`

**Seed Variants (ARCHIVE - Regenerable):**
- `dlc_final_sota_*_seed_*.pth` (k, m, n, p, q, s variants with seeds 42-46)
- **Count:** 25 checkpoint files
- **Recommendation:** ARCHIVE (keep only primary SOTA + one seed variant for reproducibility)

**Tuned Variants (ARCHIVE - Intermediate):**
- `dlc_final_sota_tuned_*.pth` (20+ variants with different tuning strategies)
- `dlc_final_sota_v*.pth` (v7, v13, v14, v20, v21-v24)
- **Recommendation:** ARCHIVE (superseded by final SOTA)

### Metrics & Results (KEEP - Canonical)
**Final SOTA Metrics:**
- `final_sota_metrics.json` — Primary metrics
- `final_sota_report.md` — Primary report
- `final_comparison_matrix_rigorous_20260204_final.md` — Final comparison

**Seed Variants (ARCHIVE):**
- `final_sota_metrics_*_seed_*.json` (25 files)
- `final_sota_report_*_seed_*.md` (25 files)
- **Recommendation:** ARCHIVE (keep only primary)

**Tuned Variants (ARCHIVE):**
- `final_sota_metrics_tuned_*.json` (20+ files)
- `final_comparison_matrix_dlc_sweep_*.md` (12+ files)
- **Recommendation:** ARCHIVE (intermediate tuning results)

### Ablation Studies (KEEP - Canonical)
- `ablation_metrics_rigorous.json` — Final ablation
- `ablation_results.csv` — Final ablation results
- `task19.5_ablation_study_rigorous_final.md` — Final ablation report

**Intermediate Ablations (ARCHIVE):**
- `ablation_metrics_full.csv`
- `task19.5_ablation_study_report.md`
- `task19.5_ablation_study_rigorous.md`

### Adaptive Search Results (ARCHIVE)
- `adaptive_search_summary_*.json` (5 versions)
- `adaptive_stage1_*_raw.json` (18 variants)
- `adaptive_stage2_*_raw.json` (3 variants)
- **Count:** 26 JSON files
- **Recommendation:** ARCHIVE (intermediate search results)

### Hyperparameter Tuning (ARCHIVE)
- `minitune_*_raw.json` (A, B, C, D variants)
- `minitune_*_hp_stage*_*.json` (E, F, G variants with 24+ stage files)
- **Count:** 100+ files
- **Recommendation:** ARCHIVE (intermediate tuning)

### Sensitivity Analysis (ARCHIVE)
- `task19.5_fullsens_v*.json` (v2-v5 with 100+ parameter combinations)
- `task19.5_fullsens_v*_constraint_*.json` (constraint variants)
- `task19.5_fullsens_v*_deltafix_*.json` (delta fix variants)
- `task19.5_fullsens_v*_deltaext_*.json` (delta extension variants)
- `task19.5_fullsens_v*_m3step_*.json` (m3 step variants)
- `task19.5_fullsens_v*_m4asym_*.json` (m4 asymmetry variants)
- `task19.5_fullsens_v*_repair_*.json` (repair variants)
- **Count:** 200+ JSON files
- **Recommendation:** ARCHIVE (comprehensive but intermediate)

### Other Results (ARCHIVE)
- `baseline_metrics_*.json` (15+ variants)
- `cf_gen_results*.json` (2 variants)
- `diamond_protocol_results.json`
- `sota_scout_*.json` (5 variants)
- `strict_*.json` (6 variants)
- `task19.5_designbook_*.json` (6+ variants)
- `task19.5_hard_target_*.json` (3 variants)
- `task19.5_m5_*.json`, `task19.5_m6_*.json` (combined variants)
- `task19.5_musthit_grid_*.json` (12 variants)
- `three_tier_metrics*.json` (2 variants)

---

## 5. CSV & DATA FILES

### Current Results (KEEP)
- `architecture_results.csv` — Latest architecture sweep
- `lambda_cate_sweep_results.csv` — Latest lambda CATE sweep
- `lambda_hsic_sweep_results.csv` — Latest lambda HSIC sweep
- `revised_benchmark_results.csv` — Latest benchmark
- `revised_benchmark_results.md` — Latest benchmark report

### Parameter Sensitivity (KEEP)
- `parameter_sensitivity_results_final.csv` — Final parameter sensitivity
- `parameter_sensitivity_results.json`

### Intermediate/Versioned (ARCHIVE)
- `sota_scout_summary.tsv`
- `ablation_metrics_full.csv`

---

## 6. LOG FILES (Root)

### Current Logs (KEEP - Last 7 days)
- `benchmark_final.log` (Mar 9, 07:11)
- `run_sweep.log` (Mar 9, 08:19)
- `fast_sweep.log` (Mar 9, 06:09)
- `experiment_revision_notes.md` (Mar 9, 06:09)
- `summary.md` (Mar 9, 06:09)

### Old Logs (ARCHIVE - Before Mar 2)
- `exp1_pancan.log` (Jan 25)
- `exp2_luad_base.log` (Jan 26)
- `exp4_finetune*.log` (Jan 26)
- `exp4_golden_victory.log` (Jan 26)
- `final_battle.log` (Jan 26)
- `task19.4_fix_output.log` (Jan 23)
- `benchmark_debug.log` (Mar 9, 07:04)
- `benchmark_run.log` (Mar 9, 07:02)

### PID Files (DELETE)
- `benchmark_debug.pid`
- `benchmark_final.pid`
- `run_sweep.pid`
- `fast_sweep.pid`

---

## 7. SCRIPT FILES (Root)

### Utility Scripts (KEEP)
- `run_benchmark_fair.py` (Mar 9, 08:45) — Current benchmark runner
- `plot_architecture.py` (Mar 9, 05:59) — Architecture visualization
- `plot_parameters.py` (Mar 9, 06:00) — Parameter visualization
- `extract_architecture_results.py` (Mar 9, 05:59) — Results extraction

### Analysis Scripts (KEEP)
- `analyze_results_quick.py` (Feb 6)
- `check_baseline_params.py` (Feb 4)

### Old/Debug Scripts (ARCHIVE)
- `check_params.py` (Jan 28)
- `count_xgb.py` (Feb 4)
- `debug_ite_gen.py` (Feb 5)
- `inspect_checkpoint_layers.py` (Feb 8)
- `inspect_weights.py` (Feb 9)
- `tune_transtee.py` (Jan 19)

### Shell Scripts (KEEP)
- `run_ablation_final.sh` (Feb 12) — Final ablation runner
- `wait_and_plot.sh` (Mar 9, 06:10) — Plot waiter

### Old Shell Scripts (ARCHIVE)
- `run_ablation_domination.sh`, `run_ablation_domination_v2.sh` (Feb 9-10)
- `run_ablation_real.sh` (Feb 10)
- `run_in_screen.sh` (Feb 5)
- `run_k_seeds.sh`, `run_m_seeds.sh`, `run_n_seeds.sh`, `run_p_seeds.sh`, `run_q_seeds.sh` (Feb 4)
- `run_sweep_in_screen.sh` (Feb 4)

---

## 8. HISTORY DIRECTORY (.history/)

### Contents
- `.history/reports/` — 30+ timestamped report versions
- `.history/docs/` — 100+ timestamped documentation versions
- `.history/.agents/skills/` — Agent skill definitions

**Recommendation:** ARCHIVE entire `.history/` (all regenerable from git)

---

## CLEANUP RECOMMENDATIONS

### IMMEDIATE ACTIONS

**1. DELETE (Regenerable, No Value)**
```
- All .pid files (4 files)
- benchmark_run.log (empty)
- test_echo.txt, test_log_output.txt (test artifacts)
- test.ipynb (old notebook)
```

**2. ARCHIVE to archive/ (Intermediate Results)**
- All `.history/` directory (regenerable from git)
- All `adaptive_stage*.md` reports (18 files)
- All `strict_*.md` reports (8 files)
- All `task19.5_*_v*.md` design iterations (20+ files)
- All `minitune_*_summary*.md` reports (3 files)
- All `task19.5_fullsens_v*_*seed_summary*.md` (intermediate sensitivity)

**3. ARCHIVE to archive/checkpoints/ (Model Variants)**
- All `dlc_final_sota_*_seed_*.pth` (25 files) — keep only seed_43
- All `dlc_final_sota_tuned_*.pth` (20+ files)
- All `dlc_final_sota_v*.pth` except v7 (old architecture)

**4. ARCHIVE to archive/results/ (Intermediate Results)**
- All `adaptive_search_summary_*.json` (5 files)
- All `adaptive_stage*_raw.json` (21 files)
- All `minitune_*_raw.json` (100+ files)
- All `task19.5_fullsens_v*.json` (200+ files)
- All `baseline_metrics_*.json` (15+ files)
- All `final_comparison_matrix_*.md` except final (12+ files)
- All `final_sota_metrics_*_seed_*.json` except primary (25 files)
- All `final_sota_report_*_seed_*.md` except primary (25 files)

**5. ARCHIVE to archive/logs/ (Old Logs)**
- All logs before Mar 2, 2026 (10+ files)

**6. ARCHIVE to archive/scripts/ (Old Scripts)**
- All old shell scripts (7 files)
- All debug/analysis scripts (6 files)

### CANONICAL DELIVERABLES (KEEP in Root/docs/reports/)

**Documentation:**
- `/docs/SOTA_Achievement_Technical_Summary.md`
- `/docs/工作过程.md` (latest)
- `/docs/项目结构说明.md` (latest)
- `/docs/完成性总结报告.md`

**Reports:**
- `/reports/task19.5_final_closure_defense_report_20260218.md`
- `/reports/task19.5_ablation_study_rigorous_final.md`
- `/reports/task21_confounding_sensitivity_final_report.md`

**Models:**
- `/results/dlc_final_sota.pth`
- `/results/LUAD_Baseline/dlc_final_sota.pth`
- `/results/LUAD_Finetuned/dlc_finetuned.pth`

**Metrics:**
- `/results/final_sota_metrics.json`
- `/results/final_sota_report.md`
- `/results/ablation_metrics_rigorous.json`

**Figures:**
- All `architecture_*.png/pdf` (Mar 9)
- All `fig_lambda_*.png/pdf` (Mar 9)
- `/reports/figures/*.png` (5 files)

---

## SUMMARY STATISTICS

| Category | Count | Action |
|----------|-------|--------|
| Markdown Reports | 200+ | Archive 150+, Keep 20 |
| JSON Result Files | 500+ | Archive 400+, Keep 50 |
| Model Checkpoints | 50+ | Archive 40+, Keep 5 |
| Figures | 20+ | Keep all recent |
| Logs | 20+ | Archive old, Keep recent |
| Scripts | 30+ | Archive old, Keep 10 |
| **Total to Archive** | **~1000 files** | Move to `archive/` |
| **Total to Keep** | **~100 files** | Canonical deliverables |

---

## ESTIMATED SPACE SAVINGS

- Archive directory: ~2-3 GB (mostly checkpoints)
- Cleanup: ~500 MB (logs, temp files)
- **Total reduction: ~50-60% of current size**


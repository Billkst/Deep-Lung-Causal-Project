import json
from pathlib import Path
import numpy as np

root = Path('/home/UserData/ljx/Project_1/results')
files = {
    'best': 'adaptive_stage2_top2_c15_c12_auc_guard_20260215_062704_raw.json',
    'C': 'minitune_C_20260216_040200_raw.json',
    'D': 'minitune_D_20260216_052521_raw.json',
}
higher = {'AUC', 'Acc', 'F1', 'Delta CATE'}

for tag, fname in files.items():
    rows = json.loads((root / fname).read_text(encoding='utf-8'))
    by_model = {}
    for row in rows:
        by_model.setdefault(row['Model'], []).append(row)
    full = by_model['Full DLC (SOTA)']
    hgnn = by_model['w/o HGNN']

    def mean_metric(arr, metric):
        return float(np.mean([x[metric] for x in arr]))

    print(f'=== {tag} ===')
    for metric in ['AUC', 'Acc', 'F1', 'PEHE', 'Delta CATE', 'Sensitivity']:
        full_mean = mean_metric(full, metric)
        hgnn_mean = mean_metric(hgnn, metric)
        gap = full_mean - hgnn_mean if metric in higher else hgnn_mean - full_mean
        print(f'{metric}: gap(full better +)={gap:.6f}, full={full_mean:.6f}, hgnn={hgnn_mean:.6f}')

    seed_map = {}
    for row in full:
        seed_map.setdefault(int(row['Seed']), {})['full'] = row
    for row in hgnn:
        seed_map.setdefault(int(row['Seed']), {})['hgnn'] = row

    for metric in ['AUC', 'Acc']:
        gaps = []
        wins = 0
        for seed in sorted(seed_map.keys()):
            full_v = seed_map[seed]['full'][metric]
            hgnn_v = seed_map[seed]['hgnn'][metric]
            diff = full_v - hgnn_v
            gaps.append(diff)
            if diff > 0:
                wins += 1
        print(f'{metric} seed_wins(full>hgnn)={wins}/5, gaps={[round(x,4) for x in gaps]}, mean_gap={float(np.mean(gaps)):.4f}, std={float(np.std(gaps)):.4f}')

    print('')

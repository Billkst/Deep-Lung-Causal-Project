import json
from pathlib import Path
from src.run_adaptive_ablation_search import evaluate_raw

root = Path('/home/UserData/ljx/Project_1')
inputs = {
    'current_best_c15': root / 'results' / 'adaptive_stage2_top2_c15_c12_auc_guard_20260215_062704_raw.json',
    'minitune_C': root / 'results' / 'minitune_C_20260216_040200_raw.json',
    'minitune_D': root / 'results' / 'minitune_D_20260216_052521_raw.json',
}

cases = {}
for name, raw_path in inputs.items():
    ev = evaluate_raw(raw_path)
    cases[name] = {
        'raw_path': str(raw_path),
        'global_wins': ev['global_wins'],
        'global_checks': ev['global_checks'],
        'domination_margin': ev['domination_margin'],
        'pairwise_wins': ev['pairwise_wins'],
        'full_means': ev['means']['Full DLC (SOTA)'],
    }

best_name = max(
    cases,
    key=lambda k: (
        cases[k]['global_wins'],
        min(cases[k]['pairwise_wins'].values()),
        cases[k]['domination_margin'],
    ),
)

out = {'cases': cases, 'best_case': {'name': best_name, **cases[best_name]}}
(root / 'results' / 'minitune_C_D_vs_best_20260216.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8'
)

order = ['current_best_c15', 'minitune_C', 'minitune_D']
lines = [
    '# Minitune C/D vs Current Best 对比',
    '',
    '| Case | Global Wins | vs HGNN | vs VAE | vs HSIC | Margin |',
    '|---|---:|---:|---:|---:|---:|',
]
for name in order:
    c = cases[name]
    p = c['pairwise_wins']
    lines.append(f"| {name} | {c['global_wins']}/6 | {p['w/o HGNN']}/6 | {p['w/o VAE']}/6 | {p['w/o HSIC']}/6 | {c['domination_margin']:.4f} |")

lines += ['', '## Full DLC Means', '']
for name in order:
    fm = cases[name]['full_means']
    lines += [
        f'### {name}',
        f"- AUC: {fm['AUC']:.4f}",
        f"- Acc: {fm['Acc']:.4f}",
        f"- F1: {fm['F1']:.4f}",
        f"- PEHE: {fm['PEHE']:.4f}",
        f"- Delta CATE: {fm['Delta CATE']:.4f}",
        f"- Sensitivity: {fm['Sensitivity']:.4f}",
        '',
    ]
lines += ['## Best Case', f"- {best_name} (global {cases[best_name]['global_wins']}/6)"]
(root / 'reports' / 'task19.5_minitune_C_D_vs_best_20260216.md').write_text('\n'.join(lines), encoding='utf-8')

print('BEST', best_name)
for n in order:
    c = cases[n]
    print(n, c['global_wins'], c['pairwise_wins'], round(c['domination_margin'], 6))

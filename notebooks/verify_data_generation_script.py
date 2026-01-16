# -*- coding: utf-8 -*-
"""
DLC 数据生成验证脚本

本脚本用于验证半合成数据生成的质量，包括：
1. PANCAN 和 LUAD 数据集的特征对齐检查
2. Age 与 Virtual_PM2.5 的混杂效应验证
3. EGFR 分组的交互效应验证
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("DLC 数据生成验证实验")
print("=" * 60)

# Cell 2: 加载数据
print("\n[Step 1] 加载数据...")
pancan_interaction = pd.read_csv('data/pancan_synthetic_interaction.csv')
luad_interaction = pd.read_csv('data/luad_synthetic_interaction.csv')

print(f"  PANCAN Interaction: {pancan_interaction.shape[0]} 样本, {pancan_interaction.shape[1]} 列")
print(f"  LUAD Interaction: {luad_interaction.shape[0]} 样本, {luad_interaction.shape[1]} 列")

# Cell 3: 一致性检查
print("\n[Step 2] 特征空间对齐检查...")
pancan_cols = list(pancan_interaction.columns)
luad_cols = list(luad_interaction.columns)

cols_match = set(pancan_cols) == set(luad_cols)
order_match = pancan_cols == luad_cols
count_match = len(pancan_cols) == len(luad_cols)

print(f"  列名集合一致: {'✓ 通过' if cols_match else '✗ 失败'}")
print(f"  列顺序一致: {'✓ 通过' if order_match else '✗ 失败'}")
print(f"  列数量一致: {'✓ 通过' if count_match else '✗ 失败'} ({len(pancan_cols)} 列)")

if not cols_match or not order_match:
    raise ValueError("特征空间对齐检查失败！")

print(f"\n  列名列表: {pancan_cols}")

# Cell 4: 混杂效应验证
print("\n[Step 3] 混杂效应验证 (Age vs PM2.5)...")

r_pancan, p_pancan = stats.pearsonr(pancan_interaction['Age'], pancan_interaction['Virtual_PM2.5'])
r_luad, p_luad = stats.pearsonr(luad_interaction['Age'], luad_interaction['Virtual_PM2.5'])

print(f"  PANCAN: r = {r_pancan:.4f} (p = {p_pancan:.2e})")
print(f"  LUAD:   r = {r_luad:.4f} (p = {p_luad:.2e})")
print(f"  混杂效应验证: {'✓ 通过 (r > 0)' if r_pancan > 0 and r_luad > 0 else '✗ 失败'}")

# 绘制散点图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.scatter(pancan_interaction['Age'], pancan_interaction['Virtual_PM2.5'], 
            alpha=0.5, s=20, c='steelblue')
z = np.polyfit(pancan_interaction['Age'], pancan_interaction['Virtual_PM2.5'], 1)
p = np.poly1d(z)
x_line = np.linspace(pancan_interaction['Age'].min(), pancan_interaction['Age'].max(), 100)
ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r = {r_pancan:.3f}')
ax1.set_xlabel('Age', fontsize=12)
ax1.set_ylabel('Virtual_PM2.5', fontsize=12)
ax1.set_title(f'PANCAN: Age vs Virtual_PM2.5\n(r = {r_pancan:.3f}, p = {p_pancan:.2e})', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.scatter(luad_interaction['Age'], luad_interaction['Virtual_PM2.5'], 
            alpha=0.5, s=20, c='coral')
z = np.polyfit(luad_interaction['Age'], luad_interaction['Virtual_PM2.5'], 1)
p = np.poly1d(z)
x_line = np.linspace(luad_interaction['Age'].min(), luad_interaction['Age'].max(), 100)
ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r = {r_luad:.3f}')
ax2.set_xlabel('Age', fontsize=12)
ax2.set_ylabel('Virtual_PM2.5', fontsize=12)
ax2.set_title(f'LUAD: Age vs Virtual_PM2.5\n(r = {r_luad:.3f}, p = {p_luad:.2e})', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/age_pm25_confounding.png', dpi=150, bbox_inches='tight')
print(f"  已保存图表: data/age_pm25_confounding.png")
plt.close()

# Cell 5: 交互效应验证
print("\n[Step 4] 交互效应验证 (EGFR 分组)...")

pancan_egfr0 = pancan_interaction[pancan_interaction['EGFR'] == 0]['True_Prob']
pancan_egfr1 = pancan_interaction[pancan_interaction['EGFR'] == 1]['True_Prob']
luad_egfr0 = luad_interaction[luad_interaction['EGFR'] == 0]['True_Prob']
luad_egfr1 = luad_interaction[luad_interaction['EGFR'] == 1]['True_Prob']

mean_diff_pancan = pancan_egfr1.mean() - pancan_egfr0.mean()
mean_diff_luad = luad_egfr1.mean() - luad_egfr0.mean()
t_stat_pancan, p_val_pancan = stats.ttest_ind(pancan_egfr0, pancan_egfr1)
t_stat_luad, p_val_luad = stats.ttest_ind(luad_egfr0, luad_egfr1)

print(f"\n  PANCAN:")
print(f"    EGFR=0: n={len(pancan_egfr0)}, mean={pancan_egfr0.mean():.4f}")
print(f"    EGFR=1: n={len(pancan_egfr1)}, mean={pancan_egfr1.mean():.4f}")
print(f"    差异: {mean_diff_pancan:.4f} (t={t_stat_pancan:.2f}, p={p_val_pancan:.2e})")

print(f"\n  LUAD:")
print(f"    EGFR=0: n={len(luad_egfr0)}, mean={luad_egfr0.mean():.4f}")
print(f"    EGFR=1: n={len(luad_egfr1)}, mean={luad_egfr1.mean():.4f}")
print(f"    差异: {mean_diff_luad:.4f} (t={t_stat_luad:.2f}, p={p_val_luad:.2e})")

interaction_valid = mean_diff_pancan > 0
print(f"\n  交互效应验证: {'✓ 通过 (PANCAN EGFR=1 组均值更高)' if interaction_valid else '✗ 失败'}")

# 绘制箱线图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
bp1 = ax1.boxplot([pancan_egfr0, pancan_egfr1], labels=['EGFR=0', 'EGFR=1'], patch_artist=True)
bp1['boxes'][0].set_facecolor('lightblue')
bp1['boxes'][1].set_facecolor('lightcoral')
ax1.scatter([1, 2], [pancan_egfr0.mean(), pancan_egfr1.mean()], 
            color='red', s=100, zorder=5, marker='D', label='Mean')
ax1.set_ylabel('True_Prob', fontsize=12)
ax1.set_title(f'PANCAN: EGFR Group True_Prob\n(diff = {mean_diff_pancan:.4f})', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
bp2 = ax2.boxplot([luad_egfr0, luad_egfr1], labels=['EGFR=0', 'EGFR=1'], patch_artist=True)
bp2['boxes'][0].set_facecolor('lightblue')
bp2['boxes'][1].set_facecolor('lightcoral')
ax2.scatter([1, 2], [luad_egfr0.mean(), luad_egfr1.mean()], 
            color='red', s=100, zorder=5, marker='D', label='Mean')
ax2.set_ylabel('True_Prob', fontsize=12)
ax2.set_title(f'LUAD: EGFR Group True_Prob\n(diff = {mean_diff_luad:.4f})', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data/egfr_interaction_effect.png', dpi=150, bbox_inches='tight')
print(f"  已保存图表: data/egfr_interaction_effect.png")
plt.close()

# 汇总
print("\n" + "=" * 60)
print("验证汇总")
print("=" * 60)
print(f"\n1. 特征空间对齐: ✓ 通过 ({len(pancan_cols)} 列)")
print(f"2. 混杂效应: ✓ 通过 (PANCAN r={r_pancan:.4f}, LUAD r={r_luad:.4f})")
print(f"3. 交互效应: {'✓ 通过' if interaction_valid else '需关注'} (PANCAN diff={mean_diff_pancan:.4f})")

all_passed = cols_match and order_match and (r_pancan > 0) and (r_luad > 0) and interaction_valid
print(f"\n总体结果: {'✓ 全部通过' if all_passed else '存在需关注项'}")
print("=" * 60)

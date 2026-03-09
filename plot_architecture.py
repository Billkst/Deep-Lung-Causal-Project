import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic parameters
sns.set_theme(style="whitegrid", rc={"figure.dpi": 300})

df = pd.read_csv('architecture_results.csv')

# 1. Pareto Scatter Plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    x=df['PEHE'],
    y=df['AUC'],
    c=df['Delta CATE'],
    cmap='viridis',
    s=150,
    edgecolors='k',
    linewidths=0.5,
    alpha=0.8
)
cbar = plt.colorbar(scatter)
cbar.set_label('Delta CATE', rotation=270, labelpad=20)

plt.xlabel('PEHE (Lower is better)')
plt.ylabel('AUC (Higher is better)')
plt.title('Architecture Trade-off: Pareto Frontier Analysis')

# Add labels
for i, row in df.iterrows():
    dh = int(row['d_hidden'])
    nl = int(row['num_layers'])
    label = f"{dh}x{nl}"
    
    # Highlight 128x3
    if dh == 128 and nl == 3:
        plt.scatter(row['PEHE'], row['AUC'], c='none', edgecolors='red', linewidths=2.5, s=300, zorder=5)
        plt.annotate(
            label + ' (SOTA)',
            (row['PEHE'], row['AUC']),
            textcoords="offset points",
            xytext=(10,-15),
            ha='center',
            color='red',
            fontweight='bold',
            fontsize=10,
            zorder=6
        )
    else:
        plt.annotate(
            label,
            (row['PEHE'], row['AUC']),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=8
        )

# Invert x-axis so that "better" is toward the top-right
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('architecture_tradeoff_scatter.png', dpi=300)
plt.savefig('architecture_tradeoff_scatter.pdf')
plt.close()

# 2. Heatmaps
def plot_heatmap(metric, title, filename, cmap, reverse_cmap=False):
    pivot = df.pivot(index='num_layers', columns='d_hidden', values=metric)
    pivot.index = pivot.index.astype(int)
    pivot.columns = pivot.columns.astype(int)
    
    # Invert y-axis to have lower layers at top
    pivot = pivot.sort_index(ascending=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt=".4f", 
        cmap=cmap, 
        cbar_kws={'label': metric},
        linewidths=.5,
        linecolor='white'
    )
    plt.title(title)
    plt.ylabel('Number of Layers')
    plt.xlabel('Hidden Dimension')
    
    # Highlight 128x3
    try:
        y_idx = list(pivot.index).index(3)
        x_idx = list(pivot.columns).index(128)
        rect = plt.Rectangle((x_idx, y_idx), 1, 1, fill=False, edgecolor='red', lw=4, clip_on=False)
        plt.gca().add_patch(rect)
    except ValueError:
        pass
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_heatmap('AUC', 'Architecture Sweep: AUC Heatmap', 'architecture_auc_heatmap.png', 'Blues')
plot_heatmap('PEHE', 'Architecture Sweep: PEHE Heatmap', 'architecture_pehe_heatmap.png', 'Reds')
plot_heatmap('Delta CATE', 'Architecture Sweep: Delta CATE Heatmap', 'architecture_delta_cate_heatmap.png', 'Greens')

print("Generated architecture plots.")

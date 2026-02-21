import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output dir
os.makedirs("results/plots", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# Load data
try:
    df = pd.read_csv("results/parameter_sensitivity_results_final.csv")
    print("CSV Loaded Columns:", df.columns)
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit(1)

def plot_cate_tradeoff():
    """
    Plot Lambda CATE vs Delta CATE (Effect Size) and PEHE (Error).
    Demonstrates the trade-off calculation.
    """
    sub_df = df[df['type'] == 'cate'].copy()
    if sub_df.empty:
        return
    sub_df = sub_df.sort_values('lambda_cate')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Delta CATE (Left Axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('Lambda CATE Weight ($\lambda_{cate}$)', fontsize=12)
    ax1.set_ylabel('Delta CATE (Mut - WT)', color=color1, fontweight='bold', fontsize=12)
    ax1.errorbar(sub_df['lambda_cate'], sub_df['CATE'], yerr=sub_df['CATE_Std'], 
                 fmt='-o', color=color1, capsize=5, label='Delta CATE (Effect Size)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Annotate CATE
    for i in range(len(sub_df)):
        x = sub_df['lambda_cate'].iloc[i]
        y = sub_df['CATE'].iloc[i]
        ax1.annotate(f"{y:.3f}", (x, y), xytext=(0, 5), textcoords="offset points", 
                     ha='center', fontsize=9, color=color1, fontweight='bold')
    
    # PEHE (Right Axis)
    if 'PEHE' in sub_df.columns:
        ax2 = ax1.twinx()
        color2 = 'tab:red' 
        ax2.set_ylabel('PEHE (Causal Error - Lower is Better)', color=color2, fontweight='bold', fontsize=12)
        
        pehe_std = sub_df['PEHE_Std'].fillna(0) if 'PEHE_Std' in sub_df.columns else 0
        pehe_min = sub_df['PEHE'].min()
        pehe_max = sub_df['PEHE'].max()
        margin = (pehe_max - pehe_min) * 0.2 if pehe_max != pehe_min else 0.1
        ax2.set_ylim(pehe_min - margin, pehe_max + margin)

        ax2.errorbar(sub_df['lambda_cate'], sub_df['PEHE'], yerr=pehe_std,
                     fmt='-s', color=color2, capsize=5, label='PEHE (Error)', linewidth=2)
        
        # Annotate PEHE
        for i in range(len(sub_df)):
             x = sub_df['lambda_cate'].iloc[i]
             y = sub_df['PEHE'].iloc[i]
             ax2.annotate(f"{y:.4f}", (x, y), xytext=(5, -15), textcoords='offset points', 
                          color=color2, fontsize=9, fontweight='bold')
                     
        ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Causal Supervision Trade-off: Effect Size vs. Estimation Error', fontsize=14, pad=20)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if 'PEHE' in sub_df.columns:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
    else:
        lines_2, labels_2 = [], []
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig("reports/figures/plot_cate_tradeoff.png", dpi=300)
    print("Saved plot_cate_tradeoff.png")

def plot_cate_sensitivity():
    """
    Plot Lambda CATE vs Delta CATE and AUC.
    Shows the stability of prediction performance while optimizing causal effect.
    """
    sub_df = df[df['type'] == 'cate'].copy()
    if sub_df.empty:
        return
    sub_df = sub_df.sort_values('lambda_cate')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Axis 1: CATE
    color1 = 'tab:blue'
    ax1.set_xlabel('Lambda CATE Weight ($\lambda_{cate}$)', fontsize=12)
    ax1.set_ylabel('Delta CATE', color=color1, fontweight='bold', fontsize=12)
    ax1.errorbar(sub_df['lambda_cate'], sub_df['CATE'], yerr=sub_df['CATE_Std'], 
                 fmt='-o', color=color1, capsize=5, label='Delta CATE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Annotate CATE
    for i in range(len(sub_df)):
        x = sub_df['lambda_cate'].iloc[i]
        y = sub_df['CATE'].iloc[i]
        # Shift text slightly up
        ax1.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", 
                     ha='center',  fontsize=9, color=color1, weight='bold')
    
    # Axis 2: AUC
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('AUC (Model Performance)', color=color2, fontweight='bold', fontsize=12)
    ax2.errorbar(sub_df['lambda_cate'], sub_df['AUC'], yerr=sub_df['AUC_Std'],
                 fmt='-s', color=color2, capsize=5, label='AUC')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Annotate AUC
    for i in range(len(sub_df)):
        x = sub_df['lambda_cate'].iloc[i]
        y = sub_df['AUC'].iloc[i]
        # Shift text slightly down
        ax2.annotate(f"{y:.3f}", (x, y), xytext=(0, -12), textcoords="offset points", 
                     ha='center', fontsize=9, color=color2, weight='bold')

    plt.title('Impact of CATE Weight on Prediction Stability', fontsize=14, pad=20)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig("reports/figures/plot_cate_sensitivity.png", dpi=300)
    print("Saved plot_cate_sensitivity.png")

def plot_hsic_sensitivity():
    """Plot Lambda HSIC vs Metrics with annotations"""
    sub_df = df[df['type'] == 'hsic'].copy()
    if sub_df.empty:
        return
    sub_df = sub_df.sort_values('lambda_hsic')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'tab:green'
    
    # Use symlog to handle 0 values properly
    ax1.set_xscale('symlog', linthresh=0.01)
    ax1.set_xlabel('Lambda HSIC ($\lambda_{hsic}$)', fontsize=12)
    ax1.set_ylabel('Delta CATE', color=color1, fontweight='bold', fontsize=12)
    
    x_vals = sub_df['lambda_hsic']

    ax1.errorbar(x_vals, sub_df['CATE'], yerr=sub_df['CATE_Std'].fillna(0),
                 fmt='-o', color=color1, capsize=5, label='Delta CATE', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Manually set xticks to ensure 0 is shown properly
    ax1.set_xticks([0, 0.01, 0.1, 1, 10])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Annotate CATE
    for i in range(len(sub_df)):
        x = x_vals.iloc[i]
        y = sub_df['CATE'].iloc[i]
        ax1.annotate(f"{y:.3f}", (x, y), xytext=(0, 5), textcoords="offset points", 
                     ha='center', fontsize=9, color=color1, weight='bold')

    ax2 = ax1.twinx()
    color2 = 'tab:purple'
    ax2.set_ylabel('AUC (Classification Performance)', color=color2, fontweight='bold', fontsize=12)
    ax2.errorbar(x_vals, sub_df['AUC'], yerr=sub_df['AUC_Std'].fillna(0),
                 fmt='-s', color=color2, capsize=5, label='AUC', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Annotate AUC
    for i in range(len(sub_df)):
        x = x_vals.iloc[i]
        y = sub_df['AUC'].iloc[i]
        ax2.annotate(f"{y:.3f}", (x, y), xytext=(0, -12), textcoords="offset points", 
                     ha='center', fontsize=9, color=color2, weight='bold')
    
    plt.title('Impact of Deconfounding Strength (HSIC) on Model Stability', fontsize=14, pad=20)
    
    # Mark SOTA
    ax1.axvline(0.1, color='red', linestyle='--', alpha=0.5)
    ax1.text(0.12, sub_df['CATE'].max(), 'SOTA (0.1)', color='red', rotation=0, verticalalignment='top')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig("reports/figures/plot_hsic_sensitivity.png", dpi=300)
    print("Saved plot_hsic_sensitivity.png")

def plot_arch_heatmap():
    """Heatmap for Architecture Search (Combined CATE, PEHE, and AUC)"""
    sub_df = df[df['type'] == 'arch'].copy()
    if sub_df.empty:
        return

    # Pivot Data
    pivot_cate = sub_df.pivot(index='d_hidden', columns='num_layers', values='CATE')
    pivot_pehe = sub_df.pivot(index='d_hidden', columns='num_layers', values='PEHE')
    pivot_auc = sub_df.pivot(index='d_hidden', columns='num_layers', values='AUC')
    
    # 3-Panel Heatmap: CATE (Effect), PEHE (Error), AUC (Performance)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 8))
    
    # Common function to add SOTA box
    def add_sota_box(ax, pivot_table, color='gold'):
        # Updated SOTA: 128 units, 3 layers (Based on Seed 46 Champion & Occam's Razor)
        if 128.0 in pivot_table.index and 3.0 in pivot_table.columns:
            r_idx = list(pivot_table.index).index(128.0)
            # Find column index for num_layers=3.0
            c_idx = list(pivot_table.columns).index(3.0)
            rect = patches.Rectangle((c_idx, r_idx), 1, 1, fill=False, edgecolor=color, lw=4, clip_on=False)
            ax.add_patch(rect)
    
    # 1. CATE Heatmap
    sns.heatmap(pivot_cate, annot=True, fmt=".3f", cmap="RdBu_r", center=0, 
                cbar_kws={'label': 'Delta CATE'}, ax=ax1, annot_kws={"size": 10, "weight": "bold"})
    ax1.set_title("Effect Size (Delta CATE)\n(Higher is Stronger)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Hidden Dimension ({hidden}$)", fontsize=12)
    ax1.set_xlabel("Number of Layers ($)", fontsize=12)
    add_sota_box(ax1, pivot_cate, 'gold')

    # 2. PEHE Heatmap (New!)
    # Use a colormap where Darker/Redder means Higher Error (Bad), Lighter means Low Error (Good)
    sns.heatmap(pivot_pehe, annot=True, fmt=".3f", cmap="Reds", 
                cbar_kws={'label': 'PEHE (Loss)'}, ax=ax2, annot_kws={"size": 10, "weight": "bold"})
    ax2.set_title("Causal Error (PEHE)\n(Lower is Better)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("") 
    ax2.set_xlabel("Number of Layers ($)", fontsize=12)
    add_sota_box(ax2, pivot_pehe, 'blue') # Blue box for contrast on Red map

    # 3. AUC Heatmap
    sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="YlGnBu", 
                cbar_kws={'label': 'AUC (Prediction)'}, ax=ax3, annot_kws={"size": 10, "weight": "bold"})
    ax3.set_title("Prediction Performance (AUC)\n(Higher is Better)", fontsize=14, fontweight='bold')
    ax3.set_ylabel("") 
    ax3.set_xlabel("Number of Layers ($)", fontsize=12)
    add_sota_box(ax3, pivot_auc, 'red')
    
    plt.suptitle("Architecture Trilemma: Causal Effect (Left) vs. Estimation Error (Middle) vs. Predictive Power (Right)", fontsize=18, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/plot_arch_heatmap.png", dpi=300, bbox_inches='tight')
    print("Saved plot_arch_heatmap.png (3-Panel)")

def plot_metric_correlations():
    """
    Plot correlation matrix of all metrics to justify selection.
    Answers: Why not plot F1 vs CATE? Because F1 and AUC are highly correlated.
    """
    # Define metric list (Sensitivity often present but casing might vary, here it's Sensitivity)
    possible_metrics = ['AUC', 'ACC', 'F1', 'Sensitivity', 'CATE', 'PEHE']
    valid_metrics = [m for m in possible_metrics if m in df.columns]
    
    if len(valid_metrics) < 2:
        return

    # Calculate correlation
    corr = df[valid_metrics].corr()
    
    # Setup plot
    plt.figure(figsize=(10, 8))
    # Generate mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Draw heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Metric Correlation Matrix\n(Justifies use of Representative Metrics)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("reports/figures/plot_metric_correlations.png", dpi=300)
    print("Saved plot_metric_correlations.png")

if __name__ == "__main__":
    plot_cate_tradeoff()
    plot_cate_sensitivity()
    plot_hsic_sensitivity()
    plot_arch_heatmap()
    plot_metric_correlations()

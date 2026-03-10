import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_sensitivity(csv_file, param_name, display_name, file_prefix, chosen_val):
    df = pd.read_csv(csv_file)
    x = df[param_name]
    
    # Sort just in case
    df = df.sort_values(by=param_name)
    x = df[param_name]
    
    # --- Tradeoff Plot (Delta CATE & PEHE) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8), dpi=300)
    
    # Top: Delta CATE
    ax1.errorbar(x, df['Delta CATE'], yerr=df['Delta_CATE_Std'], fmt='o--', capsize=5, label='Delta CATE', color='darkorange')
    ax1.set_ylabel('Delta CATE')
    ax1.set_title(f'Sensitivity Analysis: {display_name} Trade-off')
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Highlight chosen default value
    chosen_row = df[df[param_name] == chosen_val]
    if not chosen_row.empty:
        ax1.scatter(chosen_val, chosen_row['Delta CATE'], color='red', s=150, zorder=5, edgecolors='black', label='Chosen Default')
        
    for i, row in df.iterrows():
        ax1.annotate(f"{row['Delta CATE']:.3f}", (row[param_name], row['Delta CATE']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Bottom: PEHE
    ax2.errorbar(x, df['PEHE'], yerr=df['PEHE_Std'], fmt='s--', capsize=5, label='PEHE', color='teal')
    ax2.set_xlabel(display_name)
    ax2.set_ylabel('PEHE (Lower is Better)')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    if not chosen_row.empty:
        ax2.scatter(chosen_val, chosen_row['PEHE'], color='red', s=150, zorder=5, edgecolors='black', label='Chosen Default')
        
    for i, row in df.iterrows():
        ax2.annotate(f"{row['PEHE']:.3f}", (row[param_name], row['PEHE']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Use log scale for X if values range greatly (e.g., 0.01 to 10.0 for hsic)
    if param_name == 'lambda_hsic' and max(x) >= 10:
        ax2.set_xscale('symlog', linthresh=0.01)
        ax2.set_xticks([0.0, 0.01, 0.1, 1.0, 10.0])
        ax2.set_xticklabels(['0', '0.01', '0.1', '1.0', '10.0'])

    plt.tight_layout()
    plt.savefig(f'{file_prefix}_tradeoff.png')
    plt.savefig(f'{file_prefix}_tradeoff.pdf')
    plt.savefig(f'{file_prefix}_tradeoff.svg')
    plt.close()

    # --- AUC Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8), dpi=300)
    
    # Top: Delta CATE
    ax1.errorbar(x, df['Delta CATE'], yerr=df['Delta_CATE_Std'], fmt='o--', capsize=5, label='Delta CATE', color='darkorange')
    ax1.set_ylabel('Delta CATE')
    ax1.set_title(f'Hyperparameter Trade-off: {display_name} vs AUC')
    ax1.grid(True, linestyle=':', alpha=0.7)
    if not chosen_row.empty:
        ax1.scatter(chosen_val, chosen_row['Delta CATE'], color='red', s=150, zorder=5, edgecolors='black')
        
    for i, row in df.iterrows():
        ax1.annotate(f"{row['Delta CATE']:.3f}", (row[param_name], row['Delta CATE']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Bottom: AUC
    ax2.errorbar(x, df['AUC'], yerr=df['AUC_Std'], fmt='^--', capsize=5, label='AUC', color='purple')
    ax2.set_xlabel(display_name)
    ax2.set_ylabel('AUC (Higher is Better)')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    if not chosen_row.empty:
        ax2.scatter(chosen_val, chosen_row['AUC'], color='red', s=150, zorder=5, edgecolors='black', label='Chosen Default')
        
    for i, row in df.iterrows():
        ax2.annotate(f"{row['AUC']:.3f}", (row[param_name], row['AUC']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    if param_name == 'lambda_hsic' and max(x) >= 10:
        ax2.set_xscale('symlog', linthresh=0.01)
        ax2.set_xticks([0.0, 0.01, 0.1, 1.0, 10.0])
        ax2.set_xticklabels(['0', '0.01', '0.1', '1.0', '10.0'])

    plt.tight_layout()
    plt.savefig(f'{file_prefix}_auc.png')
    plt.savefig(f'{file_prefix}_auc.pdf')
    plt.savefig(f'{file_prefix}_auc.svg')
    plt.close()

if __name__ == '__main__':
    # Current chosen defaults (conservative/balanced choices, not necessarily statistical optima)
    # lambda_hsic = 0.1: moderate deconfounding
    # lambda_cate = 2.0: balanced trade-off between effect size and prediction accuracy
    plot_sensitivity('lambda_hsic_sweep_results.csv', 'lambda_hsic', 'HSIC Weight (λ_hsic)', 'fig_lambda_hsic', 0.1)
    plot_sensitivity('lambda_cate_sweep_results.csv', 'lambda_cate', 'CATE Weight (λ_cate)', 'fig_lambda_cate', 2.0)
    print("Done plotting.")

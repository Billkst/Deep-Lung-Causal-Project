
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_sensitivity(csv_path="results/parameter_sensitivity_results_final.csv", output_path="results/parameter_sensitivity_summary.png"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Setup plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Architecture
    df_arch = df[df['type'] == 'arch'].copy()
    if not df_arch.empty:
        # Create a combined category for x-axis
        df_arch['config'] = df_arch['d_hidden'].astype(str) + " / " + df_arch['num_layers'].astype(str) + "L"
        # Sort reasonably? Maybe by d_hidden then num_layers
        df_arch = df_arch.sort_values(by=['d_hidden', 'num_layers'])
        
        sns.lineplot(data=df_arch, x='config', y='AUC', marker='o', ax=axes[0], label='AUC', color='b')
        ax2 = axes[0].twinx()
        sns.lineplot(data=df_arch, x='config', y='CATE', marker='s', ax=ax2, label='Delta CATE', color='r')
        
        axes[0].set_title("Architecture Sensitivity (Hidden/Layers)")
        axes[0].set_xlabel("Hidden Dim / Layers")
        axes[0].set_xticklabels(df_arch['config'], rotation=45)
        axes[0].set_ylabel("AUC")
        ax2.set_ylabel("Delta CATE (Lower is Better)")
        
        # Legend
        lines, labels = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[0].legend(lines + lines2, labels + labels2, loc='upper left')

    # 2. HSIC
    df_hsic = df[df['type'] == 'hsic'].copy()
    if not df_hsic.empty:
        sns.lineplot(data=df_hsic, x='lambda_hsic', y='AUC', marker='o', ax=axes[1], label='AUC', color='b')
        ax2 = axes[1].twinx()
        sns.lineplot(data=df_hsic, x='lambda_hsic', y='CATE', marker='s', ax=ax2, label='Delta CATE', color='r')
        
        axes[1].set_title("Independence Weight (HSIC) Sensitivity")
        axes[1].set_xlabel("Lambda HSIC")
        axes[1].set_xscale('log') # Log scale often better for lambda
        axes[1].set_ylabel("AUC")
        ax2.set_ylabel("Delta CATE")

    # 3. CATE
    df_cate = df[df['type'] == 'cate'].copy()
    if not df_cate.empty:
        sns.lineplot(data=df_cate, x='lambda_cate', y='AUC', marker='o', ax=axes[2], label='AUC', color='b')
        ax2 = axes[2].twinx()
        sns.lineplot(data=df_cate, x='lambda_cate', y='CATE', marker='s', ax=ax2, label='Delta CATE', color='r')
        
        axes[2].set_title("Treatment Weight (CATE) Sensitivity")
        axes[2].set_xlabel("Lambda CATE")
        axes[2].set_ylabel("AUC")
        ax2.set_ylabel("Delta CATE")
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_sensitivity()

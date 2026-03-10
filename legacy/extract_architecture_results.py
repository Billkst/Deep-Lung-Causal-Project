import pandas as pd
df = pd.read_csv('results/parameter_sensitivity_results_final.csv')
arch_df = df[df['type'] == 'arch'].copy()
# Remove unnecessary columns if needed
arch_df = arch_df[['d_hidden', 'num_layers', 'AUC', 'AUC_Std', 'PEHE', 'PEHE_Std', 'CATE', 'CATE_Std']]
arch_df.rename(columns={'CATE': 'Delta CATE', 'CATE_Std': 'Delta_CATE_Std'}, inplace=True)
arch_df.to_csv('architecture_results.csv', index=False)
print("Extracted architecture_results.csv")

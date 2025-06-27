import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

latent_dims = [2, 4, 8, 16, 32, 64, 100]
dropout_values = [0.0, 0.1, 0.5]

df_list = []

folder_path = "csv_files"

# Loop over every (latent_dim, dropout) combination
for ld in latent_dims:
    for dr in dropout_values:
        # Filename pattern: vae_per_epoch_metrics_ld{ld}_dr{dr}.csv
        csv_file = os.path.join(folder_path, f"vae_per_epoch_metrics_ld{ld}_dr{dr}.csv")
        try:
            tmp = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found. Skipping.")
            continue
        
        # Add columns for ld and dr
        tmp["latent_dim"] = ld
        tmp["dropout"] = dr
        
        df_list.append(tmp)

# Combine all into one big DataFrame
if df_list:
    df_all = pd.concat(df_list, ignore_index=True)
else:
    raise RuntimeError("No CSV files found or all were missing!")

# These are the metrics you want to plot
metrics = [
    "train_rmse",
    "val_rmse",
    "train_kl",
    "val_kl",
    "train_recon_loss",
    "val_recon_loss",
    "train_loss",
    "val_loss"
]

# For each metric, create separate plots by dropout value
for metric in metrics:
    # Create one plot per dropout value
    for dr in dropout_values:
        plt.figure(figsize=(16,10))
        
        # For each latent dimension, plot that metric vs. epoch for this dropout
        for ld in latent_dims:
            # Filter rows for that (ld, dr)
            subset = df_all[(df_all["latent_dim"] == ld) & (df_all["dropout"] == dr)]
            if not subset.empty:
                # Plot one line
                plt.plot(
                    subset["epoch"], 
                    subset[metric],
                    label=f"Latent Dim = {ld}"
                )
        
        plt.title(f"{metric} vs. epoch (Dropout = {dr})")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f"plots/{metric}_dropout_{dr}.png")
        plt.close()
        print(f"Created plot for {metric} with dropout={dr}")

# Also include the original heatmap code for completeness
# Create an empty DataFrame with 'dropout_values' as the index and 'latent_dims' as columns
df_heatmap = pd.DataFrame(index=dropout_values, columns=latent_dims)

for ld in latent_dims:
    for dr in dropout_values:
        # The filename for each combination
        summary_file = os.path.join(folder_path, f"vae_summary_ld{ld}_dr{dr}.csv")
        
        try:
            # Each summary CSV presumably has 1 row with the column 'best_val_rmse'
            tmp = pd.read_csv(summary_file)
            # Extract the single best_val_rmse value
            best_rmse = tmp["best_val_rmse"].iloc[0]
            
            # Place it in the DataFrame at row=dr, col=ld
            df_heatmap.loc[dr, ld] = best_rmse
            
        except FileNotFoundError:
            print(f"Warning: {summary_file} not found. Setting NaN.")
            df_heatmap.loc[dr, ld] = float('nan')

# Convert to float (in case it's still string)
df_heatmap = df_heatmap.astype(float)

plt.figure(figsize=(16, 10))
sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="viridis")
plt.xlabel("Latent Dimensions")
plt.ylabel("Dropout")
plt.title("Best Validation RMSE Heatmap")
plt.savefig("plots/best_val_rmse_heatmap.png")
plt.close() 
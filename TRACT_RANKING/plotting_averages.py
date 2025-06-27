import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the CSV file
csv_path = "combo_ld64_drV0.0_drA0.0_drS0.0_wr20.0_wkl0.0_wa5.0_ws2.5_metrics.csv"
# Output directory for plots
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots'))
os.makedirs(plots_dir, exist_ok=True)

# Load the CSV
csv_full_path = os.path.join(os.path.dirname(__file__), csv_path)
df = pd.read_csv(csv_full_path)

# Metric pairs: (train, val, plot title, y-label)
metric_pairs = [
    ('loss', 'Loss'),
    ('recon_loss', 'Reconstruction Loss'),
    ('kl_loss', 'KL Loss'),
    ('age_loss', 'Age Loss'),
    ('site_loss', 'Site Loss'),
    ('age_mae', 'Age MAE'),
    ('site_acc', 'Site Accuracy'),
]

for base_metric, ylabel in metric_pairs:
    train_col = f'train_{base_metric}'
    val_col = f'val_{base_metric}'
    if train_col in df.columns and val_col in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(df['epoch'], df[train_col], label=f'Train {ylabel}')
        plt.plot(df['epoch'], df[val_col], label=f'Validation {ylabel}')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'Epoch vs. {ylabel}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"{base_metric}_train_val_vs_epoch.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

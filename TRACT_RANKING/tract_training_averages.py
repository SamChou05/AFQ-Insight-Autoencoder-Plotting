#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt

# Base directory containing all tract folders
base_dir = "/Users/samchou/AFQ-Insight-Autoencoder-Plotting/TRACT_RANKING"

# Function to get the numeric tract ID from a directory path
def get_tract_id(directory):
    try:
        # Extract the numeric part from "tract_X" or "TRACT_X" pattern
        match = re.search(r'tract_(\d+)|TRACT_(\d+)', directory, re.IGNORECASE)
        if match:
            # Return the first non-None group (the tract number)
            return next(g for g in match.groups() if g is not None)
    except Exception as e:
        print(f"Error extracting tract ID from {directory}: {e}")
    return None

# Find all tract directories
tract_dirs = []
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path) and (item.lower().startswith('tract_') or item.startswith('TRACT_')):
        tract_dirs.append(item_path)

# Sort tract directories by their numeric ID
tract_dirs = sorted(tract_dirs, key=lambda x: int(get_tract_id(x)) if get_tract_id(x) else float('inf'))

# Initialize dictionaries to store all training histories
age_predictor_histories = []
site_predictor_histories = []
vae_histories = []
combined_model_histories = []

# Process each tract directory
for tract_dir in tract_dirs:
    tract_id = get_tract_id(tract_dir)
    if not tract_id:
        continue
    
    print(f"Processing tract {tract_id}...")
    
    # Process Age Predictor
    age_file = os.path.join(tract_dir, "age_predictor_training_history.csv")
    if os.path.exists(age_file):
        try:
            df = pd.read_csv(age_file)
            if not df.empty:
                df['tract_id'] = tract_id
                age_predictor_histories.append(df)
        except Exception as e:
            print(f"Error processing age predictor for tract {tract_id}: {e}")
    
    # Process Site Predictor
    site_file = os.path.join(tract_dir, "site_predictor_training_history.csv")
    if os.path.exists(site_file):
        try:
            df = pd.read_csv(site_file)
            if not df.empty:
                df['tract_id'] = tract_id
                site_predictor_histories.append(df)
        except Exception as e:
            print(f"Error processing site predictor for tract {tract_id}: {e}")
    
    # Process VAE
    vae_file = os.path.join(tract_dir, "vae_training_history.csv")
    if os.path.exists(vae_file):
        try:
            df = pd.read_csv(vae_file)
            if not df.empty:
                df['tract_id'] = tract_id
                vae_histories.append(df)
        except Exception as e:
            print(f"Error processing VAE for tract {tract_id}: {e}")
    
    # Process Combined model
    combined_file = os.path.join(tract_dir, "combined_model_training_history.csv")
    if os.path.exists(combined_file):
        try:
            df = pd.read_csv(combined_file)
            if not df.empty:
                df['tract_id'] = tract_id
                combined_model_histories.append(df)
        except Exception as e:
            print(f"Error processing combined model for tract {tract_id}: {e}")

# Output directory for results
output_dir = "/Users/samchou/AFQ-Insight-Autoencoder-Plotting/training_averages"
os.makedirs(output_dir, exist_ok=True)

# Function to compute average metrics by epoch
def compute_epoch_averages(history_list, model_type):
    if not history_list:
        print(f"No data found for {model_type}")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(history_list, ignore_index=True)
    
    # Group by epoch and calculate mean for each metric
    avg_by_epoch = combined_df.groupby('epoch').mean(numeric_only=True)
    
    # Add count of tracts for each epoch
    tract_count = combined_df.groupby('epoch')['tract_id'].nunique()
    avg_by_epoch['num_tracts'] = tract_count
    
    # Save to CSV
    avg_by_epoch.to_csv(os.path.join(output_dir, f"{model_type}_epoch_averages.csv"))
    
    return avg_by_epoch

# Compute and save averages for each model type
print("\nComputing averages by epoch...")
age_avg = compute_epoch_averages(age_predictor_histories, "age_predictor")
site_avg = compute_epoch_averages(site_predictor_histories, "site_predictor")
vae_avg = compute_epoch_averages(vae_histories, "vae")
combined_avg = compute_epoch_averages(combined_model_histories, "combined_model")

# Generate some basic plots of the average metrics over time
def plot_metric(df, metric, title, ylabel, output_file):
    if df is None or metric not in df.columns:
        print(f"Cannot plot {metric}, data not available")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[metric])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# Create a plots directory
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Plot age predictor metrics
if age_avg is not None:
    # MAE
    if 'train_mae' in age_avg.columns and 'val_mae' in age_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(age_avg.index, age_avg['train_mae'], label='Train MAE')
        plt.plot(age_avg.index, age_avg['val_mae'], label='Validation MAE')
        plt.title('Average Age Predictor MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'age_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(age_avg.index, age_avg['train_mae'], label='Train MAE')
        plt.title('Average Age Predictor Train MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'age_train_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(age_avg.index, age_avg['val_mae'], label='Validation MAE', color='orange')
        plt.title('Average Age Predictor Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'age_val_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # R2
    if 'train_r2' in age_avg.columns and 'val_r2' in age_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(age_avg.index, age_avg['train_r2'], label='Train R²')
        plt.plot(age_avg.index, age_avg['val_r2'], label='Validation R²')
        plt.title('Average Age Predictor R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'age_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(age_avg.index, age_avg['train_r2'], label='Train R²')
        plt.title('Average Age Predictor Train R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'age_train_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(age_avg.index, age_avg['val_r2'], label='Validation R²', color='orange')
        plt.title('Average Age Predictor Validation R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'age_val_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Plot site predictor metrics
if site_avg is not None:
    # Accuracy
    if 'train_acc' in site_avg.columns and 'val_acc' in site_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(site_avg.index, site_avg['train_acc'], label='Train Accuracy')
        plt.plot(site_avg.index, site_avg['val_acc'], label='Validation Accuracy')
        plt.title('Average Site Predictor Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'site_acc.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(site_avg.index, site_avg['train_acc'], label='Train Accuracy')
        plt.title('Average Site Predictor Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'site_train_acc.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(site_avg.index, site_avg['val_acc'], label='Validation Accuracy', color='orange')
        plt.title('Average Site Predictor Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'site_val_acc.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # Loss
    if 'train_loss' in site_avg.columns and 'val_loss' in site_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(site_avg.index, site_avg['train_loss'], label='Train Loss')
        plt.plot(site_avg.index, site_avg['val_loss'], label='Validation Loss')
        plt.title('Average Site Predictor Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'site_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(site_avg.index, site_avg['train_loss'], label='Train Loss')
        plt.title('Average Site Predictor Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'site_train_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(site_avg.index, site_avg['val_loss'], label='Validation Loss', color='orange')
        plt.title('Average Site Predictor Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'site_val_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Plot VAE metrics
if vae_avg is not None:
    # Loss
    if 'train_loss' in vae_avg.columns and 'val_loss' in vae_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(vae_avg.index, vae_avg['train_loss'], label='Train Loss')
        plt.plot(vae_avg.index, vae_avg['val_loss'], label='Validation Loss')
        plt.title('Average VAE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'vae_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(vae_avg.index, vae_avg['train_loss'], label='Train Loss')
        plt.title('Average VAE Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'vae_train_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(vae_avg.index, vae_avg['val_loss'], label='Validation Loss', color='orange')
        plt.title('Average VAE Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'vae_val_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # Recon Loss
    if 'train_recon_loss' in vae_avg.columns and 'val_recon_loss' in vae_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(vae_avg.index, vae_avg['train_recon_loss'], label='Train Recon Loss')
        plt.plot(vae_avg.index, vae_avg['val_recon_loss'], label='Validation Recon Loss')
        plt.title('Average VAE Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'vae_recon_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(vae_avg.index, vae_avg['train_recon_loss'], label='Train Recon Loss')
        plt.title('Average VAE Train Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'vae_train_recon_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(vae_avg.index, vae_avg['val_recon_loss'], label='Validation Recon Loss', color='orange')
        plt.title('Average VAE Validation Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'vae_val_recon_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if 'beta' in vae_avg.columns:
        plot_metric(vae_avg, 'beta', 'Average VAE Beta Value', 'Beta', 
                    os.path.join(plots_dir, 'vae_beta.png'))

# Plot combined model metrics
if combined_avg is not None:
    # Loss
    if 'train_loss' in combined_avg.columns and 'val_loss' in combined_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['train_loss'], label='Train Loss')
        plt.plot(combined_avg.index, combined_avg['val_loss'], label='Validation Loss')
        plt.title('Average Combined Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['train_loss'], label='Train Loss')
        plt.title('Average Combined Model Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_train_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['val_loss'], label='Validation Loss', color='orange')
        plt.title('Average Combined Model Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_val_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # Age MAE
    if 'train_age_mae' in combined_avg.columns and 'val_age_mae' in combined_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['train_age_mae'], label='Train Age MAE')
        plt.plot(combined_avg.index, combined_avg['val_age_mae'], label='Validation Age MAE')
        plt.title('Average Combined Model Age MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_age_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['train_age_mae'], label='Train Age MAE')
        plt.title('Average Combined Model Train Age MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_train_age_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['val_age_mae'], label='Validation Age MAE', color='orange')
        plt.title('Average Combined Model Validation Age MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_val_age_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # Site Accuracy
    if 'train_site_acc' in combined_avg.columns and 'val_site_acc' in combined_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['train_site_acc'], label='Train Site Accuracy')
        plt.plot(combined_avg.index, combined_avg['val_site_acc'], label='Validation Site Accuracy')
        plt.title('Average Combined Model Site Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_site_acc.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['train_site_acc'], label='Train Site Accuracy')
        plt.title('Average Combined Model Train Site Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_train_site_acc.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['val_site_acc'], label='Validation Site Accuracy', color='orange')
        plt.title('Average Combined Model Validation Site Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_val_site_acc.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # Age R2
    if 'train_age_r2' in combined_avg.columns and 'val_age_r2' in combined_avg.columns:
        # Combined
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['train_age_r2'], label='Train Age R²')
        plt.plot(combined_avg.index, combined_avg['val_age_r2'], label='Validation Age R²')
        plt.title('Average Combined Model Age R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_age_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Separate
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['train_age_r2'], label='Train Age R²')
        plt.title('Average Combined Model Train Age R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_train_age_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(combined_avg.index, combined_avg['val_age_r2'], label='Validation Age R²', color='orange')
        plt.title('Average Combined Model Validation Age R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'combined_val_age_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if 'current_beta' in combined_avg.columns:
        plot_metric(combined_avg, 'current_beta', 'Average Combined Model Beta Value', 'Beta', 
                    os.path.join(plots_dir, 'combined_beta.png'))
    if 'current_grl_alpha' in combined_avg.columns:
        plot_metric(combined_avg, 'current_grl_alpha', 'Average Combined Model GRL Alpha', 'Alpha', 
                    os.path.join(plots_dir, 'combined_grl_alpha.png'))

print(f"\nAnalysis complete. Results saved to {output_dir}")
print(f"Plots generated in {plots_dir}")

# Also create a simple summary table showing how training progresses across epochs
def create_epoch_summary_table(df, model_type, key_metrics):
    if df is None:
        return
    
    # Select specific epochs to show progression (start, 25%, 50%, 75%, end)
    epoch_count = len(df)
    if epoch_count <= 5:
        selected_epochs = df.index.tolist()
    else:
        selected_epochs = [
            1,  # First epoch
            max(2, int(epoch_count * 0.25)),  # ~25%
            int(epoch_count * 0.5),           # ~50%
            int(epoch_count * 0.75),          # ~75%
            epoch_count                       # Last epoch
        ]
    
    # Create summary dataframe
    summary = df.loc[selected_epochs, key_metrics].copy()
    
    # Add epoch percentage column
    summary['epoch_percentage'] = [f"{int(e/epoch_count*100)}%" for e in selected_epochs]
    
    # Reorder columns to put epoch_percentage first
    cols = ['epoch_percentage'] + key_metrics
    summary = summary[cols]
    
    # Save to CSV
    summary.to_csv(os.path.join(output_dir, f"{model_type}_epoch_summary.csv"))
    return summary

# Create summary tables
if age_avg is not None:
    create_epoch_summary_table(age_avg, "age_predictor", 
                              ['train_mae', 'val_mae', 'train_r2', 'val_r2', 'num_tracts'])
                              
if site_avg is not None:
    create_epoch_summary_table(site_avg, "site_predictor", 
                              ['train_acc', 'val_acc', 'train_loss', 'val_loss', 'num_tracts'])
                              
if vae_avg is not None:
    vae_metrics = ['train_loss', 'val_loss', 'train_recon_loss', 'val_recon_loss', 'num_tracts']
    if 'beta' in vae_avg.columns:
        vae_metrics.append('beta')
    create_epoch_summary_table(vae_avg, "vae", vae_metrics)
                              
if combined_avg is not None:
    combined_metrics = ['train_loss', 'val_loss', 'train_age_mae', 'val_age_mae', 'num_tracts']
    if 'train_site_acc' in combined_avg.columns:
        combined_metrics.append('train_site_acc')
    if 'val_site_acc' in combined_avg.columns:
        combined_metrics.append('val_site_acc')
    if 'current_beta' in combined_avg.columns:
        combined_metrics.append('current_beta')
    if 'current_grl_alpha' in combined_avg.columns:
        combined_metrics.append('current_grl_alpha')
    create_epoch_summary_table(combined_avg, "combined_model", combined_metrics)

print("Summary tables created.")
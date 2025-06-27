#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import glob
import re
from pathlib import Path

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

# Dictionary to store best metrics for each model and tract
results = {}

# Find all tract directories
tract_dirs = []
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path) and (item.lower().startswith('tract_') or item.startswith('TRACT_')):
        tract_dirs.append(item_path)

# Sort tract directories by their numeric ID
tract_dirs = sorted(tract_dirs, key=lambda x: int(get_tract_id(x)) if get_tract_id(x) else float('inf'))

for tract_dir in tract_dirs:
    tract_id = get_tract_id(tract_dir)
    if not tract_id:
        continue
    
    print(f"Processing tract {tract_id}...")
    results[tract_id] = {
        "age_predictor": {},
        "site_predictor": {},
        "vae": {},
        "combined": {}
    }
    
    # Process Age Predictor
    age_file = os.path.join(tract_dir, "age_predictor_training_history.csv")
    if os.path.exists(age_file):
        try:
            df = pd.read_csv(age_file)
            if not df.empty:
                results[tract_id]["age_predictor"] = {
                    "best_train_mae": df["train_mae"].min(),
                    "best_val_mae": df["val_mae"].min(),
                    "best_train_r2": df["train_r2"].max(),
                    "best_val_r2": df["val_r2"].max(),
                    "epochs": len(df),
                    "epoch_best_val_mae": df["val_mae"].idxmin() + 1,
                    "epoch_best_val_r2": df["val_r2"].idxmax() + 1
                }
        except Exception as e:
            print(f"Error processing age predictor for tract {tract_id}: {e}")
    
    # Process Site Predictor
    site_file = os.path.join(tract_dir, "site_predictor_training_history.csv")
    if os.path.exists(site_file):
        try:
            df = pd.read_csv(site_file)
            if not df.empty:
                results[tract_id]["site_predictor"] = {
                    "best_train_acc": df["train_acc"].max(),
                    "best_val_acc": df["val_acc"].max(),
                    "best_train_loss": df["train_loss"].min(),
                    "best_val_loss": df["val_loss"].min(),
                    "epochs": len(df),
                    "epoch_best_val_acc": df["val_acc"].idxmax() + 1,
                    "epoch_best_val_loss": df["val_loss"].idxmin() + 1
                }
        except Exception as e:
            print(f"Error processing site predictor for tract {tract_id}: {e}")
    
    # Process VAE
    vae_file = os.path.join(tract_dir, "vae_training_history.csv")
    if os.path.exists(vae_file):
        try:
            df = pd.read_csv(vae_file)
            if not df.empty:
                results[tract_id]["vae"] = {
                    "best_train_loss": df["train_loss"].min(),
                    "best_val_loss": df["val_loss"].min(),
                    "best_train_recon_loss": df["train_recon_loss"].min(),
                    "best_val_recon_loss": df["val_recon_loss"].min(),
                    "final_beta": df["beta"].iloc[-1],
                    "epochs": len(df),
                    "epoch_best_val_loss": df["val_loss"].idxmin() + 1,
                    "epoch_best_val_recon_loss": df["val_recon_loss"].idxmin() + 1
                }
        except Exception as e:
            print(f"Error processing VAE for tract {tract_id}: {e}")
    
    # Process Combined model
    combined_file = os.path.join(tract_dir, "combined_model_training_history.csv")
    if os.path.exists(combined_file):
        try:
            df = pd.read_csv(combined_file)
            if not df.empty:
                results[tract_id]["combined"] = {
                    "best_train_loss": df["train_loss"].min(),
                    "best_val_loss": df["val_loss"].min(),
                    "best_train_age_mae": df["train_age_mae"].min(),
                    "best_val_age_mae": df["val_age_mae"].min(),
                    "best_train_site_acc": df["train_site_acc"].max() if "train_site_acc" in df.columns else None,
                    "best_val_site_acc": df["val_site_acc"].max() if "val_site_acc" in df.columns else None,
                    "best_train_age_r2": df["train_age_r2"].max() if "train_age_r2" in df.columns else None,
                    "best_val_age_r2": df["val_age_r2"].max() if "val_age_r2" in df.columns else None,
                    "final_beta": df["current_beta"].iloc[-1] if "current_beta" in df.columns else None,
                    "final_grl_alpha": df["current_grl_alpha"].iloc[-1] if "current_grl_alpha" in df.columns else None,
                    "epochs": len(df),
                    "epoch_best_val_loss": df["val_loss"].idxmin() + 1,
                    "epoch_best_val_age_mae": df["val_age_mae"].idxmin() + 1,
                    "epoch_best_val_site_acc": df["val_site_acc"].idxmax() + 1 if "val_site_acc" in df.columns else None
                }
        except Exception as e:
            print(f"Error processing combined model for tract {tract_id}: {e}")

# Create summary DataFrames for each model type
age_predictor_summary = []
site_predictor_summary = []
vae_summary = []
combined_summary = []

for tract_id, tract_results in results.items():
    
    # Age predictor
    if tract_results["age_predictor"]:
        age_data = tract_results["age_predictor"]
        age_data["tract_id"] = tract_id
        age_predictor_summary.append(age_data)
    
    # Site predictor
    if tract_results["site_predictor"]:
        site_data = tract_results["site_predictor"]
        site_data["tract_id"] = tract_id
        site_predictor_summary.append(site_data)
    
    # VAE
    if tract_results["vae"]:
        vae_data = tract_results["vae"]
        vae_data["tract_id"] = tract_id
        vae_summary.append(vae_data)
    
    # Combined
    if tract_results["combined"]:
        combined_data = tract_results["combined"]
        combined_data["tract_id"] = tract_id
        combined_summary.append(combined_data)

# Convert lists to DataFrames
age_df = pd.DataFrame(age_predictor_summary)
site_df = pd.DataFrame(site_predictor_summary)
vae_df = pd.DataFrame(vae_summary)
combined_df = pd.DataFrame(combined_summary)

# Sort all DataFrames by tract_id for consistency
for df in [age_df, site_df, vae_df, combined_df]:
    if not df.empty:
        df["tract_id"] = df["tract_id"].astype(int)
        df.sort_values("tract_id", inplace=True)

# Save to CSV
output_dir = "/Users/samchou/AFQ-Insight-Autoencoder-Plotting/results_summary"
os.makedirs(output_dir, exist_ok=True)

if not age_df.empty:
    age_df.to_csv(os.path.join(output_dir, "age_predictor_summary.csv"), index=False)
if not site_df.empty:
    site_df.to_csv(os.path.join(output_dir, "site_predictor_summary.csv"), index=False)
if not vae_df.empty:
    vae_df.to_csv(os.path.join(output_dir, "vae_summary.csv"), index=False)
if not combined_df.empty:
    combined_df.to_csv(os.path.join(output_dir, "combined_model_summary.csv"), index=False)

# Create an overall summary with key metrics from each model
overall_summary = []

for tract_id, tract_results in results.items():
    summary_row = {"tract_id": tract_id}
    
    # Age predictor
    if tract_results["age_predictor"]:
        summary_row.update({
            "age_best_val_mae": tract_results["age_predictor"]["best_val_mae"],
            "age_best_val_r2": tract_results["age_predictor"]["best_val_r2"],
            "age_epochs": tract_results["age_predictor"]["epochs"]
        })
    
    # Site predictor
    if tract_results["site_predictor"]:
        summary_row.update({
            "site_best_val_acc": tract_results["site_predictor"]["best_val_acc"],
            "site_best_val_loss": tract_results["site_predictor"]["best_val_loss"],
            "site_epochs": tract_results["site_predictor"]["epochs"]
        })
    
    # VAE
    if tract_results["vae"]:
        summary_row.update({
            "vae_best_val_loss": tract_results["vae"]["best_val_loss"],
            "vae_best_val_recon_loss": tract_results["vae"]["best_val_recon_loss"],
            "vae_final_beta": tract_results["vae"]["final_beta"],
            "vae_epochs": tract_results["vae"]["epochs"]
        })
    
    # Combined
    if tract_results["combined"]:
        summary_row.update({
            "combined_best_val_age_mae": tract_results["combined"]["best_val_age_mae"],
            "combined_best_val_site_acc": tract_results["combined"]["best_val_site_acc"],
            "combined_best_val_loss": tract_results["combined"]["best_val_loss"],
            "combined_final_beta": tract_results["combined"]["final_beta"],
            "combined_final_grl_alpha": tract_results["combined"]["final_grl_alpha"],
            "combined_epochs": tract_results["combined"]["epochs"]
        })
    
    overall_summary.append(summary_row)

# Convert to DataFrame, sort by tract_id, and save
overall_df = pd.DataFrame(overall_summary)

if not overall_df.empty:
    overall_df["tract_id"] = overall_df["tract_id"].astype(int)
    overall_df.sort_values("tract_id", inplace=True)
    overall_df.to_csv(os.path.join(output_dir, "overall_model_summary.csv"), index=False)

print(f"\nAnalysis complete. Summary files saved to {output_dir}")

# Now print some high-level statistics about the results
print("\n=== OVERALL STATISTICS ===")

def print_stats(df, metric, metric_name):
    if not df.empty and metric in df.columns:
        print(f"\n{metric_name}:")
        print(f"  Mean: {df[metric].mean():.4f}")
        print(f"  Min: {df[metric].min():.4f} (Tract {df.loc[df[metric].idxmin(), 'tract_id']})")
        print(f"  Max: {df[metric].max():.4f} (Tract {df.loc[df[metric].idxmax(), 'tract_id']})")
        print(f"  Median: {df[metric].median():.4f}")

# Age predictor stats
if not age_df.empty:
    print("\nAGE PREDICTOR STATS:")
    print_stats(age_df, "best_val_mae", "Best Validation MAE")
    print_stats(age_df, "best_val_r2", "Best Validation R²")

# Site predictor stats
if not site_df.empty:
    print("\nSITE PREDICTOR STATS:")
    print_stats(site_df, "best_val_acc", "Best Validation Accuracy (%)")
    print_stats(site_df, "best_val_loss", "Best Validation Loss")

# VAE stats
if not vae_df.empty:
    print("\nVAE STATS:")
    print_stats(vae_df, "best_val_loss", "Best Validation Loss")
    print_stats(vae_df, "best_val_recon_loss", "Best Validation Reconstruction Loss")

# Combined model stats
if not combined_df.empty:
    print("\nCOMBINED MODEL STATS:")
    print_stats(combined_df, "best_val_age_mae", "Best Validation Age MAE")
    print_stats(combined_df, "best_val_site_acc", "Best Validation Site Accuracy (%)")
    print_stats(combined_df, "best_val_loss", "Best Validation Loss")

print("\nAnalysis complete!")
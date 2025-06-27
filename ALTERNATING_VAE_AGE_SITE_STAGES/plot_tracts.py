#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob

def load_tract_results(input_dir):
    """Load and combine all tract results from the given directory."""
    # Load tract names if available
    tract_names_file = os.path.join(input_dir, "tract_names.json")
    tract_names = {}
    if os.path.exists(tract_names_file):
        with open(tract_names_file, 'r') as f:
            tract_list = json.load(f)
            tract_names = {i: name for i, name in enumerate(tract_list)}
    
    # Try loading the combined file first if it exists
    combined_file = os.path.join(input_dir, "combined_tract_ranking.csv")
    if os.path.exists(combined_file):
        print(f"Loading existing combined results from {combined_file}")
        return pd.read_csv(combined_file)
    
    # Otherwise, collect all results manually
    print("No combined file found, collecting tract results individually...")
    tract_dirs = glob.glob(os.path.join(input_dir, "tract_*"))
    all_results = []
    
    for tract_dir in tract_dirs:
        if not os.path.isdir(tract_dir):
            continue
            
        tract_idx = int(tract_dir.split('_')[-1])
        tract_name = tract_names.get(tract_idx, f"tract_{tract_idx}")
        
        # Try to load summary file
        summary_file = os.path.join(tract_dir, "tract_summary.json")
        training_file = os.path.join(tract_dir, "training_results.json")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                all_results.append(summary)
        elif os.path.exists(training_file):
            # Fallback to training results file
            with open(training_file, 'r') as f:
                try:
                    results = json.load(f)
                    summary = {
                        "tract_idx": tract_idx,
                        "tract_name": tract_name,
                        "best_r2": results.get("best_val_r2", 0),
                        "best_age_mae": results.get("best_val_mae", 0)
                    }
                    all_results.append(summary)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON from {training_file}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Ensure we have the right columns
    if 'best_r2' not in df.columns and 'best_val_r2' in df.columns:
        df['best_r2'] = df['best_val_r2']
    
    if 'best_age_mae' not in df.columns and 'best_val_age_mae' in df.columns:
        df['best_age_mae'] = df['best_val_age_mae']
    
    return df

def bar_chart(df, output_dir, metric='best_r2', top_n=20, sort_ascending=False):
    """Create a horizontal bar chart of tract importance."""
    if metric not in df.columns:
        print(f"Error: metric '{metric}' not found in results")
        return
    
    # Sort the data
    df_sorted = df.sort_values(metric, ascending=sort_ascending).head(top_n)
    
    plt.figure(figsize=(12, 10))
    
    # Create the bar chart
    bars = plt.barh(df_sorted['tract_name'], df_sorted[metric])
    
    # Add value labels to the bars
    for i, v in enumerate(df_sorted[metric]):
        plt.text(v + 0.01, i, f"{v:.3f}", va='center')
    
    # Customize the chart
    metric_name = metric.replace('best_', '').upper()
    plt.xlabel(f'{metric_name}')
    plt.ylabel('Tract Name')
    plt.title(f'Top {top_n} Tracts Ranked by {metric_name}')
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f"tract_importance_{metric}.png")
    plt.savefig(output_file)
    print(f"Saved bar chart to {output_file}")
    plt.close()

def heatmap(df, output_dir):
    """Create a heatmap comparing R² and MAE."""
    if 'best_r2' not in df.columns or 'best_age_mae' not in df.columns:
        print("Error: R² or MAE metrics not found in results")
        return
    
    # Normalize the metrics for fair comparison (higher is better for both)
    df_plot = df.copy()
    df_plot['mae_normalized'] = 1 - (df_plot['best_age_mae'] - df_plot['best_age_mae'].min()) / (df_plot['best_age_mae'].max() - df_plot['best_age_mae'].min())
    
    # Create a matrix for the heatmap
    matrix = df_plot[['best_r2', 'mae_normalized']].values
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".3f", 
        xticklabels=['R²', 'Inverse MAE'],
        yticklabels=df_plot['tract_name'],
        cmap="YlGnBu"
    )
    plt.title('Tract Importance - Multiple Metrics Comparison')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "tract_importance_heatmap.png")
    plt.savefig(output_file)
    print(f"Saved heatmap to {output_file}")
    plt.close()

def scatter_plot(df, output_dir):
    """Create a scatter plot showing R² vs MAE."""
    if 'best_r2' not in df.columns or 'best_age_mae' not in df.columns:
        print("Error: R² or MAE metrics not found in results")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    scatter = plt.scatter(
        df['best_r2'], 
        df['best_age_mae'],
        c=df.index,  # Color by index
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    # Add labels for top 5 tracts
    df_top5 = df.sort_values('best_r2', ascending=False).head(5)
    for i, row in df_top5.iterrows():
        plt.annotate(
            row['tract_name'],
            (row['best_r2'], row['best_age_mae']),
            fontsize=10,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Customize the plot
    plt.xlabel('R² (higher is better)')
    plt.ylabel('MAE (lower is better)')
    plt.title('Tract Importance: R² vs MAE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    output_file = os.path.join(output_dir, "tract_importance_scatter.png")
    plt.savefig(output_file)
    print(f"Saved scatter plot to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize tract importance results")
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing tract evaluation results')
    parser.add_argument('--top-n', type=int, default=20, help='Show top N tracts in visualizations')
    parser.add_argument('--all-plots', action='store_true', help='Generate all plot types')
    parser.add_argument('--bar-chart', action='store_true', help='Generate bar chart')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap')
    parser.add_argument('--scatter', action='store_true', help='Generate scatter plot')
    args = parser.parse_args()
    
    # Set default to bar chart if no specific plots are requested
    if not (args.all_plots or args.bar_chart or args.heatmap or args.scatter):
        args.bar_chart = True
    
    # Load the results
    results_df = load_tract_results(args.input_dir)
    
    if results_df.empty:
        print("No results found to plot")
        return
    
    print(f"Loaded data for {len(results_df)} tracts")
    
    # Generate the requested plots
    if args.all_plots or args.bar_chart:
        bar_chart(results_df, args.input_dir, metric='best_r2', top_n=args.top_n)
        bar_chart(results_df, args.input_dir, metric='best_age_mae', top_n=args.top_n, sort_ascending=True)
    
    if args.all_plots or args.heatmap:
        heatmap(results_df, args.input_dir)
    
    if args.all_plots or args.scatter:
        scatter_plot(results_df, args.input_dir)
    
    # Print top tracts
    print("\nTop 10 tracts by R²:")
    for i, row in results_df.sort_values('best_r2', ascending=False).head(10).iterrows():
        print(f"{i+1}. {row['tract_name']} (idx: {row['tract_idx']}): R²={row['best_r2']:.4f}, MAE={row['best_age_mae']:.4f}")

if __name__ == "__main__":
    main()
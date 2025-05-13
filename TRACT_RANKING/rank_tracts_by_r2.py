#!/usr/bin/env python3
import os
import json
import pandas as pd
import glob
import sys
import re
from datetime import datetime

def extract_metrics(tract_dir):
    """
    Extract the best R² and MAE values from a tract's training_results.json file.
    Returns a tuple of (tract_index, best_val_r2, best_val_mae).
    """
    try:
        # Extract tract index from directory name
        dir_name = os.path.basename(tract_dir.rstrip('/'))
        match = re.search(r'tract_(\d+)', dir_name)
        if not match:
            print(f"Warning: Could not extract tract index from directory name: {dir_name}")
            return None
        
        tract_idx = int(match.group(1))
        
        result_file = os.path.join(tract_dir, "training_results.json")
        if not os.path.exists(result_file):
            print(f"Warning: No training_results.json found in {tract_dir}")
            return None
        
        with open(result_file, 'r') as f:
            data = json.load(f)
            
        # Get the metrics from the age predictor
        if 'age_predictor' in data:
            age_predictor = data['age_predictor']
            
            # Get best R² (highest value)
            val_r2_values = age_predictor.get('val_r2_epoch', [])
            if val_r2_values:
                # Filter out NaN and -inf values
                filtered_r2 = [r for r in val_r2_values if isinstance(r, (int, float)) and r > -float('inf')]
                best_r2 = max(filtered_r2) if filtered_r2 else 0
            else:
                best_r2 = 0
            
            # Get best MAE (lowest value)
            if 'best_val_mae' in age_predictor and isinstance(age_predictor['best_val_mae'], (int, float)):
                best_mae = age_predictor['best_val_mae']
            else:
                val_loss_values = age_predictor.get('val_loss_epoch', [])
                if val_loss_values:
                    # Filter out NaN and inf values
                    filtered_mae = [m for m in val_loss_values if isinstance(m, (int, float)) and m < float('inf')]
                    best_mae = min(filtered_mae) if filtered_mae else 0
                else:
                    best_mae = 0
            
            return (tract_idx, best_r2, best_mae)
        else:
            print(f"Warning: No age_predictor data found in {tract_dir}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {tract_dir}: {e}")
        return None
    except IndexError as e:
        print(f"Error: Index error in {tract_dir}: {e}")
        # Try to extract basic information even if there's an index error
        try:
            if 'tract_idx' in locals() and 'data' in locals() and 'age_predictor' in data:
                return (tract_idx, 0, 0)  # Return zeros for metrics as fallback
        except:
            pass
        return None
    except Exception as e:
        print(f"Error processing {tract_dir}: {e}")
        return None

def write_summary_report(df_by_r2, df_by_mae, df, output_dir):
    """Write a summary report file with rankings and statistics"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    summary_file = os.path.join(output_dir, 'tract_ranking_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("# Tract Ranking Summary Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary Statistics\n")
        f.write(f"Total tracts analyzed: {len(df)}\n")
        f.write(f"Average R² value: {df['best_r2'].mean():.4f}\n")
        f.write(f"Average MAE value: {df['best_mae'].mean():.4f}\n")
        f.write(f"Highest R² value: {df['best_r2'].max():.4f} (Tract {df.loc[df['best_r2'].idxmax(), 'tract_name']})\n")
        f.write(f"Lowest MAE value: {df['best_mae'].min():.4f} (Tract {df.loc[df['best_mae'].idxmin(), 'tract_name']})\n\n")
        
        f.write("## Top 10 Tracts by R² Value (higher is better)\n")
        f.write(df_by_r2.head(10).to_string(index=True) + "\n\n")
        
        f.write("## Top 10 Tracts by MAE Value (lower is better)\n")
        f.write(df_by_mae.head(10).to_string(index=True) + "\n\n")
        
        f.write("## Complete Rankings\n")
        f.write("### By R² Value (Descending)\n")
        f.write(df_by_r2.to_string(index=False) + "\n\n")
        
        f.write("### By MAE Value (Ascending)\n")
        f.write(df_by_mae.to_string(index=False) + "\n")
    
    print(f"Detailed summary report saved to {summary_file}")
    return summary_file

def main():
    # Determine the script directory
    if getattr(sys, 'frozen', False):
        script_dir = os.path.dirname(sys.executable)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're in the TRACT_RANKING directory, if not try to navigate there
    if not os.path.basename(os.getcwd()) == "TRACT_RANKING":
        if os.path.basename(script_dir) == "TRACT_RANKING":
            os.chdir(script_dir)
        else:
            potential_tract_ranking = os.path.join(os.getcwd(), "TRACT_RANKING")
            if os.path.exists(potential_tract_ranking):
                os.chdir(potential_tract_ranking)
    
    # Define the output directory
    output_dir = os.path.join(os.getcwd(), "rankings")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all tract directories
    tract_dirs = glob.glob("tract_*/")
    if not tract_dirs:
        print("Warning: No tract directories found in current directory.")
        print(f"Current directory: {os.getcwd()}")
        tract_dirs = glob.glob(os.path.join(script_dir, "tract_*/"))
        if not tract_dirs:
            print("Error: Could not find any tract directories.")
            return
    
    print(f"Found {len(tract_dirs)} tract directories")
    results = []
    
    for tract_dir in tract_dirs:
        result = extract_metrics(tract_dir)
        if result:
            results.append(result)
    
    if not results:
        print("Error: No valid results could be extracted from any tract directory.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=['tract_idx', 'best_r2', 'best_mae'])
    
    # Load tract names if available
    tract_names_file = os.path.join(os.getcwd(), 'tract_names.json')
    if os.path.exists(tract_names_file):
        try:
            with open(tract_names_file, 'r') as f:
                tract_names_list = json.load(f)
            
            # Add tract names to DataFrame
            # The tract_names.json is a list, so use the index to get the name if in range
            def get_tract_name(idx):
                if 0 <= idx < len(tract_names_list):
                    return tract_names_list[idx]
                else:
                    return f"tract_{idx}"
                
            df['tract_name'] = df['tract_idx'].apply(get_tract_name)
        except Exception as e:
            print(f"Warning: Could not load tract names from {tract_names_file}: {e}")
            df['tract_name'] = df['tract_idx'].apply(lambda idx: f"tract_{idx}")
    else:
        print(f"Warning: Tract names file not found at {tract_names_file}")
        df['tract_name'] = df['tract_idx'].apply(lambda idx: f"tract_{idx}")
    
    # Reorder columns
    df = df[['tract_idx', 'tract_name', 'best_r2', 'best_mae']]
    
    # Sort by R² value in descending order
    df_by_r2 = df.sort_values(by='best_r2', ascending=False).reset_index(drop=True)
    
    # Sort by MAE value in ascending order (lower is better)
    df_by_mae = df.sort_values(by='best_mae', ascending=True).reset_index(drop=True)
    
    # Save to CSV
    r2_output_file = os.path.join(output_dir, 'tract_ranking_by_best_r2.csv')
    mae_output_file = os.path.join(output_dir, 'tract_ranking_by_best_mae.csv')
    
    df_by_r2.to_csv(r2_output_file, index=False)
    df_by_mae.to_csv(mae_output_file, index=False)
    
    print(f"CSV files saved to {output_dir}")
    
    # Write detailed summary report
    summary_file = write_summary_report(df_by_r2, df_by_mae, df, output_dir)
    
    # Display results
    print(f"\nResults saved to rankings folder:")
    print(f"  - {os.path.basename(r2_output_file)}")
    print(f"  - {os.path.basename(mae_output_file)}")
    print(f"  - {os.path.basename(summary_file)}")
    
    print("\nTop 10 tracts by R² value (higher is better):")
    print(df_by_r2.head(10).to_string(index=True))
    
    print("\nTop 10 tracts by MAE value (lower is better):")
    print(df_by_mae.head(10).to_string(index=True))
    
    # Generate summary statistics
    print("\nSummary Statistics:")
    print(f"Total tracts analyzed: {len(df)}")
    print(f"Average R² value: {df['best_r2'].mean():.4f}")
    print(f"Average MAE value: {df['best_mae'].mean():.4f}")
    print(f"Highest R² value: {df['best_r2'].max():.4f} (Tract {df.loc[df['best_r2'].idxmax(), 'tract_name']})")
    print(f"Lowest MAE value: {df['best_mae'].min():.4f} (Tract {df.loc[df['best_mae'].idxmin(), 'tract_name']})")

if __name__ == "__main__":
    main() 
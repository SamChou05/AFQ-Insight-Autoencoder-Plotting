# Results Summary - Cross-Experiment Performance Analysis

This directory contains high-level performance summaries aggregated across all experiments in the repository.

## Purpose

The results summary provides a bird's-eye view of model performance across different:
- Autoencoder architectures (VAE, standard AE, convolutional)
- Training strategies (staged, alternating, combined)
- Prediction tasks (age, site, combined)
- Data types (FA, MD, combined tract data)

## Files

### Main Summary Files
- `overall_model_summary.csv` - Complete performance comparison across all experiments
- `age_predictor_summary.csv` - Age prediction performance across models
- `combined_model_summary.csv` - Combined age+site prediction performance
- `site_predictor_summary.csv` - Site prediction performance comparison

### File Contents

Each CSV file typically contains:
- **Experiment Name** - Identifier for the specific experiment
- **Model Type** - Architecture (VAE, AE, ConvAE, etc.)
- **R² Score** - Coefficient of determination for regression tasks
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Square Error
- **Accuracy** - Classification accuracy (for site prediction)
- **Training Time** - Time to convergence
- **Best Epoch** - Epoch with optimal performance
- **Data Type** - Input data characteristics

## Data Transfer Instructions

### From Individual Experiment Directories

After completing analysis in individual experiment folders:

```bash
# Collect CSV files from all experiments
cp FA_VAE_AGE_SITE_STAGES/csv_files/*summary.csv results_summary/
cp ALTERNATING_VAE_AGE_SITE_STAGES/csv_files/*summary.csv results_summary/
cp VAE_AGE_SITE/csv_files/*summary.csv results_summary/
# ... repeat for all experiment directories

# Or use a script to aggregate automatically
python aggregate_results.py  # If available
```

### From Experiments Repository

```bash
# Copy pre-computed summaries
scp experiments_repo/results_summary/*.csv plotting_repo/results_summary/
```

## Analysis Workflow

1. **Data Collection:** Gather summary metrics from all experiments
2. **Aggregation:** Combine results into comprehensive comparison tables
3. **Ranking:** Identify top-performing models and configurations
4. **Visualization:** Create comparison plots and performance matrices

## Key Metrics for Comparison

### Age Prediction Performance
- **R² Score:** How well the model explains age variance
- **MAE:** Average absolute error in years
- **RMSE:** Root mean square error in years

### Site Prediction Performance  
- **Accuracy:** Percentage of correctly classified sites
- **F1 Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed classification breakdown

### Reconstruction Quality
- **Reconstruction Loss:** How well the autoencoder preserves input data
- **Latent Dimension:** Size of the compressed representation
- **Compression Ratio:** Data reduction achieved

## Usage Examples

### Quick Performance Overview
```python
import pandas as pd

# Load overall summary
summary = pd.read_csv('overall_model_summary.csv')

# Find best age predictor
best_age = summary.loc[summary['Age_R2'].idxmax()]
print(f"Best age predictor: {best_age['Experiment']} (R² = {best_age['Age_R2']:.3f})")

# Find best site predictor  
best_site = summary.loc[summary['Site_Accuracy'].idxmax()]
print(f"Best site predictor: {best_site['Experiment']} (Acc = {best_site['Site_Accuracy']:.3f})")
```

### Performance Comparison
```python
# Compare VAE vs standard AE
vae_results = summary[summary['Model_Type'].str.contains('VAE')]
ae_results = summary[summary['Model_Type'].str.contains('AE') & ~summary['Model_Type'].str.contains('VAE')]

print("VAE Performance:")
print(vae_results[['Experiment', 'Age_R2', 'Site_Accuracy']].describe())

print("Standard AE Performance:")  
print(ae_results[['Experiment', 'Age_R2', 'Site_Accuracy']].describe())
```

## Expected Results Structure

The summary files help answer key research questions:

1. **Which architecture works best?** Compare VAE vs ConvAE vs FC-AE
2. **What training strategy is optimal?** Compare staged vs alternating vs joint training  
3. **How important is data type?** Compare FA vs MD vs combined tract data
4. **What are the trade-offs?** Balance between age prediction, site prediction, and reconstruction quality

## Visualization Recommendations

Create comparison plots using the summary data:
- Performance matrices showing R² scores across experiments
- Bar charts comparing different model architectures
- Scatter plots showing trade-offs between different metrics
- Heatmaps highlighting best-performing configurations

This directory serves as the starting point for understanding overall project results and identifying the most promising approaches for further investigation. 
# Training Averages - Multi-Run Aggregation Analysis

This directory contains aggregated training statistics across multiple runs of the same experiments, providing insights into training stability and convergence patterns.

## Purpose

Training averages help assess:
- **Reproducibility:** How consistent are results across different random seeds/runs?
- **Convergence Stability:** Do models converge reliably to similar performance?
- **Training Dynamics:** What are typical learning curves and convergence times?
- **Hyperparameter Sensitivity:** How much do results vary with small parameter changes?

## Directory Contents

### Aggregated Metrics Files
- `age_predictor_epoch_averages.csv` - Average age prediction performance per epoch
- `age_predictor_epoch_summary.csv` - Summary statistics for age prediction training
- `combined_model_epoch_averages.csv` - Combined model training averages
- `combined_model_epoch_summary.csv` - Combined model summary statistics
- `site_predictor_epoch_averages.csv` - Site prediction training averages
- `vae_epoch_averages.csv` - VAE training performance averages

### Visualization Outputs
- `plots/` directory containing:
  - Training curve comparisons
  - Confidence interval plots
  - Convergence time distributions
  - Performance stability visualizations

## File Structure

### Epoch Averages Files
These contain epoch-by-epoch statistics across multiple runs:
```csv
Epoch, Mean_Loss, Std_Loss, Mean_Accuracy, Std_Accuracy, Mean_R2, Std_R2, Num_Runs
1,     0.85,      0.05,     0.12,          0.02,        0.05,    0.01,   10
2,     0.78,      0.04,     0.18,          0.03,        0.12,    0.02,   10
...
```

### Summary Files  
These contain high-level statistics:
```csv
Metric,           Mean,    Std,     Min,     Max,     Q25,     Q75
Best_R2,          0.76,    0.05,    0.68,    0.83,    0.73,    0.79
Convergence_Epoch,45.2,    8.3,     32,      58,      39,      52
Final_Loss,       0.23,    0.03,    0.19,    0.28,    0.21,    0.25
```

## Data Collection Workflow

### 1. Multiple Experiment Runs
Run the same experiment multiple times with different seeds:
```bash
# In experiments repository
for seed in {1..10}; do
    python train_model.py --seed $seed --experiment_name "FA_VAE_run_$seed"
done
```

### 2. Transfer Individual Results
Copy each run's results to the plotting repository:
```bash
# Copy individual run results
for i in {1..10}; do
    scp -r experiments_repo/FA_VAE_run_$i/* plotting_repo/FA_VAE_AGE_SITE_STAGES/
done
```

### 3. Aggregate Results
Use aggregation scripts to compute averages:
```bash
cd training_averages/
python aggregate_training_runs.py --experiment_dir ../FA_VAE_AGE_SITE_STAGES/
```

## Analysis Capabilities

### Training Stability Assessment
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load epoch averages
averages = pd.read_csv('age_predictor_epoch_averages.csv')

# Plot training curves with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(averages['Epoch'], averages['Mean_R2'], label='Mean R²')
plt.fill_between(averages['Epoch'], 
                 averages['Mean_R2'] - averages['Std_R2'],
                 averages['Mean_R2'] + averages['Std_R2'], 
                 alpha=0.3, label='±1 Std')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Age Predictor Training Stability')
plt.legend()
plt.show()
```

### Convergence Analysis
```python
# Load summary statistics
summary = pd.read_csv('age_predictor_epoch_summary.csv')

# Analyze convergence characteristics
convergence_epochs = summary[summary['Metric'] == 'Convergence_Epoch']
print(f"Average convergence: {convergence_epochs['Mean'].iloc[0]:.1f} ± {convergence_epochs['Std'].iloc[0]:.1f} epochs")

# Check training stability (coefficient of variation)
final_r2 = summary[summary['Metric'] == 'Best_R2']
cv = final_r2['Std'].iloc[0] / final_r2['Mean'].iloc[0]
print(f"Training stability (CV): {cv:.3f}")
```

## Key Metrics Tracked

### Performance Stability
- **Mean ± Std Performance:** Average and variability of final metrics
- **Coefficient of Variation:** Relative stability measure
- **Min/Max Range:** Performance bounds across runs

### Convergence Characteristics  
- **Convergence Epoch:** When training typically stabilizes
- **Convergence Reliability:** Percentage of runs that converge successfully
- **Learning Rate:** Speed of performance improvement

### Training Dynamics
- **Loss Trajectories:** How loss decreases over time
- **Performance Plateaus:** Stable performance regions  
- **Training Efficiency:** Performance per training epoch

## Comparison Across Experiments

### Cross-Architecture Stability
```python
# Compare stability across different models
experiments = ['FA_VAE', 'ALTERNATING_VAE', 'CONV_AE']
stability_comparison = {}

for exp in experiments:
    summary = pd.read_csv(f'{exp}_epoch_summary.csv')
    r2_stats = summary[summary['Metric'] == 'Best_R2']
    stability_comparison[exp] = {
        'mean': r2_stats['Mean'].iloc[0],
        'std': r2_stats['Std'].iloc[0],
        'cv': r2_stats['Std'].iloc[0] / r2_stats['Mean'].iloc[0]
    }

# Identify most stable architecture
most_stable = min(stability_comparison.items(), key=lambda x: x[1]['cv'])
print(f"Most stable architecture: {most_stable[0]} (CV: {most_stable[1]['cv']:.3f})")
```

## Visualization Outputs

The `plots/` directory contains:

### Training Curves
- Multi-run training curves with confidence bands
- Convergence time distributions
- Performance stability over epochs

### Comparison Plots
- Side-by-side stability comparisons
- Architecture performance distributions
- Training efficiency comparisons

### Statistical Summaries
- Box plots of final performance
- Histogram of convergence epochs  
- Correlation matrices of training metrics

## Usage for Publication

This analysis is particularly valuable for:
- **Methods sections:** Demonstrating training stability and reproducibility
- **Results sections:** Providing confidence intervals and statistical significance
- **Supplementary materials:** Detailed training dynamics and hyperparameter sensitivity

The aggregated data supports claims about model reliability and provides the statistical foundation for comparing different approaches. 
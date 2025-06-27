# [EXPERIMENT_NAME] - [Brief Description]

> **Template:** Copy this file to your experiment directory and customize the content

This directory contains results and analysis for the [EXPERIMENT_NAME] experiment.

## Experiment Overview

**Objective:** [Brief description of what this experiment aims to achieve]

**Architecture:** [VAE/AE/ConvAE/etc.]

**Training Strategy:** [Staged/Alternating/Joint/etc.]

**Data Type:** [FA/MD/Combined]

**Key Parameters:**
- Latent dimension: [X]
- Learning rate: [X]
- Batch size: [X]
- Training epochs: [X]
- Other relevant hyperparameters

## Directory Structure

```
EXPERIMENT_NAME/
├── csv_files/                   # Training metrics and performance data
│   ├── age_predictor_metrics.csv
│   ├── site_predictor_metrics.csv
│   ├── combined_metrics.csv
│   └── vae_metrics.csv
├── model_weights/               # Saved PyTorch model files
│   ├── best_age_predictor.pth
│   ├── best_site_predictor.pth
│   ├── best_combined_model.pth
│   └── best_vae.pth
├── plots/                       # Generated visualizations
│   ├── training_curves/
│   ├── performance_metrics/
│   └── reconstructions/
├── confusion_matrix/            # Classification performance
│   ├── confusion_matrices/
│   └── site_predictions/
├── plot.ipynb                   # Main analysis notebook
├── reconstructions_comb.ipynb   # Reconstruction visualizations
├── vae_recon.ipynb             # VAE-specific analysis
└── plot_tracts.py              # Plotting utilities
```

## Data Transfer Instructions

### From Experiments Repository

After completing the experiment in the AFQ-Insight-Autoencoder-Experiments repository:

```bash
# Transfer results from experiments repository
scp -r experiments_repo/[EXPERIMENT_NAME]/* plotting_repo/[EXPERIMENT_NAME]/

# Or transfer specific file types:
scp experiments_repo/[EXPERIMENT_NAME]/csv_files/*.csv plotting_repo/[EXPERIMENT_NAME]/csv_files/
scp experiments_repo/[EXPERIMENT_NAME]/model_weights/*.pth plotting_repo/[EXPERIMENT_NAME]/model_weights/
scp experiments_repo/[EXPERIMENT_NAME]/*.json plotting_repo/[EXPERIMENT_NAME]/
```

### Expected Files

Ensure the following files are transferred:

#### CSV Files (→ `csv_files/`)
- [ ] `age_predictor_metrics.csv` - Age prediction performance
- [ ] `site_predictor_metrics.csv` - Site prediction performance  
- [ ] `combined_metrics.csv` - Combined model performance
- [ ] `vae_metrics.csv` - VAE reconstruction metrics
- [ ] `*_summary.csv` - Summary statistics

#### Model Weights (→ `model_weights/`)
- [ ] `best_age_predictor.pth` - Best age prediction model
- [ ] `best_site_predictor.pth` - Best site prediction model
- [ ] `best_combined_model.pth` - Best combined model
- [ ] `best_vae.pth` - Best VAE model
- [ ] Additional epoch checkpoints (optional)

#### Configuration Files (→ root directory)
- [ ] `experiment_details.json` - Hyperparameters and settings
- [ ] Any additional configuration files

## Analysis Workflow

### 1. Data Verification
```bash
# Check that all required files are present
ls csv_files/
ls model_weights/
```

### 2. Launch Analysis
```bash
# Start Jupyter notebook
jupyter notebook

# Run notebooks in recommended order:
# 1. plot.ipynb - Performance analysis
# 2. reconstructions_comb.ipynb - Reconstruction quality
# 3. vae_recon.ipynb - VAE-specific analysis
```

### 3. Key Analysis Steps

1. **Performance Metrics**
   - Load training curves from CSV files
   - Generate loss and accuracy plots
   - Compute final performance statistics

2. **Model Comparison**
   - Compare age vs site prediction performance
   - Analyze training stability and convergence
   - Evaluate reconstruction quality

3. **Visualization**
   - Create training curve plots
   - Generate performance comparison charts
   - Visualize reconstruction examples

## Expected Results

### Performance Metrics
- **Age Prediction R²:** [Expected range]
- **Site Prediction Accuracy:** [Expected range]
- **Reconstruction Loss:** [Expected range]
- **Training Time:** [Expected duration]

### Key Findings
[Document key results and insights from this experiment]

### Comparison to Other Experiments
[How does this experiment compare to other approaches?]

## Troubleshooting

### Common Issues

1. **Missing Files**
   - Verify all CSV files transferred correctly
   - Check that model weights are present
   - Ensure configuration files are available

2. **Notebook Errors**
   - Install required dependencies: `pip install pandas matplotlib seaborn torch`
   - Check file paths in notebook cells
   - Verify CSV file formats and column names

3. **Memory Issues**
   - Large model files may require sufficient RAM
   - Consider using CPU-only versions for analysis

### File Format Notes
- CSV files should be comma-separated with headers
- Model weights should be PyTorch .pth files
- Configuration files should be valid JSON

## Customization

### Adding New Analysis
1. Create new notebook or modify existing ones
2. Add new plotting functions to `plot_tracts.py`
3. Update this README with new analysis steps

### Comparing to Other Experiments
1. Copy metrics to `results_summary/` directory
2. Update overall comparison notebooks
3. Add experiment to cross-experiment analysis

## Publication Notes

**Figures for Paper:**
- [List key figures generated from this experiment]

**Key Statistics:**
- [List important metrics to report]

**Comparison Points:**
- [How this experiment relates to others in the paper]

---

**Experiment Status:** [Completed/In Progress/Planned]
**Last Updated:** [Date]
**Responsible Researcher:** [Name] 
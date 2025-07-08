# AFQ-Insight-Autoencoder-Plotting

This repository contains the plotting and analysis tools for visualizing results from autoencoder experiments conducted in the [AFQ-Insight-Autoencoder-Experiments](https://github.com/SamChou05/AFQ-Insight-Autoencoder-Experiments) repository. 

## Repository Purpose

This is a **results analysis repository** that works in tandem with the experiments repository. After running experiments in the AFQ-Insight-Autoencoder-Experiments repository, you transfer the results (CSV files, model weights, etc.) to this repository for visualization and analysis.

## Workflow Overview

```
1. Run experiments in AFQ-Insight-Autoencoder-Experiments
2. Copy results to this repository using scp or similar
3. Run Jupyter notebooks in this repository for plotting and analysis
4. Generate visualizations and reconstructions for paper/presentation
```

## Repository Structure

### Main Experiment Directories

Each directory represents a different type of autoencoder experiment:

- **`FA_VAE_AGE_SITE_STAGES/`** - Fractional Anisotropy VAE with age and site prediction in stages
- **`ALTERNATING_VAE_AGE_SITE_STAGES/`** - Alternating training VAE with age and site prediction
- **`VAE_AGE_SITE/`** - Basic VAE with age and site prediction
- **`VAE_AGE_SITE_STAGES/`** - Staged VAE training with age and site prediction
- **`VAE_AGE_SITE_STAGES_ALL/`** - Extended staged VAE training
- **`TRACT_RANKING/`** - Individual tract analysis and ranking experiments
- **`ConvAE_Experiments/`** - Convolutional autoencoder experiments
- **`FC_AE_Experiments/`** - Fully connected autoencoder experiments

### Support Directories

- **`training_averages/`** - Aggregated training metrics across experiments
- **`results_summary/`** - Overall performance summaries
- **`plots/`** - General plotting utilities and outputs
- **`Experiment_Utils/`** - Shared utility functions and model definitions

### Standard Folder Structure Within Each Experiment

Each experiment directory typically contains:

```
experiment_name/
├── csv_files/              # Training metrics, performance data
├── model_weights/          # Saved PyTorch model files (.pth)
├── plots/                  # Generated visualization outputs
├── confusion_matrix/       # Classification performance matrices
├── plot.ipynb             # Main plotting notebook
├── reconstructions_comb.ipynb  # Reconstruction visualization notebook
├── vae_recon.ipynb        # VAE-specific reconstruction analysis
└── plot_tracts.py         # Python plotting utilities
```

## How to Transfer Results

### From Experiments Repository

After completing experiments in the AFQ-Insight-Autoencoder-Experiments repository:

#### Method 1: Using scp (recommended for remote servers)

```bash
# From your experiments repository location
scp -r path/to/experiment_results/* username@hostname:/path/to/AFQ-Insight-Autoencoder-Plotting/EXPERIMENT_NAME/

# Example for specific file types:
scp path/to/experiments/csv_files/*.csv username@hostname:/path/to/plotting/FA_VAE_AGE_SITE_STAGES/csv_files/
scp path/to/experiments/model_weights/*.pth username@hostname:/path/to/plotting/FA_VAE_AGE_SITE_STAGES/model_weights/
```

#### Method 2: Direct copy (local machine)

```bash
# Copy CSV files
cp experiments_repo/experiment_name/csv_files/*.csv plotting_repo/EXPERIMENT_NAME/csv_files/

# Copy model weights
cp experiments_repo/experiment_name/model_weights/*.pth plotting_repo/EXPERIMENT_NAME/model_weights/

# Copy any additional results
cp experiments_repo/experiment_name/results/* plotting_repo/EXPERIMENT_NAME/
```

### File Types to Transfer

1. **CSV Files** (`*.csv`) → `csv_files/` directory
   - Training histories
   - Performance metrics
   - Evaluation results

2. **Model Weights** (`*.pth`) → `model_weights/` directory
   - Best models
   - Epoch checkpoints
   - Component models (VAE, age predictor, site predictor)

3. **Configuration Files** (`*.json`) → Experiment root
   - Experiment parameters
   - Model configurations

4. **Log Files** (`*.out`, `*.log`) → Experiment root (optional)
   - SLURM outputs
   - Training logs

## Running Analysis

### Prerequisites

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt  # If available
# or
pip install jupyter pandas matplotlib seaborn torch numpy scikit-learn
```

### Analysis Workflow

1. **Navigate to experiment directory:**
   ```bash
   cd FA_VAE_AGE_SITE_STAGES/  # or your target experiment
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Run notebooks in order:**
   - `plot.ipynb` - Generate performance plots and training curves
   - `reconstructions_comb.ipynb` - Visualize model reconstructions
   - `vae_recon.ipynb` - VAE-specific reconstruction analysis

### Key Analysis Outputs

- **Training curves** - Loss and accuracy over epochs
- **Performance metrics** - R² scores, classification accuracy
- **Reconstruction visualizations** - Original vs reconstructed tract profiles
- **Confusion matrices** - Classification performance breakdown
- **Summary statistics** - Aggregated model performance

## Experiment-Specific Notes

### TRACT_RANKING
- Contains individual analysis for each of 48 brain tracts
- Each `tract_X_dki_XX/` folder contains complete training history for that tract
- Use `plotting_averages.py` for cross-tract comparisons

### Training Averages
- Aggregates results across multiple experiment runs
- Useful for understanding training stability and convergence patterns

### Results Summary
- High-level performance comparisons across different model architectures
- Good starting point for understanding which experiments performed best

## Customization

### Adding New Experiments

1. Create new directory following naming convention
2. Create standard subdirectories: `csv_files/`, `model_weights/`, `plots/`
3. Copy and adapt notebooks from existing experiments
4. Update this README with experiment description

### Modifying Plots

- Edit Jupyter notebooks for custom visualizations
- Modify `plot_tracts.py` for programmatic plotting
- Add new plotting functions to `Experiment_Utils/utils.py`

## Troubleshooting

### Common Issues

1. **Missing dependencies:** Install required packages listed in notebooks
2. **File path errors:** Ensure CSV and model files are in correct directories
3. **CUDA errors:** Some model loading may require GPU; use CPU versions if needed
4. **Memory issues:** Large model files may require sufficient RAM

### File Organization Tips

- Keep consistent naming between experiments and plotting directories
- Use descriptive experiment names that match the research phase
- Maintain the standard folder structure for easy navigation
- Document any custom modifications in experiment-specific README files

## Contributing

When passing this repository to another researcher:

1. Ensure all experiment results are properly transferred
2. Update experiment descriptions in this README
3. Document any custom analysis methods
4. Include environment/dependency information
5. Provide example command lines for common operations

## Related Repositories

- **Experiments Repository:** [AFQ-Insight-Autoencoder-Experiments](https://github.com/SamChou05/AFQ-Insight-Autoencoder-Experiments)
- **AFQ-Insight Base:** [AFQ-Insight](https://github.com/richford/AFQ-Insight)

---

*This repository is designed to work alongside the AFQ-Insight-Autoencoder-Experiments repository. Results from experiments should be transferred here for visualization and analysis.* 

## License

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
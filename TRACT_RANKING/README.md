# TRACT_RANKING - Individual Tract Analysis

This directory contains experiments analyzing individual brain tracts to understand their relative importance and performance in autoencoder models.

## Directory Structure

### Individual Tract Folders
- `tract_0_dki_fa/` through `tract_23_dki_fa/` - Fractional Anisotropy tracts (24 tracts)
- `tract_24_dki_md/` through `tract_47_dki_md/` - Mean Diffusivity tracts (24 tracts)

Each tract folder contains:
```
tract_X_dki_XX/
├── age_predictor_training_history.csv    # Age prediction training metrics
├── site_predictor_training_history.csv   # Site prediction training metrics
├── combined_model_training_history.csv   # Combined model training metrics
├── vae_training_history.csv             # VAE training metrics
├── best_age_predictor.pth               # Best age prediction model
├── best_site_predictor.pth              # Best site prediction model
├── best_combined_model.pth              # Best combined model
├── best_vae.pth                         # Best VAE model
└── experiment_details.json             # Experiment configuration
```

### Summary Files
- `summary_results.csv` - Performance metrics across all tracts
- `tract_names.json` - Mapping of tract IDs to anatomical names
- `tract_data_types.json` - Tract data type specifications (FA vs MD)

### Analysis Scripts
- `plotting_averages.py` - Generate cross-tract performance comparisons
- `tract_training_averages.py` - Aggregate training statistics
- `tract_result_summary.py` - Create summary reports

## Data Transfer Instructions

### From Experiments Repository

After running tract ranking experiments:

```bash
# Copy individual tract results
for i in {0..47}; do
    if [ $i -le 23 ]; then
        scp -r experiments_repo/tract_${i}_dki_fa/* plotting_repo/TRACT_RANKING/tract_${i}_dki_fa/
    else
        scp -r experiments_repo/tract_${i}_dki_md/* plotting_repo/TRACT_RANKING/tract_${i}_dki_md/
    fi
done

# Copy summary files
scp experiments_repo/summary_results.csv plotting_repo/TRACT_RANKING/
scp experiments_repo/tract_*.json plotting_repo/TRACT_RANKING/
```

## Analysis Workflow

1. **Transfer Results:** Copy all tract experiment results to appropriate folders
2. **Run Summary Scripts:** 
   ```bash
   python tract_result_summary.py    # Generate overall summary
   python tract_training_averages.py # Compute training averages
   python plotting_averages.py       # Create comparison plots
   ```
3. **Individual Analysis:** Navigate to specific tract folders for detailed analysis

## Key Analysis Questions

- Which tracts are most predictive of age?
- Which tracts are most predictive of acquisition site?
- How does reconstruction quality vary across tracts?
- What is the relative importance of FA vs MD tracts?

## Output Visualizations

- Cross-tract performance comparisons
- Training curve overlays
- Reconstruction quality heatmaps
- Feature importance rankings

## Tract Information

### FA (Fractional Anisotropy) Tracts: 0-23
These measure the directional coherence of water diffusion, indicating white matter integrity.

### MD (Mean Diffusivity) Tracts: 24-47
These measure the average rate of water diffusion, indicating tissue microstructure.

Refer to `tract_names.json` for the anatomical names corresponding to each tract ID. 
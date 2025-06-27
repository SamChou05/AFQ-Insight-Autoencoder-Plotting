# File Transfer Checklist

Use this checklist to ensure all necessary files are transferred from the experiments repository to the plotting repository.

## Before Transfer

- [ ] **Experiment completed** in AFQ-Insight-Autoencoder-Experiments repository
- [ ] **Results directory** identified in experiments repository  
- [ ] **Target directory** created/confirmed in plotting repository
- [ ] **Network access** available (if using scp from remote server)

## Required Files Transfer

### CSV Files → `csv_files/` directory
- [ ] `age_predictor_metrics.csv` - Age prediction training history
- [ ] `site_predictor_metrics.csv` - Site prediction training history
- [ ] `combined_metrics.csv` - Combined model training history
- [ ] `vae_metrics.csv` - VAE training metrics
- [ ] `*_summary.csv` - Summary statistics files
- [ ] Any additional metrics files

### Model Weights → `model_weights/` directory
- [ ] `best_age_predictor.pth` - Best age prediction model
- [ ] `best_site_predictor.pth` - Best site prediction model
- [ ] `best_combined_model.pth` - Best combined model
- [ ] `best_vae.pth` - Best VAE model
- [ ] Epoch checkpoints (optional): `*_epoch_*.pth`
- [ ] Any additional model files

### Configuration Files → Experiment root directory
- [ ] `experiment_details.json` - Hyperparameters and settings
- [ ] `model_config.json` - Model architecture details
- [ ] Any other configuration files

## Optional Files Transfer

### Log Files → Experiment root directory
- [ ] `*.out` - SLURM output files
- [ ] `*.log` - Training log files
- [ ] `*.err` - Error log files

### Additional Analysis Files
- [ ] Confusion matrices → `confusion_matrix/` directory
- [ ] Preliminary plots → `plots/` directory
- [ ] Any custom analysis scripts

## Transfer Commands

### Using scp (Remote Server)
```bash
# Complete directory transfer
scp -r username@server:/path/to/experiments/EXPERIMENT_NAME/* /local/path/plotting/EXPERIMENT_NAME/

# Specific file types
scp username@server:/path/to/experiments/EXPERIMENT_NAME/csv_files/*.csv /local/path/plotting/EXPERIMENT_NAME/csv_files/
scp username@server:/path/to/experiments/EXPERIMENT_NAME/model_weights/*.pth /local/path/plotting/EXPERIMENT_NAME/model_weights/
scp username@server:/path/to/experiments/EXPERIMENT_NAME/*.json /local/path/plotting/EXPERIMENT_NAME/
```

### Using cp (Local Transfer)
```bash
# Complete directory transfer
cp -r /path/to/experiments/EXPERIMENT_NAME/* /path/to/plotting/EXPERIMENT_NAME/

# Specific file types
cp /path/to/experiments/EXPERIMENT_NAME/csv_files/*.csv /path/to/plotting/EXPERIMENT_NAME/csv_files/
cp /path/to/experiments/EXPERIMENT_NAME/model_weights/*.pth /path/to/plotting/EXPERIMENT_NAME/model_weights/
cp /path/to/experiments/EXPERIMENT_NAME/*.json /path/to/plotting/EXPERIMENT_NAME/
```

## Post-Transfer Verification

### Check File Presence
- [ ] **CSV files present**: `ls EXPERIMENT_NAME/csv_files/`
- [ ] **Model weights present**: `ls EXPERIMENT_NAME/model_weights/`
- [ ] **Configuration files present**: `ls EXPERIMENT_NAME/*.json`

### Verify File Integrity
- [ ] **CSV files have data**: `head EXPERIMENT_NAME/csv_files/*.csv`
- [ ] **Model files are complete**: Check file sizes are reasonable (>1KB)
- [ ] **JSON files are valid**: `python -m json.tool EXPERIMENT_NAME/*.json`

### Test Data Loading
- [ ] **Open Jupyter notebook**: `jupyter notebook`
- [ ] **Run first cell** of `plot.ipynb` to test CSV loading
- [ ] **Check for error messages** in notebook output

## Experiment-Specific Checklists

### TRACT_RANKING Experiments
Additional files needed:
- [ ] `summary_results.csv` - Cross-tract performance
- [ ] `tract_names.json` - Tract ID to name mapping
- [ ] `tract_data_types.json` - Tract data specifications
- [ ] Individual tract directories: `tract_X_dki_XX/`

### Multi-Run Experiments
For experiments with multiple runs:
- [ ] **All run directories** transferred
- [ ] **Run naming consistent** (e.g., `run_1/`, `run_2/`, etc.)
- [ ] **Aggregation scripts** available in `training_averages/`

## Common Transfer Issues

### File Permission Errors
```bash
# Fix permissions if needed
chmod -R 755 EXPERIMENT_NAME/
```

### Large File Transfers
```bash
# Use compression for large transfers
scp -r -C username@server:/path/to/experiments/EXPERIMENT_NAME/* /local/path/plotting/EXPERIMENT_NAME/

# Or use rsync for resume capability
rsync -avz --progress username@server:/path/to/experiments/EXPERIMENT_NAME/ /local/path/plotting/EXPERIMENT_NAME/
```

### Partial Transfer Recovery
```bash
# Check what's missing
diff -r /path/to/experiments/EXPERIMENT_NAME/ /path/to/plotting/EXPERIMENT_NAME/

# Transfer only missing files
rsync -avz --update username@server:/path/to/experiments/EXPERIMENT_NAME/ /local/path/plotting/EXPERIMENT_NAME/
```

## Ready for Analysis

Once all items are checked:
- [ ] **All required files transferred and verified**
- [ ] **File permissions set correctly**
- [ ] **Dependencies installed**: `pip install jupyter pandas matplotlib seaborn torch`
- [ ] **Ready to run**: `cd EXPERIMENT_NAME && jupyter notebook`

## Notes

**Experiment:** ________________  
**Transfer Date:** ______________  
**Responsible Person:** __________  
**Special Instructions:** _________  
________________________________

---

**Next Step:** Run `jupyter notebook` in your experiment directory and open `plot.ipynb` to begin analysis! 
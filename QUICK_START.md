# Quick Start Guide

This guide will get you up and running with the AFQ-Insight-Autoencoder-Plotting repository in 10 minutes.

## Prerequisites

- Python 3.8+ with Jupyter notebook
- Access to results from [AFQ-Insight-Autoencoder-Experiments](https://github.com/SamChou05/AFQ-Insight-Autoencoder-Experiments)
- Basic familiarity with CSV files and PyTorch model files

## Step 1: Repository Overview (2 minutes)

This repository contains **analysis and visualization tools** for autoencoder experiments. You:
1. Run experiments in the AFQ-Insight-Autoencoder-Experiments repository
2. Copy results here using `scp` or `cp`
3. Run Jupyter notebooks to generate plots and analysis

## Step 2: Directory Structure (1 minute)

```
├── FA_VAE_AGE_SITE_STAGES/     # Main VAE experiment
├── ALTERNATING_VAE_AGE_SITE_STAGES/  # Alternating training VAE
├── TRACT_RANKING/              # Individual tract analysis
├── results_summary/            # Cross-experiment comparisons
├── training_averages/          # Multi-run aggregations
└── README.md                   # Full documentation
```

Each experiment directory has:
- `csv_files/` - Training metrics
- `model_weights/` - Saved models (.pth files)
- `plots/` - Generated visualizations
- `*.ipynb` - Analysis notebooks

## Step 3: Transfer Your Results (3 minutes)

### From Remote Server (most common)
```bash
# Copy entire experiment results
scp -r username@server:/path/to/experiments/FA_VAE_results/* FA_VAE_AGE_SITE_STAGES/

# Or copy specific file types
scp username@server:/path/to/experiments/csv_files/*.csv FA_VAE_AGE_SITE_STAGES/csv_files/
scp username@server:/path/to/experiments/model_weights/*.pth FA_VAE_AGE_SITE_STAGES/model_weights/
```

### From Local Machine
```bash
cp -r ../experiments_repo/FA_VAE_results/* FA_VAE_AGE_SITE_STAGES/
```

### What Files to Transfer
✅ **Required:**
- `*.csv` files (training metrics) → `csv_files/`
- `*.pth` files (model weights) → `model_weights/`

📋 **Optional but helpful:**
- `*.json` files (configurations)
- `*.log` or `*.out` files (training logs)

## Step 4: Install Dependencies (1 minute)

```bash
pip install jupyter pandas matplotlib seaborn torch numpy scikit-learn
```

Or if you have a requirements.txt:
```bash
pip install -r requirements.txt
```

## Step 5: Run Your First Analysis (3 minutes)

### Navigate to Your Experiment
```bash
cd FA_VAE_AGE_SITE_STAGES/  # or your experiment directory
```

### Launch Jupyter
```bash
jupyter notebook
```

### Run Notebooks in Order
1. **`plot.ipynb`** - Start here for performance plots
2. **`reconstructions_comb.ipynb`** - Visualize model reconstructions  
3. **`vae_recon.ipynb`** - VAE-specific analysis

### What You'll See
After running `plot.ipynb`, you'll get:
- Training loss curves
- R² performance plots
- Age vs Site prediction comparisons
- Model convergence analysis

## Troubleshooting (Quick Fixes)

### "File not found" errors
```bash
# Check your files are in the right place
ls csv_files/
ls model_weights/
```

### "Module not found" errors
```bash
pip install [missing_module_name]
```

### "Empty plots" or "No data"
- Verify CSV files have data (not just headers)
- Check that column names match what notebooks expect

## Next Steps

### For Individual Experiment Analysis
- Explore the specific experiment directory
- Customize notebooks for your specific research questions
- Generate publication-ready figures

### For Cross-Experiment Comparison
- Visit `results_summary/` for overall performance comparisons
- Check `training_averages/` for multi-run stability analysis
- Use `TRACT_RANKING/` for individual tract insights

### For Adding New Experiments
1. Create new directory following the naming pattern
2. Copy `EXPERIMENT_TEMPLATE.md` and customize it
3. Transfer your results and run analysis

## Common Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Run Experiments │───▶│ Copy Results     │───▶│ Generate Plots  │
│ (other repo)    │    │ (scp/cp)         │    │ (Jupyter)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ Analyze Results  │
                       │ Compare Models   │
                       └──────────────────┘
```

## Getting Help

1. **Full Documentation:** See `README.md` for comprehensive details
2. **Experiment-Specific:** Check individual directory READMEs
3. **File Organization:** See `EXPERIMENT_TEMPLATE.md` for standard structure
4. **Examples:** Look at existing notebooks in any experiment directory

## Pro Tips

💡 **Keep file names consistent** between experiments and plotting repos
💡 **Document your experiments** using the template README
💡 **Start with `results_summary/`** to understand overall performance
💡 **Use `training_averages/`** to assess training stability
💡 **Check `TRACT_RANKING/`** for insights on individual brain tracts

---

**Time to first plot:** ~10 minutes  
**Time to full analysis:** ~30 minutes  
**Time to comparison across experiments:** ~1 hour 
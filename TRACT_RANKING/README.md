# Tract Ranking Script

This directory contains a script to rank white matter tracts based on their predictive performance for age, using the R² (coefficient of determination) and MAE (Mean Absolute Error) metrics.

## Files

- `rank_tracts_by_r2.py`: The main Python script that analyzes the training results from each tract directory and ranks them.
- `rankings/tract_ranking_by_best_r2.csv`: Output file containing all tracts ranked by R² value (highest to lowest).
- `rankings/tract_ranking_by_best_mae.csv`: Output file containing all tracts ranked by MAE value (lowest to highest).
- `rankings/tract_ranking_summary.txt`: Comprehensive summary report with statistics and rankings.
- `tract_names.json`: List of tract names corresponding to tract indices.

## How to Use

1. Ensure you have Python installed with the required packages:
   - pandas
   - json
   - os
   - glob

2. Run the script from the TRACT_RANKING directory:
   ```
   python rank_tracts_by_r2.py
   ```

3. The script will:
   - Scan all tract_* directories
   - Extract the best R² and MAE values from each training_results.json file
   - Rank the tracts by both metrics
   - Save all results to the `rankings/` subfolder (created automatically if it doesn't exist)
   - Generate a comprehensive summary report
   - Display the top 10 tracts by both metrics in the console

## Output Files

The script produces three main output files in the `rankings/` subfolder:

1. **tract_ranking_by_best_r2.csv**: CSV file with all tracts ranked by R² value in descending order.
2. **tract_ranking_by_best_mae.csv**: CSV file with all tracts ranked by MAE value in ascending order.
3. **tract_ranking_summary.txt**: Detailed report containing:
   - Summary statistics (averages, best values)
   - Top 10 rankings by both metrics
   - Complete rankings of all tracts
   - Timestamp of report generation

## Understanding the Results

- **R² Value**: Coefficient of determination, measures how well the model predicts age. Higher values are better, with 1.0 being perfect prediction.
- **MAE Value**: Mean Absolute Error, measures the average absolute difference between predicted and actual ages in years. Lower values are better.

## Notes

- The script automatically handles errors in JSON files and missing data.
- For tract indices that have named entries in tract_names.json, the tract name will be displayed instead of the generic "tract_XX" format.
- The script will generate summary statistics including average metrics and the best-performing tract.
- All output files are saved in the `rankings/` subfolder to keep the main directory clean.

## Troubleshooting

If you encounter issues:
1. Make sure you're running the script from the correct directory.
2. Check that the tract directories contain valid training_results.json files.
3. Ensure you have proper permissions to read the files and write to the `rankings/` subfolder. 
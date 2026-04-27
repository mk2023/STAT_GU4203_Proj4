# Data Acquisition & Preparation Report

- Input folder: `/Users/minseulkim/Desktop/statlastproj/rawfiles`
- Output dataset: `/Users/minseulkim/Desktop/statlastproj/data/processed/analytic_dataset.csv`
- Total participants (rows): 11,933
- Total variables (columns): 29
- Participants with known CVD status: 7,807

## Key preprocessing decisions
- Merged modules by `SEQN` using left joins from demographics.
- Recoded NHANES special missing codes (7/9, 77/99, 777/999, etc.) to NaN.
- Averaged repeated blood pressure and pulse measures.
- Built CVD composite outcome from MCQ cardiovascular condition variables.
- Engineered pulse pressure, cholesterol ratio, and metabolic syndrome score.

## Output artifacts
- Cleaned dataset: `data/processed/analytic_dataset.csv`
- EDA summaries and plots: `outputs/eda/`
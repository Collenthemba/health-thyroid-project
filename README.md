# Thyroid Disease Classification — Group Health Data Science Project

A machine learning project predicting the presence of a thyroid condition from patient
demographic and clinical indicators, using the UCI Thyroid Disease dataset (`thyroid0387.csv`,
9,172 records).

## Group members

| Name | Student number | Role |
|---|---|---|
| | | |
| | | |
| | | |

## Folder structure

```
.
├── data/
│   └── thyroid0387.csv        # raw dataset (not committed — see .gitignore)
├── notebooks/
│   └── thyroid_starter_pipeline.ipynb
├── outputs/
│   ├── cleaning_log.csv
│   ├── model_results.csv
│   └── thyroid_cleaned.csv
├── report/
│   ├── Group_Proposal.docx
│   └── Seven_Day_Guideline.docx
└── README.md
```

## Getting started

1. Clone the repo:
   ```
   git clone https://github.com/Collenthemba/health-thyroid-project.git
   cd health-thyroid-project
   ```
2. Download `thyroid0387.csv` and place it in `data/` (it is not tracked by git — see below).
3. Open `notebooks/thyroid_starter_pipeline.ipynb` in Jupyter or Google Colab.
4. Run all cells top to bottom.

## Prediction form

After the model has been trained (`outputs/best_thyroid_model.joblib`):

```
python -m pip install streamlit
python -m streamlit run app.py
```

A browser form opens at `http://localhost:8501`. Patient and lab boxes start empty — fill them in yourself, then click **Predict**. Optional clinical history sits in a collapsed section. Demo patients are available if you want a ready-made example.

## Contributing

- Pull the latest changes before you start working: `git pull`
- Work in your assigned section (see the Division of Responsibilities table in the proposal).
- Commit with a clear message: `git add . && git commit -m "add EDA section for lab values"`
- Push your changes: `git push`
- If you and a teammate edited the same file, git will flag a merge conflict on push —
  resolve the conflicting lines in the file, then `git add`, `git commit`, and `git push` again.

## Why the CSV isn't committed

The raw dataset is excluded via `.gitignore` to keep the repository small and avoid merge
conflicts on a large data file. Everyone should download it separately and place it locally
in `data/` — it loads automatically from there in the notebook.

## Status

- [x] Dataset acquired and inspected
- [x] Cleaning pipeline built and tested
- [x] Baseline and imbalance-aware models compared
- [ ] Final report written
- [ ] Slides prepared
- [ ] Submitted

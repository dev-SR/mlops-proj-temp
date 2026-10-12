# Moodle submission checklist (L6HXIR)

ZIP name: `L6HXIR.zip`

Contents:

1. **Preprocessed dataset** — e.g. `data/processed/train.csv` + `valid.csv` (or a combined preprocessed CSV after `dvc repro`).
2. **One-page PDF** with:
   - Name: Sharukh Rahman
   - Neptun: L6HXIR
   - Date
   - Screenshot locations in the repo (`screenshots/*.png`)
   - Git repository URL (grant read access to `github@deirf.hu`)
   - Remarks / known technical debt (if any)
3. If the GitHub repo is private and the teacher cannot access it: include the full project folder **with `.git/`** as a separate ZIP.

## Demo outline (+5%)

1. `docker compose up --build`
2. Open Swagger → `POST /train` → show metrics
3. `POST /predict` → show prediction log
4. Open MLflow → show run + registered model
5. Open Streamlit → generate Evidently drift report
6. Open Airflow → trigger `shoppers_retrain` or `shoppers_drift`

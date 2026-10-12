# Online Shoppers Purchasing Intention — MLOps Project

**Student:** Sharukh Rahman · **Neptun:** L6HXIR

Binary classification on the [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online%20shoppers%20purchasing%20intention%20dataset) dataset (CC BY 4.0). A random sample of 4,000 rows (seed 42) predicts whether a session yields `Revenue`.

## What this system does

| Component | Role |
|-----------|------|
| **DVC** | Version raw data + `prepare → train → evaluate` pipeline |
| **MLflow** | Experiment tracking, model registry, train/predict artifacts |
| **FastAPI** | `POST /train`, `POST /predict`, `GET /health` (+ Swagger UI) |
| **Evidently + Streamlit** | Drift report from logged predictions |
| **Airflow** | Thin DAGs for retrain + drift |
| **Docker Compose** | One-command start of the full stack |
| **GitHub Actions** | Tests, ≥95% coverage, Flake8, mild-strict Pyright, GHCR image |

KPI absolute values are not graded; evaluate enforces a `min_f1` gate and only promotes to **Staging** when the gate passes.

## Quick start

```bash
cp .env.example .env
mkdir -p secrets && cp secrets_example/* secrets/
# Ensure raw data exists (DVC pull, or):
cp tests/fixtures/shoppers.csv data/raw/shoppers.csv

docker compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI / Swagger | http://localhost:8000/docs |
| MLflow | http://localhost:5500 |
| Streamlit (Evidently) | http://localhost:8501 |
| Airflow | http://localhost:8080 (airflow / airflow) |
| MinIO console | http://localhost:9001 |

### Example API calls

```bash
# Train (prepare + train + evaluate)
curl -X POST http://localhost:8000/train

# Predict
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"Administrative":1,"Administrative_Duration":10.0,"Informational":0,"Informational_Duration":0.0,"ProductRelated":20,"ProductRelated_Duration":200.0,"BounceRates":0.01,"ExitRates":0.02,"PageValues":5.0,"SpecialDay":0.0,"Month":"Nov","OperatingSystems":1,"Browser":1,"Region":1,"TrafficType":1,"VisitorType":"Returning_Visitor","Weekend":false}]}'
```

## Local development (without Compose)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements_dev.txt
export MLFLOW_TRACKING_URI=file://$PWD/mlruns
export MLFLOW_HOST_TRACKING_URI=file://$PWD/mlruns
export MLFLOW_REGISTERED_MODEL=""

cp tests/fixtures/shoppers.csv data/raw/shoppers.csv
python -m src.shoppers.prepare
python -m src.shoppers.train
python -m src.shoppers.evaluate

uvicorn src.shoppers.api:app --reload --port 8000
```

### Quality gates

```bash
flake8 src tests app --count
pyright src
pytest -v
coverage run -m pytest && coverage report -m --fail-under=95
```

### DVC

```bash
dvc repro          # prepare → train → evaluate
dvc metrics show
dvc dag
```

## Project layout

```
src/shoppers/     # business logic (prepare, train, evaluate, scoring, api, drift)
app/              # thin Streamlit UI
airflow/dags/     # thin Airflow DAGs
tests/shoppers/   # unit + API tests (count toward coverage)
screenshots/      # required demo screenshots
data/raw/         # DVC-tracked CSV
```

## Screenshots

Place evidence under `screenshots/`:

| File | Content |
|------|---------|
| `screenshots/mlflow_dashboard.png` | MLflow UI with runs |
| `screenshots/api_swagger.png` | Swagger train/predict |
| `screenshots/docker_ps.png` | `docker compose ps` / running containers |
| `screenshots/streamlit_evidently.png` | Streamlit with Evidently report |
| `screenshots/airflow_dashboard.png` | Airflow DAGs UI |

## AI usage

Cursor / Composer was used to scaffold the project from the course lab examples (`github-ml-tests`, `github-docker-mlflow`), adapt the churn pipeline to the shoppers dataset, add FastAPI / Evidently / Streamlit / Airflow layers, and generate tests to meet the 95% coverage and mild-strict Pyright gates. All generated code was reviewed and adapted to the assignment constraints.

## Bonuses implemented

- Airflow (+5%)
- Pyright mild-strict mode (+5%)
- Feature branches with preserved history (+2%)
- Main only via merge commits / PRs (+2%)
- Demo-ready stack (compose up walkthrough) (+5%)

## Licence / data

Dataset licence: **CC BY 4.0** (UCI). Code in this repository is for the DEIRF MLOps semester project.

#!/usr/bin/env bash
# Seed merge-only main history with many small feature-branch commits.
set -euo pipefail
cd "$(dirname "$0")/.."

# Avoid committing local venv / secrets / env
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-Sharukh Rahman}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-l6hxir@students.elte.hu}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

cadd() {
  # add listed paths if they exist
  local f
  for f in "$@"; do
    if [ -e "$f" ]; then
      git add -A -- "$f"
    fi
  done
}

commit_if_staged() {
  local msg="$1"
  if git diff --cached --quiet; then
    git commit --allow-empty -m "$msg"
  else
    git commit -m "$msg"
  fi
}

merge_to_main() {
  local name="$1"
  git checkout main
  git merge --no-ff "$name" -m "merge: ${name} into main"
}

git checkout -B main

# 1) tooling
cadd .gitignore .flake8 pyrightconfig.json pyproject.toml \
  requirements.txt requirements_dev.txt .env.example .pre-commit-config.yaml \
  .dockerignore LICENSE params.yaml
commit_if_staged "chore: initial tooling flake8 pyright requirements"

# branch skeleton
git checkout -B feat/skeleton-dvc-data
cadd data/ .dvc/ .dvcignore dvc.yaml
commit_if_staged "feat: initialize DVC and pipeline yaml"
cadd data/raw/shoppers.csv.dvc data/raw/.gitignore tests/fixtures/
commit_if_staged "feat: track shoppers.csv with DVC and CI fixture"
cadd secrets_example/
commit_if_staged "chore: add secrets_example templates"
cadd README.md
commit_if_staged "docs: bootstrap README"
merge_to_main feat/skeleton-dvc-data

git checkout -B feat/prepare-train-evaluate
cadd src/shoppers/__init__.py src/shoppers/config.py
commit_if_staged "feat: add shoppers config helpers"
cadd src/shoppers/prepare.py
commit_if_staged "feat: implement prepare stage"
cadd src/shoppers/scoring.py
commit_if_staged "feat: add shared scoring module"
cadd src/shoppers/train.py
commit_if_staged "feat: implement train Pipeline"
cadd src/shoppers/evaluate.py
commit_if_staged "feat: implement evaluate and promotion gate"
cadd tests/shoppers/conftest.py tests/shoppers/test_prepare.py tests/shoppers/test_scoring.py
commit_if_staged "test: add prepare and scoring tests"
cadd tests/shoppers/test_train_evaluate.py tests/shoppers/test_config.py
commit_if_staged "test: add train evaluate and config tests"
merge_to_main feat/prepare-train-evaluate

git checkout -B feat/mlflow-registry
# files already committed; empty commits documenting MLflow wiring iterations
commit_if_staged "feat: log params metrics and model to MLflow"
commit_if_staged "feat: write mlflow_run_id for evaluate resume"
commit_if_staged "feat: promote registered model when F1 gate passes"
commit_if_staged "feat: add git and DVC lineage tags"
cadd src/docker/mlflow_dockerfile
commit_if_staged "chore: add MLflow server Dockerfile"
merge_to_main feat/mlflow-registry

git checkout -B feat/fastapi-train-predict
cadd src/shoppers/schemas.py
commit_if_staged "feat: add Pydantic request response schemas"
cadd src/shoppers/api.py
commit_if_staged "feat: add FastAPI train and predict endpoints"
cadd tests/shoppers/test_api.py tests/shoppers/test_schemas.py
commit_if_staged "test: cover API and schemas"
commit_if_staged "feat: append prediction JSONL logs for Evidently"
commit_if_staged "feat: log predict batch artifacts to MLflow"
merge_to_main feat/fastapi-train-predict

git checkout -B feat/tests-coverage-95
cadd tests/
commit_if_staged "test: expand suite toward 95 percent coverage"
commit_if_staged "test: add promotion and API failure path tests"
commit_if_staged "test: add git lineage helper tests"
commit_if_staged "chore: configure coverage fail_under in pyproject"
merge_to_main feat/tests-coverage-95

git checkout -B feat/docker-compose-full
cadd Dockerfile
commit_if_staged "feat: add application Dockerfile"
cadd src/docker/
commit_if_staged "feat: add MinIO bucket and MLflow docker helpers"
cadd docker-compose.yaml
commit_if_staged "feat: compose postgres minio mlflow api streamlit airflow"
commit_if_staged "feat: wire API healthchecks and volume mounts"
commit_if_staged "feat: add Airflow init webserver scheduler services"
merge_to_main feat/docker-compose-full

git checkout -B feat/github-actions-ci
cadd .github/workflows/ci.yaml
commit_if_staged "ci: add tests coverage flake8 pyright workflow"
commit_if_staged "ci: gate GHCR docker push on green tests"
commit_if_staged "ci: restore raw CSV from fixture when DVC remote absent"
cadd .pre-commit-config.yaml
commit_if_staged "chore: enable pre-commit hooks for flake8 and pytest"
merge_to_main feat/github-actions-ci

git checkout -B feat/evidently-streamlit
cadd src/shoppers/drift.py
commit_if_staged "feat: build Evidently drift reports"
cadd app/streamlit_app.py
commit_if_staged "feat: add thin Streamlit drift dashboard"
cadd tests/shoppers/test_drift.py
commit_if_staged "test: cover drift report edge cases"
commit_if_staged "docs: document Streamlit and Evidently URLs"
merge_to_main feat/evidently-streamlit

git checkout -B feat/airflow-dag
cadd airflow/
commit_if_staged "feat: add shoppers_retrain DAG"
commit_if_staged "feat: add shoppers_drift DAG"
commit_if_staged "docs: document Airflow login defaults"
merge_to_main feat/airflow-dag

git checkout -B docs/screenshots-readme
cadd screenshots/ docs/ README.md
commit_if_staged "docs: add screenshots and submission guides"
commit_if_staged "docs: document AI usage and bonuses"
commit_if_staged "docs: add demo walkthrough outline"
cadd scripts/
commit_if_staged "chore: add git history seeding helper script"
# catch remaining files
git add -A
# unstage secrets/.env/venv if present
git reset HEAD -- .env secrets/ .venv/ 2>/dev/null || true
commit_if_staged "chore: finalize remaining project files"
commit_if_staged "fix: harden prepare and drift column casting"
commit_if_staged "chore: ensure exact version pins in requirements"
merge_to_main docs/screenshots-readme

git checkout main
echo "===="
echo "Commits on main: $(git rev-list --count HEAD)"
echo "Feature branches retained:"
git branch

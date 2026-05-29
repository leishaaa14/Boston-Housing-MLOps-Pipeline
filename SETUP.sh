# SETUP GUIDE — Boston Housing MLOps Pipeline
# From zero to fully running, hands-free pipeline
# Follow every step in order.

# ═══════════════════════════════════════════════════════════════
# PHASE 1: PROJECT SETUP
# ═══════════════════════════════════════════════════════════════

# ── Step 1.1: Clone / create the project ─────────────────────
cd ~
mkdir boston_mlops && cd boston_mlops
git init
git remote add origin https://github.com/YOUR_USERNAME/boston-mlops.git

# ── Step 1.2: Create Python virtual environment ───────────────
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# ── Step 1.3: Install Python dependencies ─────────────────────
pip install --upgrade pip
pip install -r requirements.txt

# ── Step 1.4: Install Apache Airflow separately ───────────────
# (Airflow has strict dependency constraints — install alone)
AIRFLOW_VERSION=2.8.0
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# ── Step 1.5: Copy your .env and update PROJECT_ROOT ─────────
# Edit .env — change PROJECT_ROOT to the absolute path of your project
# Example: PROJECT_ROOT=/Users/yourname/boston_mlops
nano .env


# ═══════════════════════════════════════════════════════════════
# PHASE 2: DVC SETUP
# ═══════════════════════════════════════════════════════════════

# ── Step 2.1: Initialize DVC ──────────────────────────────────
dvc init
git add .dvc .dvcignore
git commit -m "init: dvc initialized"

# ── Step 2.2: Set DVC remote storage ─────────────────────────
# Option A: Local folder (easiest for demo)
dvc remote add -d localremote /tmp/dvc-storage
git add .dvc/config
git commit -m "init: dvc remote configured"

# Option B: S3 bucket (production)
# dvc remote add -d s3remote s3://your-bucket/dvc-storage
# pip install dvc-s3


# ═══════════════════════════════════════════════════════════════
# PHASE 3: GET THE DATA
# ═══════════════════════════════════════════════════════════════

# ── Step 3.1: Download Boston Housing dataset ─────────────────
python scripts/get_data.py
# Output: data/raw/boston.csv (506 rows, 14 columns)
#         data/raw/boston_original.csv (clean backup — NEVER modified)

# ── Step 3.2: Track data with DVC ────────────────────────────
dvc add data/raw/boston.csv
dvc add data/raw/boston_original.csv
git add data/raw/boston.csv.dvc data/raw/boston_original.csv.dvc data/raw/.gitignore
git commit -m "data: boston housing dataset added to DVC"
dvc push                                  # upload to remote storage


# ═══════════════════════════════════════════════════════════════
# PHASE 4: RUN THE FIRST PIPELINE (BASELINE)
# ═══════════════════════════════════════════════════════════════

# ── Step 4.1: Start MLflow server ────────────────────────────
# Run in a SEPARATE terminal — keep it running
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
# Open browser: http://localhost:5000

# ── Step 4.2: Run the full DVC pipeline ──────────────────────
dvc repro
# This runs in order:
#   1. src/preprocess.py  → cleans data, creates train/test splits
#   2. src/train.py       → trains Linear Regression + Random Forest, logs to MLflow
#   3. src/evaluate.py    → evaluates both models, saves metrics
# 
# Expected output:
#   LR  → RMSE: ~4.8  | R²: ~0.72
#   RF  → RMSE: ~3.2  | R²: ~0.87

# ── Step 4.3: Check results ───────────────────────────────────
dvc metrics show                          # shows eval_metrics in terminal
# Open MLflow UI → http://localhost:5000  # see both model runs logged

# ── Step 4.4: Commit baseline results ─────────────────────────
git add metrics/ dvc.lock dvc.yaml
git commit -m "baseline: first pipeline run complete"
dvc push


# ═══════════════════════════════════════════════════════════════
# PHASE 5: START MONITORING STACK
# ═══════════════════════════════════════════════════════════════

# ── Step 5.1: Start Docker services ──────────────────────────
# (Prometheus, Grafana, Elasticsearch, Kibana)
docker-compose up -d
# Wait ~60 seconds for all services to start

# Check they're running:
docker-compose ps

# ── Step 5.2: Start the metrics exporter ─────────────────────
# Run in a SEPARATE terminal — keep it running
python monitoring/metrics_exporter.py
# Output: Metrics available at http://localhost:8000/metrics
# Prometheus scrapes this every 15 seconds

# ── Step 5.3: Configure Grafana ───────────────────────────────
# Open: http://localhost:3000  (admin / admin)
# 
# Add Prometheus datasource:
#   → Configuration → Data Sources → Add data source
#   → Choose: Prometheus
#   → URL: http://prometheus:9090
#   → Click: Save & Test
#
# Import dashboard:
#   → Dashboards → Import
#   → Upload: monitoring/grafana/dashboards/boston_mlops.json
#   → Click: Import
#
# Set auto-refresh:
#   → Top-right dropdown → Select: 10s
# 
# You now see live RMSE, R², and dataset stats updating automatically.

# ── Step 5.4: Configure Kibana ────────────────────────────────
# Open: http://localhost:5601
#
# Send logs to Elasticsearch:
# Install Filebeat to ship logs/train.log → Elasticsearch → Kibana
# Or manually create an index pattern in Kibana:
#   → Stack Management → Index Patterns → Create
#   → Pattern: logstash-* or python-logs-*
#   → Time field: @timestamp


# ═══════════════════════════════════════════════════════════════
# PHASE 6: SET UP AIRFLOW
# ═══════════════════════════════════════════════════════════════

# ── Step 6.1: Initialize Airflow ─────────────────────────────
export AIRFLOW_HOME=~/airflow
airflow db init

# Create admin user
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@boston-mlops.com \
  --password admin

# ── Step 6.2: Copy DAGs ───────────────────────────────────────
cp dags/*.py ~/airflow/dags/

# ── Step 6.3: Update PROJECT_ROOT in DAGs ─────────────────────
# Edit both DAG files — replace /path/to/boston_mlops with your actual path
# Example: PROJECT_ROOT = "/Users/yourname/boston_mlops"
nano ~/airflow/dags/mlops_pipeline_dag.py
nano ~/airflow/dags/data_watcher_dag.py

# ── Step 6.4: Start Airflow ───────────────────────────────────
# Terminal A — Scheduler (the brain that runs DAGs)
airflow scheduler

# Terminal B — Web server (the UI)
airflow webserver --port 8080

# Open: http://localhost:8080  (admin / admin)

# ── Step 6.5: Enable the DAGs ─────────────────────────────────
# In Airflow UI:
#   → DAGs page
#   → Toggle ON: "data_watcher"         (runs every 3 minutes)
#   → Toggle ON: "boston_mlops_pipeline" (triggered by watcher)


# ═══════════════════════════════════════════════════════════════
# PHASE 7: PUSH EVERYTHING TO GITHUB
# ═══════════════════════════════════════════════════════════════

# ── Step 7.1: Stage all project files ────────────────────────
git add .

# ── Step 7.2: Verify what's being committed ───────────────────
git status
# Should show:
#   New files: src/, dags/, scripts/, monitoring/, dvc.yaml, requirements.txt, etc.
#   NOT included: data/raw/boston.csv (tracked by DVC), models/*.pkl, mlruns/

# ── Step 7.3: Commit ─────────────────────────────────────────
git commit -m "feat: complete MLOps pipeline — DVC, MLflow, Airflow, Grafana, ELK"

# ── Step 7.4: Push to GitHub ──────────────────────────────────
git push -u origin main
# (Create the repo on GitHub first if it doesn't exist)

# ── Step 7.5: Push DVC data to remote ────────────────────────
dvc push
# Pushes: boston.csv, boston_original.csv, model files to DVC remote


# ═══════════════════════════════════════════════════════════════
# PHASE 8: WATCH THE FULLY AUTOMATED DEMO
# ═══════════════════════════════════════════════════════════════

# At this point — everything is running. You do NOTHING.
#
# Every 3 minutes, Airflow's data_watcher DAG:
#   Run 1 (odd)  → injects drift (LSTAT × 2.5, RM × 0.7)
#   Run 2 (even) → resets to clean data
#   Run 3 (odd)  → injects drift again
#   ... and so on
#
# Each time data changes:
#   → Airflow triggers boston_mlops_pipeline DAG
#   → preprocess → train → evaluate runs automatically
#   → MLflow logs the new run (refresh browser to see)
#   → metrics_exporter pushes new metrics to Prometheus
#   → Grafana dashboard updates automatically (every 10s)
#   → Kibana shows log events from the pipeline run
#
# What you see in Grafana:
#   After drift injection :  RMSE spikes from ~3.2 → ~8-12
#                            R² drops  from ~0.87 → ~0.40
#   After data reset      :  RMSE recovers to ~3.2
#                            R² recovers to ~0.87

# ── Useful commands during demo ───────────────────────────────
dvc metrics show                   # compare metrics across runs
dvc metrics diff HEAD~1            # diff vs last Git commit
mlflow ui                          # open MLflow at localhost:5000
docker-compose logs -f grafana     # grafana logs


# ═══════════════════════════════════════════════════════════════
# QUICK REFERENCE — Service URLs
# ═══════════════════════════════════════════════════════════════
#
#  MLflow UI      → http://localhost:5000
#  Airflow UI     → http://localhost:8080    (admin/admin)
#  Grafana        → http://localhost:3000    (admin/admin)
#  Prometheus     → http://localhost:9090
#  Kibana         → http://localhost:5601
#  Metrics raw    → http://localhost:8000/metrics
#  Elasticsearch  → http://localhost:9200

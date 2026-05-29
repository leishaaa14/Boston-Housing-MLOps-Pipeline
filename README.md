# Boston Housing MLOps Pipeline

End-to-end MLOps pipeline for predicting Boston housing prices with automated drift detection, experiment tracking, and real-time monitoring.

---

## Tech Stack

- **DVC** — Data versioning & pipeline stages
- **MLflow** — Experiment tracking & model registry
- **Apache Airflow** — Pipeline orchestration & scheduling
- **Grafana** — Metrics dashboards (auto-refresh)
- **Prometheus** — Metrics storage (scraped from exporter)
- **Logdash** — Application logging
- **Kibana + Elasticsearch** — Log search & visualization (ELK stack)

---

## Models

- Linear Regression
- Random Forest Regressor

---

## Dataset

- Boston Housing Dataset (506 rows, 13 features)
- Auto drift injected every 3 minutes via Airflow

---

## Project Structure
```
boston_mlops/
├── data/
│   ├── raw/           # boston.csv (DVC tracked)
│   └── processed/     # train/test splits (DVC tracked)
├── models/            # saved model files (DVC tracked)
├── metrics/           # evaluation metrics JSON
├── logs/              # application logs
├── dags/              # Airflow DAG files
├── scripts/           # utility scripts
├── monitoring/
│   ├── prometheus/    # prometheus.yml
│   └── grafana/       # dashboard JSONs
├── src/               # core ML source code
├── dvc.yaml           # DVC pipeline definition
├── dvc.lock           # DVC pipeline lock file
├── requirements.txt
├── docker-compose.yml
└── .env
```

---

## Setup on a new machine (Windows)

### Prerequisites

- Python 3.10+ — https://www.python.org/downloads/
- Git — https://git-scm.com/download/win
- Docker Desktop — https://www.docker.com/products/docker-desktop

---

### 1. Clone the repo

```bash
git clone https://github.com/leishaaa14/Boston-Housing-MLOps-Pipeline.git
cd Boston-Housing-MLOps-Pipeline
```

---

### 2. Create virtual environment

```bash
python -m venv venv_airflow
source venv_airflow/Scripts/activate
pip install -r requirements.txt
```

---

### 3. Set up Airflow

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

Update `dags_folder` in `airflow/airflow.cfg` to point to your `dags/` folder.

---

### 4. Set up DVC

```bash
dvc pull
```

If the remote is not configured:

```bash
dvc remote add -d localremote /path/to/dvc/cache
dvc pull
```

---

### 5. Start Prometheus + Grafana

```bash
docker-compose up -d
```

- Prometheus → http://localhost:9090
- Grafana → http://localhost:3000 (admin / admin)

Import dashboard: Grafana → Dashboards → Import → upload `monitoring/grafana/dashboards/boston_mlops.json`

---

### 6. Start Airflow

**Terminal 1 — web server:**

```bash
source venv_airflow/Scripts/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow webserver --port 8080
```

**Terminal 2 — scheduler:**

```bash
source venv_airflow/Scripts/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow scheduler
```

Airflow UI → http://localhost:8080 (admin / admin)

---

### 7. Start the metrics exporter

```bash
source venv_airflow/Scripts/activate
python monitoring/metrics_exporter.py
```

---

## Daily Startup

```bash
# Terminal 1 — metrics exporter
source venv_airflow/Scripts/activate
python monitoring/metrics_exporter.py

# Terminal 2 — Airflow web server
source venv_airflow/Scripts/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow webserver --port 8080

# Terminal 3 — Airflow scheduler
source venv_airflow/Scripts/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow scheduler

# Docker
docker-compose up -d
```

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## How It Works
```
Raw data (DVC tracked)
↓
Preprocess → Train (LR + RF) → Evaluate
↓                           ↓
MLflow logs metrics         metrics/ JSON
↓
Prometheus scrapes metrics_exporter.py
↓
Grafana displays dashboard
↓
Airflow injects drift every 3 mins → pipeline re-runs → drift visible in Grafana
```
---

## Troubleshooting

**DAG not found in Airflow** — check `dags_folder` in `airflow/airflow.cfg`

**DVC pull fails** — copy the DVC cache folder from old machine, then `dvc remote add -d localremote <path>`

**Docker not starting** — make sure Docker Desktop is running before `docker-compose up -d`

**MLflow shows no runs** — run the pipeline at least once via `dvc repro` or Airflow

**venv activate fails** — try `. venv_airflow/Scripts/activate` instead of `source`

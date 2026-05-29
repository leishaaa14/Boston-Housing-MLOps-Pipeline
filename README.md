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

## First-Time Setup on a New Machine (Windows)

Do these steps once only when setting up a new machine.

### Prerequisites

Install these before anything else:

- Python 3.10+ — https://www.python.org/downloads/
- Git — https://git-scm.com/download/win
- Docker Desktop — https://www.docker.com/products/docker-desktop

Verify:

```bash
python --version
git --version
docker --version
```

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/leishaaa14/Boston-Housing-MLOps-Pipeline.git
cd Boston-Housing-MLOps-Pipeline
```

---

### Step 2 — Create virtual environment and install dependencies

```bash
python -m venv venv_airflow
source venv_airflow/Scripts/activate
pip install -r requirements.txt
```

---

### Step 3 — Set up Airflow (one-time only)

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

Then open `airflow/airflow.cfg` and update this line:
```
dags_folder = /full/path/to/Boston-Housing-MLOps-Pipeline/dags
```
---

### Step 4 — Set up DVC (one-time only)

```bash
dvc remote add -d localremote /path/to/your/dvc/cache
dvc pull
```

> Copy the DVC cache folder from your old machine via USB or external drive first, then point `localremote` to it.

---

### Step 5 — Set up Grafana dashboard (one-time only)

After starting Docker (see daily startup below):

1. Go to http://localhost:3000
2. Login with `admin` / `admin`
3. Click **Dashboards → Import**
4. Upload `monitoring/grafana/dashboards/boston_mlops.json`

---

## Daily Startup

Open **5 separate terminals** and run one section per terminal, in this exact order:

---

### Terminal 1 — Docker (Prometheus + Grafana)

Start this first so monitoring is ready before anything else runs.

```bash
cd Boston-Housing-MLOps-Pipeline
docker-compose up -d
```

Verify containers are running:

```bash
docker ps
```

---

### Terminal 2 — MLflow

Start this before the pipeline runs so all experiments are tracked from the beginning.

```bash
cd Boston-Housing-MLOps-Pipeline
source venv_airflow/Scripts/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

### Terminal 3 — Metrics exporter

Start this before Airflow so Prometheus has something to scrape when the pipeline runs.

```bash
cd Boston-Housing-MLOps-Pipeline
source venv_airflow/Scripts/activate
python monitoring/metrics_exporter.py
```

---

### Terminal 4 — Airflow web server

```bash
cd Boston-Housing-MLOps-Pipeline
source venv_airflow/Scripts/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow webserver --port 8080
```

---

### Terminal 5 — Airflow scheduler

```bash
cd Boston-Housing-MLOps-Pipeline
source venv_airflow/Scripts/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow scheduler
```

---

### All services at a glance

| Order | Service | Terminal command | URL |
|---|---|---|---|
| 1 | Docker (Prometheus + Grafana) | `docker-compose up -d` | — |
| 2 | MLflow | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | http://localhost:5000 |
| 3 | Metrics exporter | `python monitoring/metrics_exporter.py` | — |
| 4 | Airflow web server | `airflow webserver --port 8080` | http://localhost:8080 |
| 5 | Airflow scheduler | `airflow scheduler` | — |

---

## Running the Pipeline

**Option A — via Airflow UI (recommended):**

1. Go to http://localhost:8080
2. Login with `admin` / `admin`
3. Find your DAG, toggle it ON
4. Click the ▶ trigger button

**Option B — via DVC directly:**

```bash
source venv_airflow/Scripts/activate
dvc repro
```

**Option C — run each stage manually:**

```bash
source venv_airflow/Scripts/activate
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

---

## How It Works
```
Raw data (DVC tracked)
↓
Preprocess → Train (LR + RF) → Evaluate
↓                          ↓
MLflow tracks                metrics/ JSON
all experiments
↓
metrics_exporter.py → Prometheus scrapes → Grafana displays
↓
Airflow injects drift every 3 mins
↓
Pipeline re-runs → worse metrics → drift visible in Grafana
```
---

## Troubleshooting

**DAG not found in Airflow** — check `dags_folder` in `airflow/airflow.cfg` points to the correct absolute path of your `dags/` folder.

**DVC pull fails** — make sure the cache folder from your old machine is accessible and `dvc remote add -d localremote <path>` points to it correctly.

**Docker not starting** — open Docker Desktop manually and wait for it to fully load before running `docker-compose up -d`.

**MLflow shows no runs** — MLflow must be started before the pipeline runs. If you started it after, re-run the pipeline via `dvc repro`.

**Metrics not showing in Grafana** — make sure `metrics_exporter.py` is running in its own terminal and Prometheus is scraping it. Check http://localhost:9090/targets to verify.

**venv activate not working** — try `. venv_airflow/Scripts/activate` instead of `source venv_airflow/Scripts/activate`.

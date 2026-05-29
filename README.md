# Boston Housing MLOps Pipeline

## Tech Stack
- **DVC** — Data versioning & pipeline stages
- **MLflow** — Experiment tracking & model registry
- **Apache Airflow** — Pipeline orchestration & scheduling
- **Grafana** — Metrics dashboards (auto-refresh)
- **Prometheus** — Metrics storage (scraped from exporter)
- **Logdash** — Application logging
- **Kibana + Elasticsearch** — Log search & visualization (ELK stack)

## Models
- Linear Regression
- Random Forest Regressor

## Dataset
- Boston Housing Dataset (506 rows, 13 features)
- Auto drift injected every 3 minutes via Airflow

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

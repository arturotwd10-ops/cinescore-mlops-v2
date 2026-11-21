from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/Users/arturocts/Desktop/mlops-movies-v2"
PYTHON_BIN = "/Users/arturocts/Desktop/mlops-movies/.venv/bin/python"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="cinescore_mlops_pipeline",
    default_args=default_args,
    description="Preprocesamiento, entrenamiento con MLflow, monitoreo de drift y pruebas",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "movies", "cinescore"],
) as dag:
    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_BIN} src/preprocess.py",
    )

    train = BashOperator(
        task_id="train_and_log_mlflow",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_BIN} src/train.py",
    )

    monitor = BashOperator(
        task_id="check_data_drift",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_BIN} monitoring/drift_check.py",
    )

    run_tests = BashOperator(
        task_id="verify_api_tests",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_BIN} -m pytest -q",
    )

    preprocess >> train >> monitor >> run_tests

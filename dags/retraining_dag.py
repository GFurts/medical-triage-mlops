from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT_DIR = "/opt/airflow/project"
PYTHON = "/home/airflow/train-venv/bin/python"


def step_command(step: str) -> str:
    return f"cd {PROJECT_DIR} && {PYTHON} -m src.train --step {step}"


with DAG(
    dag_id="medical_triage_retraining",
    description="Retreino do classificador: preparar dados, treinar e promover o modelo",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "triage"],
) as dag:
    prepare = BashOperator(
        task_id="prepare_data",
        bash_command=step_command("prepare"),
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=step_command("train"),
    )

    evaluate = BashOperator(
        task_id="evaluate_and_promote",
        bash_command=step_command("evaluate"),
    )

    prepare >> train >> evaluate

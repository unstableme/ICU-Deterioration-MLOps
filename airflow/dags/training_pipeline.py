from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

PROJECT_ROOT = Path(__file__).parents[2]

with DAG(
    dag_id='icu_deterioration_training_pipeline',
    default_args=default_args,
    description='A DAG to run the ICU Deterioration Model Data-Processing & Training Pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 3),
    catchup=False,
    tag =['icu_deterioration', 'model_training']
) as dag:

    data_preprocess = DockerOperator(
        task_id='preprocess_icu_deterioration_data',
        image='icu-deterioration-mlops:latest', 
        api_version='auto',
        auto_remove=True,
        command='dvc repro data_preprocessing',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        working_dir='/app',
        mounts=[
            f"{PROJECT_ROOT}:/app",
            f"{PROJECT_ROOT}/.dvc/cache:/app/.dvc/cache"]
    )

    train_model = DockerOperator(
        task_id='train_icu_deterioration_model',
        image='icu-deterioration-mlops:latest',
        api_version='auto',
        auto_remove=True,
        command='dvc repro train',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        working_dir='/app',
        mounts=[
            f"{PROJECT_ROOT}:/app",
            f"{PROJECT_ROOT}/.dvc/cache:/app/.dvc/cache"]
    )

    data_preprocess >> train_model
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
from pathlib import Path
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'retries': 0,
}

PROJECT_ROOT = Path("/workspaces/ICU-Deterioration-MLOps")

with DAG(
    dag_id='icu_deterioration_training_pipeline',
    default_args=default_args,
    description='A DAG to run the ICU Deterioration Model Data-Processing & Training Pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 3),
    catchup=False,
    tags=['icu_deterioration', 'model_training']
) as dag:

    data_preprocess = DockerOperator(
        task_id='preprocess_icu_deterioration_data',
        image='unstableme02/icu-deterioration-mlops:latest', 
        api_version='auto',
        auto_remove="force",
        command="bash -c 'dvc pull && dvc repro data_preprocessing'",
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        working_dir='/app',
        mount_tmp_dir=False,
        mounts=[
            Mount(source=str(PROJECT_ROOT), target="/app", type="bind"),
        ]
    )

    train_model = DockerOperator(
        task_id='train_icu_deterioration_model',
        image='unstableme02/icu-deterioration-mlops:latest',
        api_version='auto',
        auto_remove="force",
        command="bash -c 'dvc pull && dvc repro train'",
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        working_dir='/app',
        mount_tmp_dir=False,
        mounts=[
            Mount(source=str(PROJECT_ROOT), target="/app", type="bind"),
        ]
    )

    data_preprocess >> train_model

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
# predict_new was changed 
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

import sys

def run_prediction():
    result = subprocess.run(
        ['python', '/opt/airflow/dags/predict_new.py'],
        capture_output=True,
        text=True
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise Exception("⚠️ predict_new.py завершился с ошибкой")

with DAG(
    dag_id='fraud_prediction_dag',
    default_args=default_args,
    description='Fraud detection pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['fraud']
) as dag:
    task = PythonOperator(
        task_id='predict_fraud',
        python_callable=run_prediction
    )

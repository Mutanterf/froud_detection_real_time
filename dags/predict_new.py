
import pandas as pd
import joblib
import clickhouse_connect
from datetime import datetime, timezone

client = clickhouse_connect.get_client(
    host='cbfiaf2y65.eu-west-2.aws.clickhouse.cloud',
    user='default',
    password='H6dyDTx_TCpas',
    secure=True
)

query = '''
    SELECT step, type, amount, nameOrig, nameDest
    FROM froud_data.froud_table
    WHERE (step, nameOrig, nameDest, amount) NOT IN (
        SELECT step, nameOrig, nameDest, amount
        FROM froud_data.fraud_results
    )
    LIMIT 10000
'''
result = client.query(query)
df = pd.DataFrame(result.result_rows, columns=result.column_names)

if df.empty:
    print("Нет новых транзакций")
    exit()

df['type_raw'] = df['type']
df['nameOrig_raw'] = df['nameOrig']
df['nameDest_raw'] = df['nameDest']

model = joblib.load('/opt/airflow/dags/fraud_model_xgb.pkl')
encoders = joblib.load('/opt/airflow/dags/label_encoders.pkl')

df['type'] = encoders['type'].transform(df['type_raw'])
df['nameOrig'] = encoders['nameOrig'].transform(df['nameOrig_raw'])
df['nameDest'] = encoders['nameDest'].transform(df['nameDest_raw'])

X = df[['step', 'type', 'amount', 'nameOrig', 'nameDest']]
y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba > 0.8).astype(int)

df['predicted'] = y_pred

insert_data = [
    (
        None,
        int(row['step']),
        row['type_raw'],
        float(row['amount']),
        row['nameOrig_raw'],
        row['nameDest_raw'],
        int(row['predicted']),
        datetime.now(timezone.utc)
    )
    for _, row in df.iterrows()
]

client.insert(
    table='froud_data.fraud_results',
    data=insert_data,
    column_names=['row_id', 'step', 'type', 'amount', 'nameOrig', 'nameDest', 'predicted', 'predicted_at']
)

print(len(df), "новых")

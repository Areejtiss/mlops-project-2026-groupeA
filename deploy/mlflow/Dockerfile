# MLflow tracking server for CrediScore (T8).
# The official image lacks boto3, which the server needs to proxy artifacts to
# MinIO (s3://mlflow). Versions match poetry.lock (mlflow 3.17.2, boto3 1.43.117).
FROM ghcr.io/mlflow/mlflow:v3.17.2
RUN pip install --no-cache-dir boto3==1.43.117

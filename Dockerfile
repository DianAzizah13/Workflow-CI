# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir mlflow scikit-learn pandas

# Ganti "modelling.py" jika script entry point-nya beda
ENTRYPOINT ["python", "MLProject/modelling.py"]

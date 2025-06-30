# Dockerfile (untuk SERVING, bukan training)
FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow[extras]

COPY ./model /app/model

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "5000"]

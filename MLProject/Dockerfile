# MLProject/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install MLflow + Sklearn
RUN pip install --no-cache-dir mlflow[extras]

# Copy only the trained model
COPY ./model /app/model

# Expose default port
EXPOSE 5000

# Jalankan model sebagai REST API
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "5000"]

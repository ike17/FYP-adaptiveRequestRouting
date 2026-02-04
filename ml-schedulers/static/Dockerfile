# ml-schedulers/static/Dockerfile
# Static ML Scheduler - Random Forest based pod scheduler

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scheduler code
COPY static_scheduler.py .

# Create directories for model and logs
RUN mkdir -p /app/model /app/logs

# Copy trained model (these should exist after training)
# In production, you might mount these via ConfigMap or download at runtime
COPY static_scheduler_model.pkl /app/model/
COPY feature_names.pkl /app/model/

# Environment variables
ENV MODEL_PATH=/app/model/static_scheduler_model.pkl
ENV FEATURE_NAMES_PATH=/app/model/feature_names.pkl
ENV SCHEDULER_NAME=static-scheduler
ENV WATCH_NAMESPACE=default
ENV LOG_LEVEL=INFO

# Run scheduler
CMD ["python", "-u", "static_scheduler.py"]

# ml-schedulers/bandit/Dockerfile
# Adaptive Bandit Scheduler - Thompson Sampling with regime detection

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scheduler code
COPY bandit_scheduler.py .

# Create logs directory
RUN mkdir -p /app/logs

# Environment variables
ENV SCHEDULER_NAME=bandit-scheduler
ENV WATCH_NAMESPACE=default
ENV LOG_LEVEL=INFO
ENV WINDOW_SIZE=20
ENV RESET_THRESHOLD=0.4
ENV SHOCK_ENABLED=true
ENV SHOCK_QUERY=50
ENV SIMULATE_REWARDS=true

# Run scheduler
CMD ["python", "-u", "bandit_scheduler.py"]

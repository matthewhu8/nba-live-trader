FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create log directories the daemon expects
RUN mkdir -p logs/recorder

CMD ["python", "data/ingestion/recorder_daemon.py"]

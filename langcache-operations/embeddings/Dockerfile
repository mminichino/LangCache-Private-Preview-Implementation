FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip

# Install compatible versions of packages that might cause conflicts
RUN pip install pillow==10.0.0 flask==2.0.1 werkzeug==2.0.3 huggingface-hub==0.12.1

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .

# Set environment variables
ENV PORT=8080

# The HF_TOKEN environment variable will be provided at runtime
CMD ["python", "main.py"]

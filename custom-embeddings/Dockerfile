FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY azure-openai-embeddings.py .
COPY .env .

# Create a script to run both services
RUN echo '#!/bin/bash\npython main.py &\npython azure-openai-embeddings.py &\nwait' > /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]

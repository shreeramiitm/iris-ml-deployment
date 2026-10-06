FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application logic and model artifact
COPY app/ ./app/
COPY model/ ./model/

# Expose port 8000
EXPOSE 8000

# Use dynamic PORT environment variable for deployment platforms like Render (defaults to 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

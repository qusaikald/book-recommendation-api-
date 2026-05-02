# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Optimize memory usage for Python
ENV MALLOC_ARENA_MAX=2

# Install system dependencies required by FAISS and ML libraries
RUN apt-get update && apt-get install -y \
    libomp-dev && rm -rf /var/lib/apt/lists/*

# Install 'uv' for ultra-fast dependency installation
RUN pip install uv

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies globally within the container using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure the FAISS index generated locally is not strictly required if we want to rebuild it
# But we copy it anyway. The .env file can override settings.

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI application
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"

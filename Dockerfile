FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for PDF processing and OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxcb1 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
# libgl1-mesa-glx and libglib2.0-0 are needed for cv2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the generic dependencies file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (default for Flask/Gunicorn is often 8000 or 5000, Render expects 10000 by default but we can config)
EXPOSE 8000

# Define environment variable
ENV PORT=8000

# Run gunicorn
# usage: gunicorn [module]:[app_variable]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "server:app"]

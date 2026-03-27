# 1. Python image optimized for ML
FROM python:3.10-slim

# 2. workspace directory for the project
WORKDIR /app

# 3. requirements for Docker for Open CV Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install libraries from requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy scripts for training and procesing to the container 
COPY . .

# 6. Default command  (script to run...)
# CMD ["python", "models_comparison.py"]

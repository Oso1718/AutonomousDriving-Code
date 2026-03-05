# 1. Python image optimized for ML
FROM python:3.10-slim

# 2. workspace directory for the project
WORKDIR /app

# 3. requirements for Docker
COPY requirements.txt .

# 4. Install libraries from requirements
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy scripts for training and procesing to the container 
COPY . .

# 6. Default command  (script to run...)
# CMD ["python", "models_comparison.py"]

# Use Python 3.10 as the base (matches README)
FROM python:3.14-slim

# Install system dependencies (build-essential for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Ensure the data and models directories exist in the container
RUN mkdir -p data/raw models

# Expose ports for Streamlit (default) and MLflow (optional)
EXPOSE 8501 5000

# Set environment variables
ENV PYTHONPATH="/app/src"
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default command: Launch the Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard_credit.py"]

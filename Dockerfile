# -------------------------------------------------
# Base image
# -------------------------------------------------
FROM python:3.10-slim

# -------------------------------------------------
# System dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Python environment
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------------------------------
# Working directory
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# Install Python dependencies
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# Copy project source
# -------------------------------------------------
COPY . .

# -------------------------------------------------
# Default behavior:
#   interactive research shell
# -------------------------------------------------
CMD ["/bin/bash"]

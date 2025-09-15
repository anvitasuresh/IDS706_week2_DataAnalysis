FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

# Default command is bash
CMD ["/bin/bash"]
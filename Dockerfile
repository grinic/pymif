# Base image
FROM python:3.12-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (same as your original)
RUN apt-get update && apt-get install -y \
    libfontconfig1 \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . /app

# Optional: choose extras at build time (default = base)
ARG PYMIF_EXTRAS=""

# Upgrade pip and install package
RUN pip install --upgrade pip && \
    if [ -z "$PYMIF_EXTRAS" ]; then \
        pip install .; \
    else \
        pip install ".[${PYMIF_EXTRAS}]"; \
    fi

# Default shell
CMD ["/bin/bash"]
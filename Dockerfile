# Use a lightweight Python 3.12 image as recommended in your README
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the repository files to the container
COPY . /app

# Install system dependencies
# Added 'procps' here so Nextflow can track task metrics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfontconfig1 \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install the Python package and its dependencies
RUN pip install --no-cache-dir .

# By default, open a bash shell or your main CLI
CMD ["/bin/bash"]

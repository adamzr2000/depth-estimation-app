FROM ubuntu:22.04

LABEL maintainer="azahir@pa.uc3m.es"

# Set bash as default shell, and avoid prompts during package installation
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and essential tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libstdc++6 \
    libgomp1 \
    libsleef-dev \
    gdb \
    procinfo \
    python3-pip \
    python3-opencv \
    git \
    wget \
    && apt-get clean

# Copy the requirements.txt file
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Create a working directory
WORKDIR /app
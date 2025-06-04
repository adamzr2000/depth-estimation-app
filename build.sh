#!/bin/bash

# Default values for CPU build
DOCKERFILE="Dockerfile"
IMAGE_TAG="depth-estimation-app"

# If --gpu flag is passed, switch to GPU Dockerfile and tag
if [[ "$1" == "--gpu" ]]; then
    echo "Building GPU Docker image..."
    DOCKERFILE="Dockerfile-gpu"
    IMAGE_TAG="depth-estimation-app:gpu"
else
    echo "Building CPU Docker image..."
    IMAGE_TAG="depth-estimation-app"
fi

# Build the Docker image
sudo docker build -f "$DOCKERFILE" . -t "$IMAGE_TAG"


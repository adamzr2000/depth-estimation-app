#!/bin/bash

echo "Starting CPU vAccel container..."
docker run --rm \
  -v "$(pwd)/app/app.py:/app/app.py" \
  -v "$(pwd)/app/app_test.py:/app/app_test.py" \
  --name cpu-vaccel \
  -p 8193:8192 \
  -d harbor.nbfc.io/nubificus/vaccel-monocular-cpu:x86_64 \
  tail -f /dev/null

if docker ps | grep -q cpu-vaccel; then
  echo "✅ CPU vAccel container is running."
else
  echo "❌ Failed to start CPU vAccel container."
fi

echo "Starting GPU vAccel container..."
docker run --gpus all --rm \
  --name gpu-vaccel \
  -v "$(pwd)/app/app.py:/app/app.py" \
  -v "$(pwd)/app/app_test.py:/app/app_test.py" \
  -p 8192:8192 \
  -d harbor.nbfc.io/nubificus/vaccel-monocular:x86_64

if docker ps | grep -q gpu-vaccel; then
  echo "✅ GPU vAccel container is running."
else
  echo "❌ Failed to start GPU vAccel container."
fi


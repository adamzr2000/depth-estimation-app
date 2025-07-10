#!/bin/bash

# Default RPC address
RPC_ADDRESS="tcp://10.5.1.21:8192"

# Parse optional --rpc argument
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rpc)
      RPC_ADDRESS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--rpc <address>]"
      exit 1
      ;;
  esac
done

echo "Starting CPU vAccel container..."
echo "Using VACCEL_RPC_ADDRESS=${RPC_ADDRESS}"

docker run --rm -d \
  --name vaccel-robot \
  -v "$(pwd)/app/app.py:/app/app.py" \
  -v "$(pwd)/app/app_test.py:/app/app_test.py" \
  -v "$(pwd)/app/templates/index.html:/app/templates/index.html" \
  -p 5554:5554 \
  --group-add video \
  -e VACCEL_EXEC_DLCLOSE_ENABLED=0 \
  -e VACCEL_EXEC_DLOPEN_MODE=lazy \
  -e VACCEL_LOG_LEVEL=1 \
  -e VACCEL_PLUGINS=libvaccel-rpc.so \
  -e VACCEL_RPC_ADDRESS="$RPC_ADDRESS" \
  harbor.nbfc.io/nubificus/vaccel-monocular-cpu:x86_64 \
  tail -f /dev/null

if docker ps | grep -q vaccel-robot; then
  echo "✅ CPU vAccel container is running."
else
  echo "❌ Failed to start CPU vAccel container."
fi

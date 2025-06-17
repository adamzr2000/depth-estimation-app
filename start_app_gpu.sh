#!/bin/bash

# Default command to run inside the container
default_cmd="python3 test_keras_tf_version.py"

# If the user passed arguments, use those as the full command; otherwise use the default
if [ "$#" -gt 0 ]; then
  cmd="$*"
else
  cmd="$default_cmd"
fi

docker run \
    -it \
    --rm \
    --name monocular-gpu \
    --hostname monocular-gpu \
    -v "$(pwd)"/app:/app \
    -e LD_LIBRARY_PATH=/app/models/monocular_deployed/lib_gpu:$LD_LIBRARY_PATH \
    --privileged \
    -p 5554:5554 \
    --runtime=nvidia \
    --group-add video \
    --gpus all \
    depth-estimation-app:gpu \
    bash -c "cd /app && $cmd"

echo "Done."

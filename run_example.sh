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
    --name monocular \
    --hostname monocular \
    -v "$(pwd)"/app:/app \
    -e LD_LIBRARY_PATH=/app/models/monocular_deployed:$LD_LIBRARY_PATH \
    -e OMP_NUM_THREADS=16 \
    -e DNNL_NUM_THREADS=16 \
    -e KMP_AFFINITY="granularity=fine,compact,1,0" \
    --privileged \
    -p 5554:5554 \
    --group-add video \
    depth-estimation-app:latest \
    bash -c "cd /app && $cmd"

echo "Done."


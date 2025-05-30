#!/bin/bash
docker run \
    -it \
    --rm \
    --name depth-estimation-app \
    --hostname depth-estimation-app \
    -v "$(pwd)"/app:/app \
    -e LD_LIBRARY_PATH=/app/monocular_deployed:$LD_LIBRARY_PATH \
    -e OMP_NUM_THREADS=16 \
    -e DNNL_NUM_THREADS=16 \
    -e KMP_AFFINITY="granularity=fine,compact,1,0" \
    --privileged \
    -p 5554:5554 \
    --runtime=nvidia \
    --group-add video \
    --gpus all \
    depth-estimation-app-gpu:latest \
    bash

echo "Done."

docker run \
    -it \
    --rm \
    --name depth-estimation-app \
    --hostname depth-estimation-app \
    -v $(pwd)/app:/app \
    -v $(pwd)/optimized-app:/optimized-app \
    -e LD_LIBRARY_PATH=/optimized-app:$LD_LIBRARY_PATH \
    --privileged \
    -p 5000:5000 \
    --runtime=nvidia \
    --group-add video \
    --gpus all \
    depth-estimation-app-gpu:latest \
    bash

echo "Done."
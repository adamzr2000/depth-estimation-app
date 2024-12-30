docker run \
    -it \
    --name depth-estimation-app \
    -v $(pwd)/app:/app \
    --rm \
    --net host \
    --privileged \
    --runtime=nvidia \
    --group-add video \
    --gpus all \
    depth-estimation-app-gpu:latest \
    bash 

echo "Done."
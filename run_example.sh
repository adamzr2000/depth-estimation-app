docker run \
    -it \
    --name depth-estimation-app \
    -v $(pwd)/app:/app \
    --rm \
    --net host \
    --privileged \
    --group-add video \
    depth-estimation-app:latest 

echo "Done."
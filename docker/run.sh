#!/bin/sh
#
# Run the docker container.

. docker/env.sh
docker run \
  -dit \
  --gpus all \
  -v $PWD:/workspace \
  -p $API_HOST_PORT:$API_CONTAINER_PORT \
  -p $MLFLOW_HOST_PORT:$MLFLOW_CONTAINER_PORT \
  --name $CONTAINER_NAME\
  --rm \
  --shm-size=2g \
  $IMAGE_NAME
docker exec \
  -d \
  $CONTAINER_NAME sh /workspace/docker/init.sh
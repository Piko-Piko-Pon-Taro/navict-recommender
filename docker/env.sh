#!/bin/sh
#
# Set environment variables.

export IMAGE_NAME=${USER}_navict_recommender
export CONTAINER_NAME=${USER}_navict_recommender
export API_HOST_PORT=8000
export API_CONTAINER_PORT=8000
export MLFLOW_HOST_PORT=8888
export MLFLOW_CONTAINER_PORT=8888
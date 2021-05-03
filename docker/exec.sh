  
#!/bin/sh
#
# Run shell in the docker container.

. docker/env.sh
docker exec \
  -it \
  $CONTAINER_NAME bash
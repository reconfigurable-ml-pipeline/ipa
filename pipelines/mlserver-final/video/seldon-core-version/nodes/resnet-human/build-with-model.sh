#!/bin/bash

# Define the folders to copy
folders=(
    /mnt/myshareddir/torchhub/resnet
)

REPOS=(
    sdghafouri
)

dockerfile="Dockerfile"
IMAGE_NAME=video-final-with-model:resnet-human

# Create the models directory
mkdir models

# Copy the folders to the models directory
for folder in "${folders[@]}"; do
    cp -r "$folder" models/
done

# Generate the Dockerfile
mlserver dockerfile --include-dockerignore .
sed -i 's/seldonio/sdghafouri/g' Dockerfile
sed -i 's/1.3.0.dev4-slim/custom-2-slim/g' Dockerfile

# Build and push the Docker image
DOCKER_BUILDKIT=1 docker build . --tag=$IMAGE_NAME
for REPO in "${REPOS[@]}"; do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done

# Delete the models directory
rm -r models

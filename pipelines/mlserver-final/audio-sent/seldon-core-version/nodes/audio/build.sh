REPOS=(
    sdghafouri)
IMAGE_NAME=audio-sent-final:audio
mlserver dockerfile --include-dockerignore .
sed -i 's/seldonio/sdghafouri/g' Dockerfile
sed -i 's/1.3.0.dev4-slim/custom-2-slim/g' Dockerfile
DOCKER_BUILDKIT=1 docker build . --tag=$IMAGE_NAME
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done
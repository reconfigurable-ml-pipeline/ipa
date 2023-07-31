REPOS=(
    sdghafouri)
IMAGE_NAME=video-final:yolo
mlserver dockerfile --include-dockerignore .
sed -i '/^USER 1000/i \
RUN microdnf update -y && microdnf install -y git' Dockerfile
sed -i 's/seldonio/sdghafouri/g' Dockerfile
sed -i 's/1.3.0.dev4-slim/custom-2-slim/g' Dockerfile
DOCKER_BUILDKIT=1 docker build . --tag=$IMAGE_NAME
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done
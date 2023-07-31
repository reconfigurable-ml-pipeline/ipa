# Customized MLServer

1. We will use the forked version at the branch [configure-custom](https://github.com/saeid93/MLServer/tree/configure-custom)
2. To install MLServer in the debug mode, activate your python environment:
```bash
make install-dev
```
3. There are two changes in this branch:
4. In the [Makefile](https://github.com/saeid93/MLServer/blob/84716ac670fcd8552038ebc29f6fc0ed59ebd171/Makefile#L33) the following lines have been added for building the image
```bash
build-saeid:
./hack/build-saeid.sh ${VERSION}
```
5. In the [hack](https://github.com/saeid93/MLServer/tree/configure-custom/hack) folder we have added [build-saeid.sh](https://github.com/saeid93/MLServer/blob/configure-custom/hack/build-saeid.sh) for generating image of our forked version.
6. To generate a custom image go to the [root folder] of the forked MLServer and make it using our own custom command with mentioning the version:
```bash
make build-saeid
```
7. Now go to the root folder of one of the models in the our own repo, [e.g.](https://github.com/reconfigurable-ml-pipeline/infernece-pipeline-joint-optimization/tree/main/pipelines/pipeline-connnection-latency/mlserver-mock/seldon-core-version/nodes/node-one) and use the docker mlserver command to make the Dockerfiles:
```bash
mlserver dockerfile --include-dockerignore .
```
8. Previous steps will make two file `Dockerfile` and `Dockerfile.dockerignore`, replace the base image with the custom image built in step 4 e.g. see [here](https://github.com/reconfigurable-ml-pipeline/infernece-pipeline-joint-optimization/blob/0364a841816fcf6baea84996cf229ee7491a130c/pipelines/pipeline-connnection-latency/mlserver-mock/seldon-core-version/nodes/node-one/Dockerfile#L31):
```bash
...
    chmod -R 776 $(dirname $MLSERVER_ENV_TARBALL)

FROM seldonio/mlserver:1.2.0.dev14-slim
SHELL ["/bin/bash", "-c"]

# Copy all potential sources for custom environments
COPY \
...
```
replaced with:
```bash
...
    chmod -R 776 $(dirname $MLSERVER_ENV_TARBALL)

FROM sdghafouri/mlserver:custom-slim
SHELL ["/bin/bash", "-c"]

# Copy all potential sources for custom environments
COPY \
...
```
9. build and push image
```bash
REPOS=(
    sdghafouri)
IMAGE_NAME=grpc-pipeline:node-one
# mlserver build . -t $IMAGE_NAME
DOCKER_BUILDKIT=1 docker build . --tag=$IMAGE_NAME
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done
```
10. Alternative to the steps 6, 7, 8 you can use the `bash build.sh` file for each model.

# Contribution to the [UpStream MLServer Repo](https://github.com/SeldonIO/MLServer)

1. Make an issue on the upstream branch
2. Make a new branch
3. Apply changes to the new branch
4. Run tests with `make test` (remember to make sure that you have `pip install tox` in your system)
5. pull to your own repo
6. Merge from upstream



Seldon core comes in two version, for the instruction of installing each respective version see below

# Seldon Core V1

## Seldon core isntallation - Istio

Seldon is a framework for making complex grpc and rest apis for the trained ML models

1. According to [doc](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html) install the istio version
```bash
kubectl create namespace seldon-system
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --namespace seldon-system
```
2. setup the ingress with istio [Guide](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html)
```yaml
cat <<EOF | kubectl apply -f -          
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
EOF
```

The `metadata.name` field is how the Seldon core is able to detect this Gateway. Bear in mind that the Gateway should also be in the `istio-system` namespace.

To access Seldon Services you have three options:

3.1. **Node Port (Recommended)** Edit the `kubectl edit service istio-ingressgateway -n istio-system` and make this change
```yaml
  - name: http2
    nodePort: 32000
    port: 80
    protocol: TCP
    targetPort: 8080
```
The Seldon core is available on `<cluster-ip>:32000`

3.2. **Port Forward** to the ingress port 80 (since the isio ingress you deployd in the former step is operating on port 80) to port 8004 and detach tmux (therfore the connection will stay open) - if you are using microk8s istio use the port 8080 as the target-port for http is 8080 in microk8s istio.
```
kubectl port-forward $(kubectl get pods -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].metadata.name}') -n istio-system 8004:8080
```
The Seldon core is ready to go on localhost:8004! For information about the Seldon core endpoint addresses see [endpoint-references](https://docs.seldon.io/projects/seldon-core/en/latest/ingress/istio.html#istio-configuration-annotation-reference)


3.3. **Ingress**


# Seldon Core V2 Dcoker Compose

1. Install Java +8 for kafka
```bash
sudo apt update
sudo apt install default-jdk
```

2. Install [Docker Compose](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

3. Install [Go lang](https://go.dev/doc/install) for compiling Seldon CLI

```bash
wget https://go.dev/dl/go1.19.4.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.19.4.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin >> ~/.zshrc
export GO111MODULE=on
```

3. Install [Seldon CLI](https://docs.seldon.io/projects/seldon-core/en/v2.0.0/contents/getting-started/cli.html)

```bash
git clone https://github.com/SeldonIO/seldon-core --branch=v2
cd seldon-core/operator
make build-seldon
cd bin
sudo cp seldon /usr/local/bin
```

4. Clone the [Seldon repo V2](https://github.com/SeldonIO/seldon-core/tree/v2)
```bash
git clone https://github.com/SeldonIO/seldon-core --branch=v2
```

5. Specify the local storage
```
mkdir ~/storage
export LOCAL_MODEL_FOLDER=~/storage
```

6. Based on the [Docker Installation](https://docs.seldon.io/projects/seldon-core/en/v2.0.0/contents/getting-started/docker-installation/index.html) install Seldon V2 with the following:
```bash
make deploy-local
```

7. To uninstall Seldon Docker installation:
```bash
make undeploy-local
```

# Seldon Core V2 K8S

1. Install Java +8 for kafka
```bash
sudo apt update
sudo apt install default-jdk
```

2. Install [Go lang](https://go.dev/doc/install) for compiling Seldon CLI

```bash
wget https://go.dev/dl/go1.19.4.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.19.4.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin >> ~/.zshrc
export GO111MODULE=on
```

3. Install [Seldon CLI](https://docs.seldon.io/projects/seldon-core/en/v2.0.0/contents/getting-started/cli.html)

```bash
git clone https://github.com/SeldonIO/seldon-core --branch=v2
cd seldon-core/operator
make build-seldon
cd bin
sudo cp seldon /usr/local/bin
```

3. Make sure ansible is installed using:
```bash
pip install ansible
```

4. Install the ecosystem following [Ansible installation](https://docs.seldon.io/projects/seldon-core/en/v2/contents/getting-started/kubernetes-installation/ansible.html#setup-ecosystem). The ecosystem contains the following packages

```bash
cd seldon-core/ansible
ansible-playbook playbooks/setup-ecosystem.yaml
```
It will install the follwoing dependancies
|                         | type   | default                     | comment                                                 |
|-------------------------|--------|-----------------------------|---------------------------------------------------------|
| seldon_mesh_namespace   | string | seldon-mesh                 | namespace to install seldon                             | 
| full_install            | bool   | yes                         | enables full ecosystem installation                     |
| install_kafka           | bool   | {{ full_install }}          | installs Kafka using seldonio.k8s.strimzi_kafka         |
| install_prometeus       | bool   | {{ full_install }}          | installs Prometheus using seldonio.k8s.prometheus       |
| install_certmanager     | bool   | {{ full_install }}          | installs certmanager using seldonio.k8s.certmanager     |
| install_jaeger          | bool   | {{ full_install }}          | installs Jaeger using seldonio.k8s.jaeger               |
| install_opentelemetry   | bool   | {{ full_install }}          | installs OpenTelemetry using seldonio.k8s.opentelemetry |
| configure_kafka         | bool   | {{ install_kafka }}         | configures Kafka using V2 specific resources            |
| configure_prometheus    | bool   | {{ install_prometheus }}    | configure Prometheus using V2 specific resources        |
| configure_jaeger        | bool   | {{ install_jaeger }}        | configure Jaeger using V2 specific resoruces            |
| configure_opentelemetry | bool   | {{ install_opentelemetry }} | configure OpenTelemetry using V2 specific resources     |


TODO solve the problem

5. Install the seldon-mesh 

6. Clone the [Seldon repo V2](https://github.com/SeldonIO/seldon-core/tree/v2)
```
git clone https://github.com/SeldonIO/seldon-core --branch=v2

```
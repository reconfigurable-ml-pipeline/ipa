# Microk8s

## Prometheus and Grafana

1. Just use the builtin add-on
```
microk8s enable prometheus
```
2. For enabling outside access make the `service/prometheus-k8s` and `service/grafana` of type NodePort isntead of ClusterIP using the following command and editing the `type` field. Also set the `web` nodePort for prometheus to 30090 and `http` port for grafana to 30300 (in some versions the namespace is `observibility` instead of `monitoring`)
```
kubectl edit svc prometheus-k8s -n monitoring
kubectl edit svc grafana -n monitoring
```
3. Find the Prometheus and grafana node ports using

For prometheus
```
kubectl get service prometheus-k8s -n monitoring -o jsonpath="{.spec.ports[0].nodePort}"
```
The output should be:
```
30090% 
```

For Grafana
```
kubectl get service grafana -n monitoring -o jsonpath="{.spec.ports[0].nodePort}"
```
The output should be:
```
30300% 
```


default credentials for accesing Grafana are:
```
username: admin
password: admin
```

4. Both of the grafana and prometheus are now accessible via the following links
```
<your node ip>:<prometheus port>
<your node ip>:<grafana port>
```

5. Following [Seldon-metrics](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/analytics.html) to integrate the Seldon core metrics into prometheus and grafana just deploy the following PodMonitor resource
```
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: seldon-podmonitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/managed-by: seldon-core
  podMetricsEndpoints:
    - port: metrics
      interval: 1s
      path: /prometheus
  namespaceSelector:
    any: true
EOF
```
  5.1. [Optional] To monitor triton server metrics:
```
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: triton-podmonitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: triton
  podMetricsEndpoints:
    - port: metrics
      path: /metrics
  namespaceSelector:
    any: true
EOF
```
  5.2. [Optional] To monitor Tensorflow serving, manually add a label "model_server: tfserving" to your Kubernetes Pods/Deployments, and apply the  PodMonitor bellow:
 ```
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: tfserving-podmonitor
  namespace: monitoring
  labels:
    podmonitor: tfserving
spec:
  namespaceSelector:
    any: true
  podMetricsEndpoints:
  - interval: 1s
    path: /monitoring/prometheus/metrics
  selector:
    matchLabels:
      model_server: tfserving
EOF
```
6. Add these podmonitors and servicemonitors for istio
```
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: envoy-stats-monitor-pod
  namespace: monitoring
  labels:
    monitoring: istio-proxies
    release: istio
spec:
  selector:
    matchLabels:
      istio.io/rev: default
    matchExpressions:
    - {key: istio-prometheus-ignore, operator: DoesNotExist}
  namespaceSelector:
    any: true
  jobLabel: envoy-stats
  podMetricsEndpoints:
  - path: /stats/prometheus
    interval: 1s
    relabelings:
    - action: keep
      sourceLabels: [__meta_kubernetes_pod_container_name]
      regex: "istio-proxy"
    - action: keep
      sourceLabels: [__meta_kubernetes_pod_annotationpresent_prometheus_io_scrape]
    - sourceLabels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      targetLabel: __address__
    - action: labeldrop
      regex: "__meta_kubernetes_pod_label_(.+)"
    - sourceLabels: [__meta_kubernetes_namespace]
      action: replace
      targetLabel: namespace
    - sourceLabels: [__meta_kubernetes_pod_name]
      action: replace
      targetLabel: pod_name
```

and

```
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: envoy-stats-monitor
  namespace: monitoring
  labels:
    monitoring: istio-proxies
    release: istio
spec:
  selector:
    matchExpressions:
    - {key: istio-prometheus-ignore, operator: DoesNotExist}
  namespaceSelector:
    any: true
  jobLabel: envoy-stats
  podMetricsEndpoints:
  - path: /stats/prometheus
    interval: 1s
    relabelings:
    - action: keep
      sourceLabels: [__meta_kubernetes_pod_container_name]
      regex: "istio-proxy"
    - action: keep
      sourceLabels: [__meta_kubernetes_pod_annotationpresent_prometheus_io_scrape]
    - sourceLabels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      targetLabel: __address__
    - action: labeldrop
      regex: "__meta_kubernetes_pod_label_(.+)"
    - sourceLabels: [__meta_kubernetes_namespace]
      action: replace
      targetLabel: namespace
    - sourceLabels: [__meta_kubernetes_pod_name]
      action: replace
      targetLabel: pod_name
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: istio-component-monitor
  namespace: monitoring
  labels:
    monitoring: istio-components
    release: istio
spec:
  jobLabel: istio
  targetLabels: [app]
  selector:
    matchExpressions:
    - {key: istio, operator: In, values: [pilot]}
  namespaceSelector:
    any: true
  endpoints:
  - port: http-monitoring
    interval: 1s
```
7. Following [kaili installation](https://istio.io/latest/docs/ops/integrations/kiali/#installation) and [jeager installation](https://istio.io/latest/docs/tasks/observability/distributed-tracing/jaeger/) install this two dashboard for extra infromation:
```
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/jaeger.yaml
```
make them loadbalancer and change the nodeport of kiali to `32001` and jeager `http-query` to `31166` for persistent access. For jaeger change the tracing service.

8. Setup this [grafana dashboard](grafana-dashboard.json) for our most used metrics

9. To check the scrape intervals and other prometheus configs use the following command:
```
kubectl -n monitoring get secret prometheus-k8s -ojson | jq -r '.data["prometheus.yaml.gz"]' | base64 -d | gzip --uncompress | grep interval
```

10. Changing the scrapping interval
Prometheus scrape the metrics from the endpoints within a fixed time interval this is set to 30s
For changing this do the following steps:
  1. 

# Bare Metal

1. We'll be installing this [this](docs/installing-prometheus.md) prometheus+grafana monitoring using Helm charts ([source](https://github.com/geerlingguy/kubernetes-101/tree/master/episode-10)). You can find guide to installing helm itself [here](https://helm.sh/docs/intro/install/). Alternate Prometeus installation solution [here](https://github.com/prometheus-operator/prometheus-operator).

This Helm chart installs the following in your cluster:

  - kube-state-metrics (gathers metrics from cluster resources)
  - Prometheus Node Exporter (gathers metrics from Kubernetes nodes)
  - Grafana
  - Grafana dashboards and Prometheus rules for Kubernetes monitoring

To install it, first add the Prometheus Community Helm repo and run `helm repo update`:

```
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

Then install the stack into the `monitoring` namespace:

```
kubectl create namespace monitoring
helm install prometheus --namespace monitoring prometheus-community/kube-prometheus-stack
```

Watch the progress in Lens, or via `kubectl`:

```
kubectl get deployments -n monitoring -w
```

2. For enabling outside access make the `service/prometheus-k8s` and `service/grafana` of type NodePort isntead of ClusterIP using the following command and editing the `type` field. Also set the port for prometheus to 30090 and for grafana to 30300. Name of the containers might be different based on the version.
```
kubectl edit svc prometheus-k8s -n monitoring
kubectl edit svc grafana -n monitoring
```
3. Find the Prometheus and grafana node ports using

For prometheus
```
kubectl get service prometheus-k8s -n monitoring -o jsonpath="{.spec.ports[0].nodePort}"
30090% 
```
For Grafana
```
kubectl get service grafana -n monitoring -o jsonpath="{.spec.ports[0].nodePort}"
30300% 
```

default credentials for accesing Grafana are:
```
username: admin
password: admin
```

4. Both of the grafana and prometheus are now accessible via the following links
```
<your node ip>:<prometheus port>
<your node ip>:<grafana port>
```

5. Following [Seldon-metrics](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/analytics.html) to integrate the Seldon core metrics into prometheus and grafana just deploy the following PodMonitor resource
```
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: seldon-podmonitor
spec:
  selector:
    matchLabels:
      app.kubernetes.io/managed-by: seldon-core
  podMetricsEndpoints:
    - port: metrics
      path: /prometheus
  namespaceSelector:
    any: true
EOF
```

## Unistalling Monitoring Stack

```
helm uninstall prometheus --namespace monitoring
```

This removes all the Kubernetes components associated with the chart and deletes the release.


CRDs created by this chart are not removed by default and should be manually cleaned up:
```
kubectl delete crd alertmanagerconfigs.monitoring.coreos.com
kubectl delete crd alertmanagers.monitoring.coreos.com
kubectl delete crd podmonitors.monitoring.coreos.com
kubectl delete crd probes.monitoring.coreos.com
kubectl delete crd prometheuses.monitoring.coreos.com
kubectl delete crd prometheusrules.monitoring.coreos.com
kubectl delete crd servicemonitors.monitoring.coreos.com
kubectl delete crd thanosrulers.monitoring.coreos.com
```

# Useful links
* clarification of difference between Prometheus operator and [Prometheus Helm - StackOverflow](https://stackoverflow.com/questions/54422566/what-is-the-difference-between-the-core-os-projects-kube-prometheus-and-promethe)

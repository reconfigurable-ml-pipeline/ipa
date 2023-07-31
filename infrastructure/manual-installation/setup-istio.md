# Microk8s
1. Istio is available as one of the [Microk8s addons](https://istio.io/latest/docs/setup/platform-setup/microk8s/), simply enable it as:
```
microk8s enable community
microk8s enable istio
```
2. Run the simple app in [istio-documentation-sample-app](https://istio.io/latest/docs/setup/getting-started/#bookinfo) and make sure it is running on the server and make sure that it is running the sample applicatio

3. Install Prometheus for Kiali:
```
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/prometheus.yaml
```
4. To install Kiali (for monitoring traces)
```
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
```
5. to enable outside access to it change it service to LoadBalancer on nodeport 32001
```
kubectl edit svc kiali -n istio-system
```

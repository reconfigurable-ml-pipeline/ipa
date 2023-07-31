To test Seldon core endpoint correct configuration and installation deploy the following:
```
cat <<EOF | kubectl apply -f -
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sklearn
spec:
  predictors:
  - graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.14.0-dev/sklearn/iris
    name: default
    replicas: 1
    svcOrchSpec:
      env:
      - name: SELDON_LOG_LEVEL
        value: DEBUG
EOF
```
Wait for the SeldonDeployment to be running:
```bash
kubectl get all                                             
NAME                                               READY   STATUS    RESTARTS   AGE
pod/sklearn-default-0-classifier-f7668b586-wg42v   2/2     Running   0          71s

NAME                                 TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)             AGE
service/sklearn-default              ClusterIP   10.152.183.32   <none>        8000/TCP,5001/TCP   71s
service/sklearn-default-classifier   ClusterIP   10.152.183.71   <none>        9000/TCP,9500/TCP   71s

NAME                                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/sklearn-default-0-classifier   1/1     1            1           71s

NAME                                                     DESIRED   CURRENT   READY   AGE
replicaset.apps/sklearn-default-0-classifier-f7668b586   1         1         1       71
```
Once all the pods of the seldon deployment are up, on your cluster nodes:
```
curl -s -d '{"data": {"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}' \
-X POST http://localhost:32000/seldon/default/sklearn/api/v1.0/predictions -H \
"Content-Type: application/json"
```
and on outside cluster machine:
```
CLUSTER_NODE_IP=192.5.86.160
curl -s -d '{"data": {"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}' \
-X POST http://$CLUSTER_NODE_IP:32000/seldon/default/sklearn/api/v1.0/predictions -H \
"Content-Type: application/json"
```
In both cases the result should be:
```
{"data":{"names":["t:0","t:1","t:2"],"ndarray":[[9.912315378486697e-07,0.0007015931307746079,0.9992974156376876]]},"meta":{"requestPath":{"classifier":"seldonio/sklearnserver:1.13.1"}}}
```

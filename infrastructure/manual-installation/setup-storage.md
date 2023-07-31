# Common Steps
## Installing NFS
NFS is our storage file system backend for more information on K8S storage system see [Volummes-k8s-doc](https://kubernetes.io/docs/concepts/storage/volumes/)

1. Follow the instructions [installing-nfs](https://cloud.netapp.com/blog/azure-anf-blg-linux-nfs-server-how-to-set-up-server-and-client)
2. In the `/etc/exports` file find the Chameleon IP network and mask and add it to the export file in the following format `/mnt/myshareddir {subnetIP}/{subnetMask}(rw,sync,no_subtree_check)` (For an unknown reason adding * deos not work for allowing all IPs). To find the network IP and mask (Make sure linux package `net-tools` is installed):
`
sudo ifconfig | grep -i mask
`

**Dont Forget** the last step
```
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

3. Install both client and server folders on your machines and make sure that a file created on one of them is accessible on the client folder too.


Bare in mind if you delete a PV or PVC you first delete the associated resource with that PVC first, otherwise it will stuck at the terminating state.

## Minio Operator installation (recommended) TODO not working follow [link](https://github.com/minio/operator)
This installation gives access to a beautiful GUI
1. Set up the PV and point it to the NFS directory and IP (in our case 10.140.81.236 and /mnt/myshareddir)


## Helm Minio installation

## Setup up Persistent Volume and Persistent Volume Claim
In k8s you have to set a PV and a claim for that PVC attached to your volume type which in our case is NFS
1. Set up the PV and point it to the NFS directory and IP (in our case 10.140.81.236 and /mnt/myshareddir)
```
IP=10.140.83.56
```
Then deploy:
```
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs
  namespace: default
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 200Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: $IP
    path: "/mnt/myshareddir"
EOF
```
2. Create the namespace for minio
```
kubectl create ns minio-system
```
3. Associate a PVC to the generated PV
```
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-nfs
  namespace: minio-system
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
EOF
```

4. Install the Minio with helm and set the value to our existing pvc, the user and admint sat here will be later used in the real

```
MINIOUSER=minioadmin
MINIOPASSWORD=minioadmin

helm repo add minio https://helm.min.io/

helm upgrade --install minio minio/minio \
--namespace minio-system \
--set accessKey=${MINIOUSER} \
--set secretKey=${MINIOPASSWORD} \
--set persistence.existingClaim=pvc-nfs
```

5. continue the rest of steps from the printed instructions
6. check the helm is installed from `helm list -n minio-system`
7. There is two options to access Minio from localhost:

7.1. **Option 1 LoadBalancer (Recommended)**: edit `kubectl edit service/minio -n minio-system` and change `spec.ports.nodePort=31900` and `spec.type=LoadBalancer`. You can now access Minio server on `http://<cluster-ip>:31900`.

7.2. **Option 2 Port forward** run the below commands:

```
export POD_NAME=$(kubectl get pods --namespace minio-system -l "release=minio" -o jsonpath="{.items[0].metadata.name}")
kubectl port-forward $POD_NAME 9000 --namespace minio-system
```
You can now access Minio server on http://localhost:9000.


8. find out access keys
```
ACCESS_KEY=$(kubectl get secret minio -n minio-system -o jsonpath="{.data.accesskey}" | base64 --decode)
SECRET_KEY=$(kubectl get secret minio -n minio-system -o jsonpath="{.data.secretkey}" | base64 --decode)
```
echo secret and access key for accessing minio dashboard:
```
echo $ACCESS_KEY
echo $SECRET_KEY
```

9. Download the Minio mc [client](https://docs.minio.io/docs/minio-client-quickstart-guide)
```
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
./mc --help
```
10. Do the following for terminal access login to the CLI:
```
sudo cp mc /usr/local/bin
``` 
11. Setup the variables for the command line
```
mc alias set minio http://localhost:9000 "$ACCESS_KEY" "$SECRET_KEY" --api s3v4
mc ls minio
```
if you have used load balancer access change http://localhost:9000 to the server
```
mc alias set minio http://localhost:31900 "$ACCESS_KEY" "$SECRET_KEY" --api s3v4
mc ls minio
```
if it is for your local computer use the cluster ip
```
CLUSTER_IP=192.5.86.160
mc alias set minio http://$CLUSTER_IP:31900 "$ACCESS_KEY" "$SECRET_KEY" --api s3v4
mc ls minio
```
To make a bucket and copy files to it:

```
mc mb minio/<bucket>
mc cp ./<filename> minio/<bucket>/
```

Applications can access the Minio on the following `<address>/<port>`
```
minio.minio-system.svc.cluster.local:9000
```

## Remote access to MinIO object storage (Seldon)
1. Set REMOTE_IP to your remote/floating IP
```
REMOTE_IP=192.5.87.211
```
2. Apply this rclone config
```
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: seldon-rclone-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: minio
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: minioadmin
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: minioadmin
  RCLONE_CONFIG_S3_ENDPOINT: http://$REMOTE_IP:31900
EOF
```
2. In your SeldonDeployment yaml, set `modelUri` as usual (as it is set when local object storage is being used) and set `envSecretRefName` to `seldon-rclone-secret` for rclone to know where the object storage resides.

> The Secret defined in step 1 and the SeldonDeployment must be in the same Kubernetes namespace.

## Remote access to MinIO object storage (Nvidia Triton)
Following [this toturial](https://thenewstack.io/deploy-nvidia-triton-inference-server-with-minio-as-model-store/)
```
kubectl create secret generic aws-credentials --from-literal=AWS_ACCESS_KEY_ID=minioadmin --from-literal=AWS_SECRET_ACCESS_KEY=minioadmin
```

## Resources
[Minio doc-MinIO Helm Chart](https://github.com/minio/minio/tree/master/helm/minio) \
[Seldon Minio installation doc](https://deploy.seldon.io/en/v1.2/contents/getting-started/production-installation/minio.html) \
[Minio Helm Chart Values](https://github.com/minio/minio/blob/master/helm/minio/values.yaml) \
[Persistent Volume, Persistent Volume Claim & Storage Class- Nana](https://youtu.be/0swOh5C3OVM) \
[NFS Persistent Volume in Kubernetes Cluster - justmeandopensource](https://youtu.be/to14wmNmRCI)

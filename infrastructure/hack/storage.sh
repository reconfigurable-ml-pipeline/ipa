#!/bin/bash

PRIVATEIP=$(hostname -I | cut -d' ' -f1)


function setup_storage() {
    REMOTEIP="$1"

    echo "Setup storage: Install NFS"
    sudo apt install -y nfs-kernel-server
    sudo mkdir /mnt/myshareddir
    sudo chown nobody:nogroup /mnt/myshareddir
    sudo chmod 777 /mnt/myshareddir
    echo "/mnt/myshareddir $PRIVATEIP/30(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
    sudo exportfs -a
    sudo systemctl restart nfs-kernel-server
    echo "Setup storage: End Install NFS"
    echo

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
    server: $PRIVATEIP
    path: "/mnt/myshareddir"
EOF

    kubectl create ns minio-system
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

    MINIOUSER=minioadmin
    MINIOPASSWORD=minioadmin

    helm repo add minio https://helm.min.io/

    helm upgrade --install minio minio/minio \
    --namespace minio-system \
    --set accessKey=${MINIOUSER} \
    --set secretKey=${MINIOPASSWORD} \
    --set persistence.existingClaim=pvc-nfs
    kubectl patch svc minio -n minio-system --type='json' -p '[{"op":"replace","path":"/spec/type","value":"LoadBalancer"}]'
    kubectl patch svc minio -n minio-system --patch '{"spec": {"type": "LoadBalancer", "ports": [{"port": 9000, "nodePort": 31900}]}}'
    ACCESS_KEY=$(kubectl get secret minio -n minio-system -o jsonpath="{.data.accesskey}" | base64 --decode)
    SECRET_KEY=$(kubectl get secret minio -n minio-system -o jsonpath="{.data.secretkey}" | base64 --decode)
    wget https://dl.min.io/client/mc/release/linux-amd64/mc
    chmod +x mc
    sudo cp mc /usr/local/bin
    mc alias set minio http://localhost:31900 "$ACCESS_KEY" "$SECRET_KEY" --api s3v4
    sudo mc ls minio

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
  RCLONE_CONFIG_S3_ENDPOINT: http://$REMOTEIP:31900
EOF

    echo "End Setup storage"
    echo
}

echo "Running script"
setup_storage "$1"

echo "Script execution complete"

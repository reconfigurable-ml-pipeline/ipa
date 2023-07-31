# Options for Installing K8S
Here are the options for installing K8S on Chameleon cloud with GPU Support
## Microk8s
  1. Use the image or install Microk8s from its [documentation](https://ubuntu.com/tutorials/install-a-local-kubernetes-with-microk8s?&_ga=2.247530752.628037779.1650564942-2133565126.1649957392#1-overview)
  2. Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) for api access both on your local and your server
  3. To enable outside access to your cluster with kubectl in one of the cluster machines [get the kubeconfig](https://microk8s.io/docs/working-with-kubectl) and copy it to the outside cluster machine (e.g. your laptop) `~/.kube/config` file
```
cd $HOME
mkdir .kube
cd .kube
microk8s config > config
  ```
  4. To enable outside externally (from an external machine e.g. your local) do the following:
      1. According to this [issue](https://github.com/canonical/microk8s/issues/421) on your cluster master node disable the firewall on the port 16443 which is the default apiserver port
      ```
      sudo ufw allow 16443
      sudo ufw enable
      ``` 
      2. Find out your master node public ip from the Chameleon dashboard and according to [question](https://stackoverflow.com/questions/63451290/microk8s-devops-unable-to-connect-to-the-server-x509-certificate-is-valid-f) use the instruction given in [authentication and authorization](https://stackoverflow.com/questions/63451290/microk8s-devops-unable-to-connect-to-the-server-x509-certificate-is-valid-f) to include the public ip in the server certificates in the file `/var/snap/microk8s/current/certs/csr.conf.template`. As said in this [answer](https://stackoverflow.com/a/65571967) that the ip should be `USE IP > 100`
      ```
      ...

      [ alt_names ]
      DNS.1 = kubernetes
      DNS.2 = kubernetes.default
      DNS.3 = kubernetes.default.svc
      DNS.4 = kubernetes.default.svc.cluster
      DNS.5 = kubernetes.default.svc.cluster.local
      IP.1 = 127.0.0.1
      IP.2 = 192.168.1.1
      IP.100 = 192.168.1.1 # USE IP > 100
      #MOREIPS
      ``` 
  5. Enable the GPU of the Microk8s [add-on:gpu](https://microk8s.io/docs/addon-gpu)
  6. Stress test GPU for checking activeness [test](https://docs.mirantis.com/mke/3.4/ops/deploy-apps-k8s/gpu-support.html)
  8. I have build the image for it with name microk8s-cluster in the Chameleon repository, just use it to fire up a server.

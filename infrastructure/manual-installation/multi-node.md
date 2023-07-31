### All nodes
In all nodes add the ip address of all the nodes to their /etc/hosts file. E.g.

```
10.140.82.56 microk8s-cluster-1-1
10.140.81.248 microk8s-cluster-1-2
10.140.82.168 microk8s-cluster-2-1
10.140.82.23 microk8s-cluster-2-2
```

### Manager Node　

In the Manager node (after initializing a microk8s cluster), run the following:

`microk8s add-node`

Then, you will see some instructions; Copy the one with the following template:

`microk8s join IP:PORT/TOKEN --worker`

---

### Minions
In minions (worker nodes), do the following (one by one):
```
sudo snap install microk8s --classic --channel=1.23/edge
sudo usermod -a -G microk8s cc
newgrp microk8s
```

and then, paste the command copied from the manager node.

But do not forget if you use servers from different clusters, you must use the public IP of the manager node rather than its private IP in the microk8s join command. 

Bear in mind that you should add the IP addresses of minios in the manager and also add the IP of the manager and its name to /etc/hosts of all of the servers.



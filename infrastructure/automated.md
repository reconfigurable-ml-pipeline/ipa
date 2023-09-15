# Inference Project Automation

1. We have used [chameleon cloud](https://www.chameleoncloud.org/)
   1. Steps to leasing a server from Chameleon cloud [chameleon-lease](manual-installation/chameleon-lease.md) and make instances according to the paper machine specifications. We recommend using Chameleon UC datacenter and compute_cascadelake_r_ib machines with with 96 Intel(R) Xeon(R) Gold 6240R CPU @2.40GHz cores and 188 Gb of RAM. Use ubuntu20.04 as the base images.

2. On the Main node (Could be any of the cluster nodes):
    1. run `source hack/zsh.sh` in the root directory; reply 'Y' to setting zsh as default
    2. run `source build.sh PUBLIC_IP` in the project root directory. This will install all the necessary components, including kuberentes, supporting libraries (e.g. MLServer and load tester) and dependancies. It will also download data from the object storage. Logs of the previous experiments and modes to be stored in Minio Object storage are saved in the two google cloud bucket [ipa-results-1](https://console.cloud.google.com/storage/browser/ipa-results-1) and [ipa-models](https://console.cloud.google.com/storage/browser/ipa-models)
3. For attaching minions to the Main node and the cluster use [this instructions](manual-installation/multi-node.md), video and audio-qa/sent pipeline fit into one node so we recommend to test them, however in case you are interested to generate full results attach the extra nodes as per instructed in the paper

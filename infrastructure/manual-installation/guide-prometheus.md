# Metrics Sources

We have three sources of metrics so far:
1. Usage metrics of containers coming from the [caadvisor](https://github.com/google/cadvisor) see [full list of caadvisor metrics](https://github.com/google/cadvisor/blob/master/docs/storage/prometheus.md#prometheus-container-metrics) and node usage metrics is coming from [node-exporter](https://github.com/prometheus/node_exporter)
2. Istio nework metrics coming from the [istio exproter](https://istio.io/latest/docs/ops/integrations/prometheus/)
3. Seldon specific metrics coming from the [Seldon core exporter](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/analytics.html)

# Metrics of interest
Here is the table of the metrics we used in this project and are of interest to us:

TODO make a table of metrics like [full list of caadvisor metrics](https://github.com/google/cadvisor/blob/master/docs/storage/prometheus.md#prometheus-container-metrics)

# Metrics
Check the following links for a better understanding of the Prometheus and PromQL language:
1. [Metrics Types](https://prometheus.io/docs/concepts/metric_types/) vs [Query Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/#expression-language-data-types)
2. [Understanding Prometheus Range Vectors](https://satyanash.net/software/2021/01/04/understanding-prometheus-range-vectors.html#:~:text=Instant%20vectors%20have%20a%20single,on%20them%3B%20Range%20vectors%20cannot.)
3. [The Anatomy of a PromQL Query](https://promlabs.com/blog/2020/06/18/the-anatomy-of-a-promql-query)


The metrics metadata are accessible from the HTTP API of the prometheus accessible at `/api/v1`. The api can also gives us the same information as the dashboard. However, the metadata of the metrics is only avaialable from the http API. E.g.
```
$PROM_SERVER=http://192.5.86.160:30090
curl -G $PROM_server/api/v1/targets/metadata --data-urlencode 'metric=container_cpu_cfs_periods_total'
```
returns:
```
{"status":"success","data":[{"target":{"endpoint":"https-metrics","instance":"10.140.81.236:10250","job":"kubelet","metrics_path":"/metrics/cadvisor","namespace":"kube-system","node":"k8s-cluster","service":"kubelet"},"type":"counter","help":"Number of elapsed enforcement period intervals.","unit":""}]}%
```

# List of useful commands for Promentheus
## contaienrs resource usgae metrics

All containers memory usages:
```
container_memory_usage_bytes{container='sample-vpa'}
```
All containers cpu usages:
```
TODO
```
A specific container memory usage
```
TODO
```
A specific container cpu usage
```
TODO
```
## latency metrics
TODO

## Guide to Grafana dashboard
TODO

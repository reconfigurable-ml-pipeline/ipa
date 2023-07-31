TODO Have a detailed explanation for each of the pipelines and the installation

## 1. Graph making capabilities
TODO brnching @saeid

TODO @saeid whatever you find useful
TODO @mehran
https://github.com/SeldonIO/seldon-core/tree/master/wrappers/s2i/python/test/pytorch-env

## 2. Request tracing
TODO pixie - write the capabilities @saeid

TODO istio - write the capabilities @saeid

## 3. Resource usage tools
TODO pixie - write the capabilities @saeid

TODO prometeous and grafana seldon integration @mehran

## 4. Load testing
TODO @mehran https://docs.seldon.io/projects/seldon-core/en/latest/reference/benchmarking.html

## 5. Autoscaling
We can scale (not autoscale) each node of the graph separately, this can be used in conjunction with the python clien API
to make our own autoscaling machenism (the easiest way of implementing the autoscaling process)
links to scaling examples [this repo link](seldon/capabilities/scaling/scale.ipynb) and [external like](https://docs.seldon.io/projects/seldon-core/en/latest/graph/scaling.html).

For horizontal autoscaling the builtin HPA can be deployed directly with the builtin autoscaler inference. Sample [code](https://docs.seldon.io/projects/seldon-core/en/latest/examples/autoscaling_example.html)

for vertical autoscaling: No interface, it should be added to the Seldon.


TODO @mehran

TODO @saeid

## 6. Language tool

A check on the model-driven DSL TODO
TODO @saeid
from barazmoon import MLServerAsyncGrpc
from barazmoon import Data
import asyncio
import time
import numpy as np

load = 1
test_duration = 5
variant = 0
platform = "seldon"
mode = "equal"

request = {
    "times": {
        "models": {
            "audio": {"arrival": 1672276157.286681, "serving": 1672276157.2869108}
        }
    },
    "model_name": "nlp-qa",
    # "outputs": [
    #     {
    #         "data": "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
    #     }
    "outputs": [{"data": "martial"}],
}


times = str([str(request["times"]["models"])])
data = request["outputs"][0]["data"]

data_shape = [1]
custom_parameters = {"times": str(times)}
data_1 = Data(data=data, data_shape=data_shape, custom_parameters=custom_parameters)

# single node inference
if platform == "seldon":
    endpoint = "localhost:32000"
    deployment_name = "nlp-qa"
    model = "nlp-qa"
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
elif platform == "mlserver":
    endpoint = "localhost:8081"
    model = "nlp-qa"
    metadata = []

workload = [load] * test_duration
data_shape = [len(data)]
data_type = "text"

start_time = time.time()

load_tester = MLServerAsyncGrpc(
    endpoint=endpoint,
    metadata=metadata,
    workload=workload,
    model=model,
    data=[data_1],
    mode=mode,  # options - step, equal, exponential
    data_shape=data_shape,
    data_type=data_type,
)

responses = asyncio.run(load_tester.start())

print(responses)
print(f"{(time.time() - start_time):2.2}s spent in total")

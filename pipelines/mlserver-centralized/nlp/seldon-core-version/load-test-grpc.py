import os
import pathlib
import asyncio
from barazmoon import MLServerAsyncGrpc
from barazmoon import Data
import asyncio
import time

load = 8
test_duration = 300
variant = 0
platform = "router"
workload = [load] * test_duration
data_shape = [1]
data_type = "text"
mode = "equal"  # options - step, equal, exponential

PATH = pathlib.Path(__file__).parent.resolve()
with open(os.path.join(PATH, "input-sample.txt"), "r") as openfile:
    data = openfile.read()

data_shape = [1]
data_1 = Data(data=data, data_shape=data_shape)

# single node inference
if platform == "router":
    endpoint = "localhost:32000"
    deployment_name = "router"
    model = "router"
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
elif platform == "seldon":
    endpoint = "localhost:32000"
    deployment_name = "router"
    model = "router"
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
elif platform == "mlserver":
    endpoint = "localhost:8081"
    model = "nlp-li"
    metadata = []

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

print(f"{(time.time() - start_time):2.2}s spent in total")
print(responses)

import os
import pathlib
import asyncio
from barazmoon import MLServerAsyncGrpc
from barazmoon import Data
import asyncio
import time

load = 5
test_duration = 5
variant = 0
platform = "seldon"
image_name = "input-sample.JPEG"
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
if platform == "seldon":
    endpoint = "localhost:32000"
    deployment_name = "nlp-li"
    model = "nlp-li"
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

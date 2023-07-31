from barazmoon import MLServerAsyncGrpc
from barazmoon import Data
import asyncio
import time
import json
import os
import pathlib
from PIL import Image
import numpy as np


def image_loader(folder_path, image_name):
    image = Image.open(os.path.join(folder_path, image_name))
    # if there was a need to filter out only color images
    # if image.mode == 'RGB':
    #     pass
    return image


PATH = pathlib.Path(__file__).parent.resolve()
data = image_loader(PATH, "input-sample.JPEG")
with open(os.path.join(PATH, "input-sample-shape.json"), "r") as openfile:
    data_shape = json.load(openfile)
    data_shape = data_shape["data_shape"]
data = np.array(data).flatten()


load = 3
test_duration = 5
variant = 0
platform = "router"
workload = [load] * test_duration
data_type = "image"
mode = "equal"  # options - step, equal, exponential
image = "input-sample.JPEG"
image_size = "input-sample-shape.json"

# single node inference
if platform == "router":
    endpoint = "localhost:32000"
    deployment_name = "router"
    model = "router"
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
elif platform == "mlserver":
    endpoint = "localhost:8081"
    model = "router"
    metadata = []
elif platform == "seldon":
    endpoint = "localhost:32000"
    deployment_name = "video"
    model = None
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]

custom_parameters = {"custom_2": "test_2"}
data_1 = Data(data=data, data_shape=data_shape, custom_parameters=custom_parameters)

# Data list
data = []
data.append(data_1)

start_time = time.time()

load_tester = MLServerAsyncGrpc(
    endpoint=endpoint,
    metadata=metadata,
    workload=workload,
    model=model,
    data=data,
    mode=mode,  # options - step, equal, exponential
    data_type=data_type,
)

responses = asyncio.run(load_tester.start())

print(f"{(time.time() - start_time):2.2}s spent in total")

import matplotlib.pyplot as plt
import numpy as np

num_failed = 0
total = 0
for second_response in responses:
    for resp in second_response:
        if "failed" in resp.keys():
            num_failed += 1
        total += 1

print(f"{num_failed} failed out of total {total}")


# # roundtrip latency
# roundtrip_lat = []
# for sec_resps in responses:
#     for resp in sec_resps:
#         request_times = resp['times']['request']
#         sending_time = request_times['arrival'] - request_times['sending']
#         roundtrip_lat.append(sending_time)
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(roundtrip_lat)), roundtrip_lat)
# ax.set(xlabel='request id', ylabel='roundtrip latency (s)', title=f'roundtrip latency, total time={round((time.time() - start_time))}')
# ax.grid()
# fig.savefig(f"grpc-compressed-image-{platform}_variant_{variant}-roundtrip_lat-load-{load}-test_duration-{test_duration}.png")
# plt.show()

# # sending time
# start_times = []
# for sec_resps in responses:
#     for resp in sec_resps:
#         times = resp['times']['request']
#         sending_time = times['sending'] - start_time
#         start_times.append(sending_time)
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(start_times)), start_times)
# ax.set(xlabel='request id', ylabel='sending time (s)', title=f'load tester sending time, total time={round((time.time() - start_time))}')
# ax.grid()
# fig.savefig(f"grpc-compressed-image-{platform}_variant_{variant}-sending_time-load-{load}-test_duration-{test_duration}.png")
# plt.show()

# # server arrival time
# server_arrival_time = []
# for sec_resps in responses:
#     for resp in sec_resps:
#         times = resp['times']
#         server_recieving_time = times['models'][model]['arrival'] - start_time
#         server_arrival_time.append(server_recieving_time)
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(server_arrival_time)), server_arrival_time)
# ax.set(xlabel='request id', ylabel='server arrival time (s)', title=f'Server recieving time of requests, total time={round((time.time() - start_time))}')
# ax.grid()
# fig.savefig(f"grpc-compressed-image-{platform}_variant_{variant}-server_arrival_time_from_start-load-{load}-test_duration-{test_duration}.png")
# plt.show()

# server arrival latency
# model = 'resnet-human'
# server_arrival_latency = []
# for sec_resps in responses:
#     for resp in sec_resps:
#         times = resp['times']
#         server_recieving_time = times['models'][model]['arrival'] - times['request']['sending']
#         server_arrival_latency.append(server_recieving_time)
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(server_arrival_latency)), server_arrival_latency)
# ax.set(xlabel='request id', ylabel='server arrival latency (s)', title=f'Server recieving latency, total time={round((time.time() - start_time))}')
# ax.grid()
# fig.savefig(f"grpc-compressed-image-{platform}_variant_{variant}-server_recieving_latency-load-{load}-test_duration-{test_duration}.png")
# plt.show()

# print(f"{np.average(server_arrival_latency)}=")
print(responses)

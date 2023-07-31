import argparse
from functools import partial
import os
import sys

from PIL import Image
import numpy as np

import asyncio
import time
import pathlib
import os
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

request = {
    "times": {
        "models": {
            "nlp-trans": {"arrival": 1672276157.286681, "serving": 1672276157.2869108}
        }
    },
    "model_name": "nlp-trans",
}

PATH = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(PATH, "input-sample.txt"), "r") as openfile:
    data = openfile.read()

times = str([str(request["times"]["models"])])

data_shape = [1]
custom_parameters = {"times": str(times)}
# data_1 = Data(
#     data=data,
#     data_shape=data_shape,
#     custom_parameters=custom_parameters
# )


if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


client = grpcclient

data_shape = [1]
data = [
    """
        After decades as a martial arts practitioner and runner, Wes \"found\" yoga in 2010.
        He came to appreciate that its breadth and depth provide a wonderful ballast to stabilize the body and mind in the fast and technology-oriented lifestyle of today;
        yoga is an antidote to stress and a path to a better understanding of oneself and others.
        He is a RYT 500 certified yoga instructor of the YogaWorks program and has trained with contemporary masters,
        including Maty Ezraty, co-founder of YogaWorks and master instructor of the Iyengar and Ashtanga traditions,
        as well as a specialization with Mr. Bernie. Clark, a master instructor of the Yin tradition.
        His courses reflect these traditions, where he combines the fundamental basis of a precise alignment with elements of balance and concentration.
        These intertwine to help provide a way to cultivate a consciousness of yourself, others and the world around you,
        as well as to create a self-evident refuge against the physical style of life. He's just doing the right thing for himself.
        He's doing the right thing for himself. He's doing the right thing for himself. He's doing the right thing for himself.
        He's doing the right thing for himself. He's doing the right for the right thing for the right now. He's doing the right thing for the right now.
        He's doing the right for the right now. He's doing the right for the right for the right for the right for the right thing for the right for
        the right for the right for the right for the right thing for the right for the right thing. He's for the right for the right for the right thing.
    """
]
endpoint = "localhost:8081"
model = "nlp-sum"
inputs = [client.InferInput("input_name", data_shape, "BYTES")]
inputs[0].set_data_from_numpy(np.array([data[0].encode()]))
user_data = UserData()

try:
    triton_client = grpcclient.InferenceServerClient(url=endpoint, verbose=False)
except Exception as e:
    print(f"client creation error {e}")


try:
    triton_client.async_infer(model, inputs, partial(completion_callback, user_data))
except Exception as e:
    print(f"send request error {e}")

import os
import time
from mlserver import MLModel
import numpy as np

# import torch
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
)
from mlserver import MLModel
from typing import List
import mlserver.types as types

# from transformers import pipeline
from dataclasses import dataclass
from urllib import response
from barazmoon import MLServerAsyncGrpc
from barazmoon import Data
from datasets import load_dataset
import asyncio
import time
import numpy as np
import mlserver.types as types


import aiohttp
import asyncio
import grpc
from mlserver.codecs.string import StringRequestCodec
from mlserver.codecs.numpy import NumpyRequestCodec
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters


async def send_requests(ch, model_name, payload, metadata):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model_name, model_version=None
    )
    response = await grpc_stub.ModelInfer(
        request=inference_request_g, metadata=metadata
    )
    return response


async def predict(payload: InferenceRequest) -> InferenceResponse:
    arrival_time = time.time()
    request_input = payload.inputs[0]
    # self.request_counter += 1
    # logger.info(f"request counter_1:\n{self.request_counter}\n")

    endpoint = "localhost:32000"
    namespace = "default"

    # --------- model one ---------
    deployment_name_one = "audio"
    model_name_one = "audio"

    metadata_one = [("seldon", deployment_name_one), ("namespace", namespace)]
    payload_input = types.InferenceRequest(inputs=[request_input])
    async with grpc.aio.insecure_channel(endpoint) as ch:
        output_one = await send_requests(
            ch, model_name_one, payload_input, metadata_one
        )
    inference_response_one = converters.ModelInferResponseConverter.to_types(output_one)

    # --------- model two ---------
    deployment_name_two = "nlp-qa"
    model_name_two = "nlp-qa"
    metadata_two = [("seldon", deployment_name_two), ("namespace", namespace)]
    input_two = inference_response_one.outputs[0]
    payload_two = types.InferenceRequest(inputs=[input_two])
    async with grpc.aio.insecure_channel(endpoint) as ch:
        payload = await send_requests(ch, model_name_two, payload_two, metadata_two)
    inference_response = converters.ModelInferResponseConverter.to_types(payload)

    # logger.info(f"request counter_2:\n{self.request_counter}\n")
    # logger.info(f"batch counter:\n{self.batch_counter}\n")
    return inference_response


endpoint = "localhost:32000"
deployment_name = "audio"
model = "audio"
namespace = "default"
metadata = [("seldon", deployment_name), ("namespace", namespace)]

batch_test = 5
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
)

input_data = ds[0]["audio"]["array"][1:500]
data_shape = [len(input_data)]
custom_parameters = {"custom_2": "test_2"}
payload = types.InferenceRequest(
    inputs=[
        types.RequestInput(
            name="audio-bytes",
            shape=[1],
            datatype="BYTES",
            data=[input_data.tobytes()],
            parameters=types.Parameters(
                dtype="f4", datashape=str(data_shape), **custom_parameters
            ),
        )
    ]
)


async def main():
    responses = await asyncio.gather(*[predict(payload) for _ in range(10)])
    print(responses)


if __name__ == "__main__":
    asyncio.run(main())

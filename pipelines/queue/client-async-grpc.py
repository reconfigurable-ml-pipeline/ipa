from urllib import response
import grpc
from pprint import PrettyPrinter
from mlserver.types import InferenceResponse
from mlserver.grpc.converters import ModelInferResponseConverter
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
from mlserver.codecs.string import StringRequestCodec
from mlserver.codecs.string import StringRequestCodec

pp = PrettyPrinter(indent=4)
from datasets import load_dataset
import mlserver.types as types
import json
import asyncio


async def send_requests(ch, payload, metadata):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model, model_version=None
    )
    response = await grpc_stub.ModelInfer(
        request=inference_request_g, metadata=metadata
    )
    return response


# single node mlserver
# endpoint = "localhost:8081"
# model = 'queue'
# metadata = []


# single node seldon+mlserver
endpoint = "localhost:32000"
deployment_name = "queue"
model = "queue"
namespace = "default"
metadata = [("seldon", deployment_name), ("namespace", namespace)]

batch_test = 10
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
)

input_data = ds[0]["audio"]["array"][0:10]
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
    async with grpc.aio.insecure_channel(endpoint) as ch:
        responses = await asyncio.gather(
            *[send_requests(ch, payload, metadata) for _ in range(10)]
        )

    pp.pprint(responses)


asyncio.run(main())

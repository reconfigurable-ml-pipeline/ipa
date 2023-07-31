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


async def request_after_grpc(grpc_stub, payload, metadata):
    await asyncio.sleep(0.001)

    response = await grpc_stub.ModelInfer(request=payload, metadata=metadata)

    return response


# single node mlserver
# endpoint = "localhost:8081"
# model = 'audio'
# metadata = []


# single node seldon+mlserver
endpoint = "localhost:32000"
deployment_name = "audio"
model = "audio"
namespace = "default"
metadata = [("seldon", deployment_name), ("namespace", namespace)]

batch_test = 10
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
)

input_data = ds[0]["audio"]["array"]
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
import time


async def main():
    t1 = time.time()
    async with grpc.aio.insecure_channel(endpoint) as ch:
        inference_request_g = converters.ModelInferRequestConverter.from_types(
            payload, model_name=model, model_version=None
        )

        grpc_stub = dataplane.GRPCInferenceServiceStub(ch)
        t2 = time.time()

        tasks = [
            asyncio.ensure_future(
                request_after_grpc(grpc_stub, inference_request_g, metadata)
            )
            for _ in range(batch_test)
        ]
        print(f"creation time: {time.time() - t2}")
        responses = await asyncio.gather(*tasks)
        # raw_jsons = list(map(
        #     lambda inference_response: StringRequestCodec.decode_response(
        #         inference_response), responses))
    print(f"time: {time.time() - t1}")

    # inference_responses = list(map(
    #     lambda response: ModelInferResponseConverter.to_types(response), responses))
    # raw_jsons = list(map(
    #     lambda inference_response: StringRequestCodec.decode_response(
    #         inference_response), inference_responses))
    # outputs = list(map(
    #     lambda raw_json: json.loads(raw_json[0]), raw_jsons))

    # pp.pprint(responses)


asyncio.run(main())

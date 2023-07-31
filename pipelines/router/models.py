import os
import time
from typing import Dict, List
from mlserver import MLModel
import json
from mlserver.logging import logger
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver import MLModel
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
)
import grpc
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
import mlserver

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "router"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )

try:
    DROP_LIMIT = float(os.environ["DROP_LIMIT"])
    logger.info(f"DROP_LIMIT set to: {DROP_LIMIT}")
except KeyError as e:
    DROP_LIMIT = 1000
    logger.info(f"DROP_LIMIT env variable not set, using default value: {DROP_LIMIT}")

try:
    MODEL_LISTS: List[str] = json.loads(os.environ["MODEL_LISTS"])
    logger.info(f"MODEL_LISTS set to: {MODEL_LISTS}")
except KeyError as e:
    raise ValueError(f"MODEL_LISTS env variable not set!")

try:
    LOGS_ENABLED = os.getenv("LOGS_ENABLED", "True").lower() in ("true", "1", "t")
    logger.info(f"LOGS_ENABLED set to: {LOGS_ENABLED}")
except KeyError as e:
    LOGS_ENABLED = True
    logger.info(
        f"LOGS_ENABLED env variable not set, using default value: {LOGS_ENABLED}"
    )

if not LOGS_ENABLED:
    logger.disabled = True


async def send_requests(ch, model_name, payload: InferenceRequest):
    grpc_stub = dataplane.GRPCInferenceServiceStub(ch)

    inference_request_g = converters.ModelInferRequestConverter.from_types(
        payload, model_name=model_name, model_version=None
    )
    response = await grpc_stub.ModelInfer(request=inference_request_g, metadata=[])
    return response


async def model_infer(model_name, request_input: InferenceRequest):
    if not LOGS_ENABLED:
        logger.disabled = True
    try:
        inputs = request_input.outputs[0]
        # logger.info(f"second node {model_name} data extracted!")
    except:
        inputs = request_input.inputs[0]
        # logger.info(f"first node {model_name} data extracted!")
    payload_input = InferenceRequest(inputs=[inputs])
    endpoint = f"{model_name}-{model_name}.default.svc.cluster.local:9500"
    async with grpc.aio.insecure_channel(endpoint) as ch:
        output = await send_requests(ch, model_name, payload_input)
    inference_response = converters.ModelInferResponseConverter.to_types(output)
    return inference_response


class Router(MLModel):
    async def load(self):
        if not LOGS_ENABLED:
            logger.disabled = True
        self.loaded = False
        self.request_counter = 0
        logger.info("Router loaded")
        mlserver.register(
            name="input_requests", description="Measuring number of input requests"
        )
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if not LOGS_ENABLED:
            logger.disabled = True
        mlserver.log(input_requests=1)

        # injecting router arrival time to the message
        arrival_time = time.time()
        pipeline_arrival = {"pipeline_arrival": str(arrival_time)}
        existing_paramteres = payload.inputs[0].parameters
        payload.inputs[0].parameters = existing_paramteres.copy(update=pipeline_arrival)
        self.request_counter += 1
        # logger.info(f"paramters: {payload.inputs[0].parameters}")
        # logger.info(f"Request counter:\n{self.request_counter}\n")

        drop_limit_exceed_payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="drop-limit-violation",
                    shape=[1],
                    datatype="BYTES",
                    data=[],
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="text"),
        )

        output = payload
        for node_index, model_name in enumerate(MODEL_LISTS):
            # logger.info(f"Getting inference responses {model_name}")
            output = await model_infer(model_name=model_name, request_input=output)
            if output.outputs[0].name == "drop-limit-violation":
                # logger.info(f"previous step:\n{self.decode(output.outputs[0])}")
                # if "early exit" in self.decode(output.outputs[0]):
                # logger.info(f"early exiting from before")
                return output
            existing_paramteres = output.outputs[0].parameters
            output.outputs[0].parameters = existing_paramteres.copy(
                update=pipeline_arrival
            )
            time_so_far = time.time() - arrival_time
            # logger.info(f"{model_name} time_so_far:\n{time_so_far}")
            # TODO add the logic of to drop here
            if time_so_far >= DROP_LIMIT and node_index + 1 != len(MODEL_LISTS):
                drop_message = f"early exit, drop limit exceeded after {model_name.replace('queue-', '')}".encode(
                    "utf8"
                )
                # logger.info("early exit from here")
                drop_limit_exceed_payload.outputs[0].data = [drop_message]
                return drop_limit_exceed_payload

        serving_time = time.time()
        times = {PREDICTIVE_UNIT_ID: {"arrival": arrival_time, "serving": serving_time}}
        # logger.info(f"times: {output.outputs[0].parameters.times}")
        model_times: Dict = eval(eval(output.outputs[0].parameters.times)[0])
        model_times.update(times)
        output_times = str([str(model_times)])
        output.outputs[0].parameters.times = output_times

        return output

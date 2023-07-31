import os
from mlserver import MLModel
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
)
from mlserver import MLModel
import grpc
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.grpc.converters as converters
import mlserver
import time
from typing import Dict, List


try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "queue"
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
    MODEL_NAME = os.environ["MODEL_NAME"]
    logger.info(f"MODEL_NAME set to: {MODEL_NAME}")
except KeyError as e:
    raise ValueError("No model is assigned to this queue")

try:
    LAST_NODE = bool(os.environ["LAST_NODE"])
    logger.info(f"LAST_NODE set to: {LAST_NODE}")
except KeyError as e:
    LAST_NODE = False
    logger.info(f"LAST_NODE env variable not set, using default value: {LAST_NODE}")

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


async def model_infer(model_name, request_input: InferenceRequest) -> InferenceResponse:
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


class Queue(MLModel):
    async def load(self):
        if not LOGS_ENABLED:
            logger.disabled = True
        self.loaded = False
        self.request_counter = 0
        mlserver.register(
            name="batch_size", description="Measuring size of the the queue"
        )
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if not LOGS_ENABLED:
            logger.disabled = True
        batch_shape = payload.inputs[0].shape[0]
        logger.info(f"batch_size: {batch_shape}")
        mlserver.log(batch_size=batch_shape)
        arrival_time = time.time()
        self.request_counter += 1
        logger.info(f"Request counter:\n{self.request_counter}\n")

        # early exit logic
        drop_message = (
            f"early exit, drop limit exceeded on {PREDICTIVE_UNIT_ID}".encode("utf8")
        )
        drop_limit_exceed_payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="drop-limit-violation",
                    shape=[batch_shape],
                    datatype="BYTES",
                    data=[drop_message] * batch_shape,
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="text"),
        )
        if payload.inputs[0].shape[0] == 1:
            pipeline_arrival = float(payload.inputs[0].parameters.pipeline_arrival)
        else:
            # in the bigger than one it is already a list so
            # it only need to be serialzed into a string list
            pipeline_arrival = list(
                map(lambda l: float(l), payload.inputs[0].parameters.pipeline_arrival)
            )
            pipeline_arrival = min(pipeline_arrival)

        # early exit before the model
        time_so_far = time.time() - pipeline_arrival
        logger.info(f"time_so_far:\n{time_so_far}")
        if time_so_far >= DROP_LIMIT:
            return drop_limit_exceed_payload

        try:
            # only image and audio model has this attributes
            if payload.inputs[0].shape[0] == 1:
                payload.inputs[0].parameters.datashape = str(
                    [payload.inputs[0].parameters.datashape]
                )
                payload.inputs[0].parameters.dtype = str(
                    [payload.inputs[0].parameters.dtype]
                )
            else:
                payload.inputs[0].parameters.datashape = str(
                    payload.inputs[0].parameters.datashape
                )
                payload.inputs[0].parameters.dtype = str(
                    payload.inputs[0].parameters.dtype
                )
        # not all nodes have the times metadata
        except AttributeError:
            pass
        try:
            # serialize times into strings to be passable to the next stage
            # in the batch size of one this is recieved as a list
            # next step expects a list of strings
            # so in the batch size of we should make a string of list
            if payload.inputs[0].shape[0] == 1:
                payload.inputs[0].parameters.times = str(
                    [payload.inputs[0].parameters.times]
                )
            else:
                # in the bigger than one it is already a list so
                # it only need to be serialzed into a string list
                payload.inputs[0].parameters.times = str(
                    payload.inputs[0].parameters.times
                )
        # first nodes do have the times metadata
        except AttributeError:
            pass

        # patch pipeline arrival for the model container
        pipeline_arrival_models = {"pipeline_arrival": str(pipeline_arrival)}
        existing_paramteres = payload.inputs[0].parameters
        # logger.info(f"existing parameters: {existing_paramteres}")
        payload.inputs[0].parameters = existing_paramteres.copy(
            update=pipeline_arrival_models
        )
        # logger.info(f"after patching parameters: {payload.inputs[0].parameters}")

        output = await model_infer(model_name=MODEL_NAME, request_input=payload)

        # early exit after the model
        time_so_far = time.time() - pipeline_arrival
        logger.info(f"time_so_far:\n{time_so_far}")
        if time_so_far >= DROP_LIMIT:
            logger.info(
                f"returning results, post model violation:\n{drop_limit_exceed_payload}"
            )
            return drop_limit_exceed_payload

        if output.outputs[0].shape[0] == 1:
            if LAST_NODE:
                if self._settings.max_batch_size == 1:
                    pass
                # if it is the last node then the outputs metadata
                # should be deceralized as a list
                else:
                    output.outputs[0].parameters.times = eval(
                        output.outputs[0].parameters.times
                    )
            elif self._settings.max_batch_size == 1:
                output.outputs[0].parameters.times = str(
                    output.outputs[0].parameters.times
                )
            else:
                output.outputs[0].parameters.times = eval(
                    output.outputs[0].parameters.times
                )
        else:
            output.outputs[0].parameters.times = eval(
                output.outputs[0].parameters.times
            )

        # datashpae output descrilizing
        try:
            if output.outputs[0].shape[0] == 1:
                if LAST_NODE:
                    # if it is the last node then the outputs metadata
                    # should be deceralized as a list
                    if self._settings.max_batch_size == 1:
                        pass
                    else:
                        output.outputs[0].parameters.datashape = eval(
                            output.outputs[0].parameters.datashape
                        )
                else:
                    output.outputs[0].parameters.datashape = str(
                        output.outputs[0].parameters.datashape
                    )
            else:
                output.outputs[0].parameters.datashape = eval(
                    output.outputs[0].parameters.datashape
                )
        # not all pipelines have the datashape metadata
        except AttributeError:
            pass
        serving_time = time.time()
        times = {PREDICTIVE_UNIT_ID: {"arrival": arrival_time, "serving": serving_time}}
        # logger.info(output)
        # logger.info(f'raw: {output.outputs[0].parameters.times}')
        # logger.info(f'first eval: {eval(output.outputs[0].parameters.times[0])}')
        if output.outputs[0].shape[0] == 1:
            if type(output.outputs[0].parameters.times) == list:
                model_times: Dict = eval(output.outputs[0].parameters.times[0])
                model_times.update(times)
                output_times = [str(model_times)]
            else:
                model_times: Dict = eval(eval(output.outputs[0].parameters.times)[0])
                model_times.update(times)
                output_times = str([str(model_times)])
                # logger.info(f"output times 2: {output_times}")
            output.outputs[0].parameters.times = output_times
        else:
            model_times: List[Dict] = list(
                map(lambda l: eval(l), output.outputs[0].parameters.times)
            )
            for model_time in model_times:
                model_time.update(times)
            output_times = list(map(lambda l: str(l), model_times))
            output.outputs[0].parameters.times = output_times
        return output

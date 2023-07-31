import os
import time
from mlserver import MLModel
import numpy as np
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
)
from mlserver import MLModel
from typing import List

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "node-one"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )


def decode_from_bin(
    inputs: List[bytes], shapes: List[List[int]], dtypes: List[str]
) -> List[np.array]:
    batch = []
    for input, shape, dtype in zip(inputs, shapes, dtypes):
        buff = memoryview(input)
        array = np.frombuffer(buff, dtype=dtype).reshape(shape)
        batch.append(array)
    return batch


def model(inputs: List[np.ndarray]):
    logger.info(f"unprocessed inputs: {inputs}")
    return list(map(lambda l: {"text": "text"}, inputs))


class NodeOne(MLModel):
    async def load(self):
        self.loaded = False
        logger.info("Loading the ML models")
        logger.info(f"max_batch_size: {self._settings.max_batch_size}")
        logger.info(f"max_batch_time: {self._settings.max_batch_time}")
        self.request_counter = 0
        self.batch_counter = 0
        self.model = model
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        logger.info(f"payload:\n{payload}")
        arrival_time = time.time()
        for request_input in payload.inputs:
            dtypes = request_input.parameters.dtype
            shapes = request_input.parameters.datashape
            batch_shape = request_input.shape[0]
            # batch one edge case
            if type(shapes) != list:
                shapes = [shapes]
            input_data = request_input.data.__root__
            logger.info(f"shapes:\n{shapes}")
            shapes = list(map(lambda l: eval(l), shapes))
            X = decode_from_bin(inputs=input_data, shapes=shapes, dtypes=dtypes)
        received_batch_len = len(X)
        logger.info(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        logger.info(f"to the model:\n{type(X)}")
        logger.info(f"type of the to the model:\n{type(X)}")
        logger.info(f"len of the to the model:\n{len(X)}")

        # model part
        output = self.model(X)
        logger.info(f"unprocessed ouptut: {output}")
        output = list(map(lambda l: l["text"], output))
        logger.info(f"model output:\n{output}")

        # times processing
        serving_time = time.time()
        times = {PREDICTIVE_UNIT_ID: {"arrival": arrival_time, "serving": serving_time}}
        batch_times = [str(times)] * batch_shape
        if self.settings.max_batch_size == 1:
            batch_times = str(batch_times)
        logger.info(f"batch shapes:\n{batch_shape}")
        logger.info(f"batch_times:\n{batch_times}")

        # processing inference response
        output_data = list(map(lambda l: l.encode("utf8"), output))
        payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="text",
                    shape=[batch_shape],
                    datatype="BYTES",
                    data=output_data,
                    parameters=Parameters(times=batch_times),
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="text"),
        )
        logger.info(f"request counter:\n{self.request_counter}\n")
        logger.info(f"batch counter:\n{self.batch_counter}\n")
        return payload

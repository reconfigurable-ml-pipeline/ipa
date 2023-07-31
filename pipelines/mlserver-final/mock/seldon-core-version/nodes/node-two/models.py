import os
import time
import numpy as np
from typing import List
from mlserver import MLModel
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
)
from mlserver import MLModel

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "node-two"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )


def model(inputs: List[np.ndarray]):
    logger.info(f"unprocessed inputs: {inputs}")
    return list(map(lambda l: l["question"], inputs))


class NodeTwo(MLModel):
    async def load(self):
        self.loaded = False
        logger.info("Loading the ML models")
        logger.info(f"max_batch_size: {self._settings.max_batch_size}")
        logger.info(f"max_batch_time: {self._settings.max_batch_time}")
        self.request_counter = 0
        self.batch_counter = 0
        self.model = model
        self.loaded = True
        logger.info("model loading complete!")
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            # check if this is a first node load
            if "times" in dir(request_input.parameters):
                prev_nodes_times = request_input.parameters.times
                # workaround for batch size of one
                if type(prev_nodes_times) == str:
                    logger.info(f"prev_nodes_times:\n{prev_nodes_times}")
                    prev_nodes_times = [eval(eval(prev_nodes_times)[0])]
                else:
                    prev_nodes_times = list(
                        map(lambda l: eval(eval(l)[0]), prev_nodes_times)
                    )
            batch_shape = request_input.shape[0]
            X = request_input.data.__root__
            X = list(map(lambda l: l.decode(), X))
            X = list(map(lambda l: {"question": l}, X))
        logger.info(f"recieved batch len:\n{batch_shape}")
        self.request_counter += batch_shape
        self.batch_counter += 1
        logger.info(f"to the model:\n{type(X)}")
        logger.info(f"type of the to the model:\n{type(X)}")
        logger.info(f"len of the to the model:\n{len(X)}")

        # model part
        output = self.model(X)
        if type(output) == dict:
            output = [output]
        output = list(map(lambda l: str(l), output))
        logger.info(f"model output:\n{output}")

        # times processing
        serving_time = time.time()
        times = {PREDICTIVE_UNIT_ID: {"arrival": arrival_time, "serving": serving_time}}

        if "times" in dir(request_input.parameters):
            this_node_times = [times] * batch_shape
            times = []
            for this_node_time, prev_nodes_time in zip(
                this_node_times, prev_nodes_times
            ):
                this_node_time.update(prev_nodes_time)
                times.append(this_node_time)
            batch_times = list(map(lambda l: str(l), times))
        else:
            batch_times = [str(times)] * batch_shape
        if self.settings.max_batch_size == 1:
            batch_times = str(batch_times)

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

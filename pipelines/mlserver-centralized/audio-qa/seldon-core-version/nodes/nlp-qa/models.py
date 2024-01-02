import os
import time
from copy import deepcopy
from mlserver import MLModel
import torch
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
)
from mlserver import MLModel
from transformers import pipeline

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "nlp-qa"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )

try:
    USE_THREADING = os.getenv("USE_THREADING", "False").lower() in ("true", "1", "t")
    logger.info(f"USE_THREADING set to: {USE_THREADING}")
except KeyError as e:
    USE_THREADING = False
    logger.info(
        f"USE_THREADING env variable not set, using default value: {USE_THREADING}"
    )

try:
    NUM_INTEROP_THREADS = int(os.environ["NUM_INTEROP_THREADS"])
    logger.info(f"NUM_INTEROP_THREADS set to: {NUM_INTEROP_THREADS}")
except KeyError as e:
    NUM_INTEROP_THREADS = 1
    logger.info(
        f"NUM_INTEROP_THREADS env variable not set, using default value: {NUM_INTEROP_THREADS}"
    )

try:
    NUM_THREADS = int(os.environ["NUM_THREADS"])
    logger.info(f"NUM_THREADS set to: {NUM_THREADS}")
except KeyError as e:
    NUM_THREADS = 1
    logger.info(f"NUM_THREADS env variable not set, using default value: {NUM_THREADS}")

if USE_THREADING:
    torch.set_num_interop_threads(NUM_INTEROP_THREADS)
    torch.set_num_threads(NUM_THREADS)

try:
    LOGS_ENABLED = os.getenv("LOGS_ENABLED", "True").lower() in ("true", "1", "t")
    logger.info(f"LOGS_ENABLED set to: {LOGS_ENABLED}")
except KeyError as e:
    LOGS_ENABLED = True
    logger.info(
        f"LOGS_ENABLED env variable not set, using default value: {LOGS_ENABLED}"
    )

try:
    WITH_MODELS = os.getenv("WITH_MODELS", "False").lower() in ("true", "1", "t")
    logger.info(f"WITH_MODELS set to: {WITH_MODELS}")
except KeyError as e:
    WITH_MODELS = False
    logger.info(
        f"USE_THREADING env variable not set, using default value: {WITH_MODELS}"
    )

if not LOGS_ENABLED:
    logger.disabled = True


class GeneralNLP(MLModel):
    async def load(self):
        if not LOGS_ENABLED:
            logger.disabled = True
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        try:
            self.MODEL_VARIANT = os.environ["MODEL_VARIANT"]
            logger.info(f"MODEL_VARIANT set to: {self.MODEL_VARIANT}")
        except KeyError as e:
            self.MODEL_VARIANT = "distilbert-base-uncased"
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}"
            )
        try:
            self.TASK = os.environ["TASK"]
            logger.info(f"TASK set to: {self.TASK}")
        except KeyError as e:
            self.TASK = "question-answering"
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.TASK}"
            )
        try:
            self.CONTEXT = os.environ["CONTEXT"]
            logger.info(f"CONTEXT set to: {self.CONTEXT}")
        except KeyError as e:
            self.CONTEXT = "default context"
            logger.info(
                f"CONTEXT env variable not set, using default value: {self.CONTEXT}"
            )
        logger.info("Loading the ML models")
        logger.info(f"max_batch_size: {self._settings.max_batch_size}")
        logger.info(f"max_batch_time: {self._settings.max_batch_time}")
        if WITH_MODELS:
            model_path = os.path.join(".", "models", self.MODEL_VARIANT)
        else:
            model_path = os.path.join("/", "mnt", "models", self.MODEL_VARIANT)
        self.model = pipeline(
            task=self.TASK,
            model=model_path,
            batch_size=self._settings.max_batch_size,
        )
        self.loaded = True
        logger.info("model loading complete!")
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if not LOGS_ENABLED:
            logger.disabled = True
        if self.loaded == False:
            self.load()
        arrival_time = time.time()
        for request_input in payload.inputs:
            prev_nodes_times = request_input.parameters.times
            logger.info(f"prev_nodes_times: {prev_nodes_times}")
            logger.info(f"prev_nodes_times types: {type(prev_nodes_times)}")
            prev_nodes_times = eval(prev_nodes_times)
            prev_nodes_times = list(map(lambda l: eval(eval(l)[0]), prev_nodes_times))
            batch_shape = request_input.shape[0]
            X = request_input.data.__root__
            X = list(map(lambda l: l.decode(), X))
            X = list(map(lambda l: {"question": l, "context": self.CONTEXT}, X))
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
        this_node_times = [times] * batch_shape
        times = []
        for this_node_time, prev_nodes_time in zip(this_node_times, prev_nodes_times):
            this_node_time.update(prev_nodes_time)
            times.append(this_node_time)
        batch_times = list(map(lambda l: str(l), times))
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

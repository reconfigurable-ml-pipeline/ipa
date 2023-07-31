import os
import torch
import time
import json
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
from copy import deepcopy
from typing import List

# import torchvision
# import sys
# sys.path.insert(0, './cache/ultralytics_yolov5_master')
# sys.path.insert(0, os.path.join(os.getcwd(), '/cache'))

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "yolo"
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

if not LOGS_ENABLED:
    logger.disabled = True


class Yolo(MLModel):
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
            self.MODEL_VARIANT = "yolov5s"
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}"
            )
        try:
            logger.info("Loading the ML models")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info(f"max_batch_size: {self._settings.max_batch_size}")
            logger.info(f"max_batch_time: {self._settings.max_batch_time}")
            # self.model = lambda l: l
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=os.path.join(os.getenv("MODEL_PATH"), f"{self.MODEL_VARIANT}.pt"),
            )
            logger.info("model loaded!")
            self.loaded = True
            logger.info("model loading complete!")
        except OSError:
            raise ValueError("model loading unsuccessful")
        self.loaded = True
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if not LOGS_ENABLED:
            logger.disabled = True
        if self.loaded == False:
            self.load()
        arrival_time = time.time()

        for request_input in payload.inputs:
            dtypes = request_input.parameters.dtype
            shapes = request_input.parameters.datashape
            batch_shape = request_input.shape[0]
            # batch one edge case
            if type(shapes) != list:
                shapes = [shapes]
                dtypes = [dtypes]
            input_data = request_input.data.__root__
            logger.info(f"shapes:\n{shapes}")
            shapes = list(map(lambda l: eval(l), shapes))
            X = decode_from_bin(inputs=input_data, shapes=shapes, dtypes=dtypes)
        received_batch_len = len(X)
        logger.info(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        logger.info(f"type of the to the model:\n{type(X)}")
        logger.info(f"len of the to the model:\n{len(X)}")
        objs = self.model(X)
        output = self.get_cropped(objs)
        # logger.info(f"model output:\n{output}")
        serving_time = time.time()
        times = {PREDICTIVE_UNIT_ID: {"arrival": arrival_time, "serving": serving_time}}
        batch_times = [str(times)] * batch_shape
        dtypes = ["u1"] * batch_shape
        # TEMP Currently considering only one person per pic, zero index
        person_pics = list(map(lambda l: l["person"][0], output))
        datashape = list(map(lambda l: list(l.shape), person_pics))
        output_data = list(map(lambda l: l.tobytes(), person_pics))
        if self.settings.max_batch_size == 1:
            batch_times = str(batch_times)
            dtypes = str(dtypes)
            datashape = str(datashape)
        payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="image",
                    shape=[batch_shape],
                    datatype="BYTES",
                    data=output_data,
                    parameters=Parameters(
                        dtype=dtypes, times=batch_times, datashape=datashape
                    ),
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="image"),
        )
        # payload_to_print = payload.outputs[0].data
        # logger.info(payload)
        logger.info(f"request counter:\n{self.request_counter}\n")
        logger.info(f"batch counter:\n{self.batch_counter}\n")
        return payload

    def get_cropped(self, result):
        """
        crops selected objects for
        the subsequent nodes
        """
        output_list = []
        for res in result.tolist():
            res = res.crop(save=False)
            liscense_labels = ["car", "truck"]
            car_labels = ["car"]
            person_labels = ["person"]
            res_output = {"person": [], "car": [], "liscense": []}
            for obj in res:
                for label in liscense_labels:
                    if label in obj["label"]:
                        res_output["liscense"].append(deepcopy(obj["im"]))
                        break
                for label in car_labels:
                    if label in obj["label"]:
                        res_output["car"].append(deepcopy(obj["im"]))
                        break
                for label in person_labels:
                    if label in obj["label"]:
                        res_output["person"].append(deepcopy(obj["im"]))
                        break
            output_list.append(res_output)
        return output_list

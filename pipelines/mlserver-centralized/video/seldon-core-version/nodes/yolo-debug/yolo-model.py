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
import logging

try:
    POD_NAME = os.environ["POD_NAME"]
    logger.info(f"POD_NAME set to: {POD_NAME}")
except KeyError as e:
    POD_NAME = "yolo"
    logger.info(f"POD_NAME env variable not set, using default value: {POD_NAME}")

# File number of files in the folder
num_files = int(len(os.listdir("./logs")) / 2)

# Add the file handler to mlserver logs
log_file_path = f"./logs/{num_files}_log_{POD_NAME}.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Add the file handler for error logs
logger_error = logging.getLogger()
error_log_file_path = f"./logs/{num_files}_error_log_{POD_NAME}.log"
error_file_handler = logging.FileHandler(error_log_file_path)
error_file_handler.setLevel(logging.ERROR)
logger_error.addHandler(error_file_handler)

# Add a stream handler for error logs to stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger_error.addHandler(stream_handler)

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
    USE_THREADING = bool(os.environ["USE_THREADING"])
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


class Yolo(MLModel):
    async def load(self):
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
        if self.loaded == False:
            self.load()
        arrival_time = time.time()

        for request_input in payload.inputs:
            dtypes: List[str] = eval(request_input.parameters.dtype)
            logger.info(f"dtypes: {dtypes}")
            shapes: List[str] = request_input.parameters.datashape
            logger.info(f"dtypes: {shapes}")
            batch_shape = request_input.shape[0]
            # batch one edge case
            if type(shapes) != list:
                shapes = [shapes]
            input_data = request_input.data.__root__
            logger.info(f"shapes:\n{shapes}")
            shapes = list(map(lambda l: eval(l), eval(shapes[0])))
            X = decode_from_bin(inputs=input_data, shapes=shapes, dtypes=dtypes)

        received_batch_len = len(X)
        logger.info(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += received_batch_len
        self.batch_counter += 1
        logger.info(f"type of the to the model:\n{type(X)}")
        logger.info(f"len of the to the model:\n{len(X)}")
        objs = self.model(X)
        output = self.get_cropped(objs)
        serving_time = time.time()
        times = {PREDICTIVE_UNIT_ID: {"arrival": arrival_time, "serving": serving_time}}
        batch_times = [str(times)] * batch_shape
        dtypes = ["u1"] * batch_shape
        # TEMP Currently considering only one person per pic, zero index
        person_pics = list(map(lambda l: l["person"][0], output))
        datashape = list(map(lambda l: list(l.shape), person_pics))
        output_data = list(map(lambda l: l.tobytes(), person_pics))
        batch_times = str(batch_times)
        dtypes = str(dtypes)
        datashape = str(datashape)

        logger.info(f"output batch times: {batch_times}")
        logger.info(f"output dtpes: {dtypes}")
        logger.info(f"output datashapes: {datashape}")

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

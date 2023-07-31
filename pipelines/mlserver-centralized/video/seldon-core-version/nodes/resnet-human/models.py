import os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
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

# TODO balooning has not been implemented yet

try:
    PREDICTIVE_UNIT_ID = os.environ["PREDICTIVE_UNIT_ID"]
    logger.info(f"PREDICTIVE_UNIT_ID set to: {PREDICTIVE_UNIT_ID}")
except KeyError as e:
    PREDICTIVE_UNIT_ID = "resnet-human"
    logger.info(
        f"PREDICTIVE_UNIT_ID env variable not set, using default value: {PREDICTIVE_UNIT_ID}"
    )

try:
    DROP_LIMIT = float(os.environ["DROP_LIMIT"])
    logger.info(f"DROP_LIMIT set to: {DROP_LIMIT}")
except KeyError as e:
    DROP_LIMIT = 1000
    logger.info(f"DROP_LIMIT env variable not set, using default value: {DROP_LIMIT}")


def decode_from_bin(
    inputs: List[bytes],
    shapes: List[List[int]],
    dtypes: List[str],
    default_shape: List[int],
) -> List[np.array]:
    batch = []
    for input, shape, dtype in zip(inputs, shapes, dtypes):
        buff = memoryview(input)
        array = np.frombuffer(buff, dtype=dtype).reshape(shape[0])
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


class ResnetHuman(MLModel):
    async def load(self) -> bool:
        if not LOGS_ENABLED:
            logger.disabled = True
        self.loaded = False
        self.request_counter = 0
        self.batch_counter = 0
        # standard resnet image transformation
        try:
            self.MODEL_VARIANT = os.environ["MODEL_VARIANT"]
            logger.info(f"MODEL_VARIANT set to: {self.MODEL_VARIANT}")
        except KeyError as e:
            self.MODEL_VARIANT = "resnet18"
            logger.info(
                f"MODEL_VARIANT env variable not set, using default value: {self.MODEL_VARIANT}"
            )
        logger.info(f"max_batch_size: {self._settings.max_batch_size}")
        logger.info(f"max_batch_time: {self._settings.max_batch_time}")
        self.batch_size = self._settings.max_batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        logger.info("Init function complete!")
        model = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        logger.error("Loading the ML models")
        # TODO cpu and gpu from env variable
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.default_shape = [253, 294, 3]
        self.resnet = model[self.MODEL_VARIANT](pretrained=True)
        self.resnet.eval()
        self.loaded = True
        logger.info("model loading complete!")
        return self.loaded

    async def predict(self, payload: InferenceRequest) -> InferenceRequest:
        if not LOGS_ENABLED:
            logger.disabled = True
        if self.loaded == False:
            self.load()
        arrival_time = time.time()

        for request_input in payload.inputs:
            batch_shape = request_input.shape[0]
            prev_nodes_times = request_input.parameters.times
            logger.info(f"prev_nodes_times: {prev_nodes_times}")
            logger.info(f"prev_nodes_times types: {type(prev_nodes_times)}")
            prev_nodes_times = eval(prev_nodes_times)
            prev_nodes_times = list(map(lambda l: eval(eval(l)[0]), prev_nodes_times))
            shapes = request_input.parameters.datashape
            dtypes = batch_shape * ["u1"]  # TEMP HACK

            logger.info(f"input shapes: {shapes}")
            logger.info(f"input shapes type: {type(shapes)}")
            input_data = request_input.data.__root__

            shapes = list(map(lambda l: eval(l), eval(shapes)))
            logger.info(f"output shapes:\n{shapes}")
            logger.info(f"shapes:\n{shapes}")
            input_data = request_input.data.__root__
            X = decode_from_bin(
                inputs=input_data,
                shapes=shapes,
                dtypes=dtypes,
                default_shape=self.default_shape,
            )
            pipeline_arrival = float(request_input.parameters.pipeline_arrival)

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
        time_so_far = time.time() - pipeline_arrival
        logger.info(f"time_so_far:\n{time_so_far}")
        if time_so_far >= DROP_LIMIT:
            return drop_limit_exceed_payload

        received_batch_len = len(X)
        logger.info(f"recieved batch len:\n{received_batch_len}")
        self.request_counter += batch_shape
        self.batch_counter += 1
        # preprocessings
        converted_images = [
            Image.fromarray(np.array(image, dtype=np.uint8)) for image in X
        ]
        X_trans = [
            self.transform(converted_image) for converted_image in converted_images
        ]
        batch = torch.stack(X_trans, axis=0)

        out = self.resnet(batch)
        percentages = torch.nn.functional.softmax(out, dim=1) * 100
        percentages = percentages.detach().numpy()
        image_net_class = np.argmax(percentages, axis=1)
        output = image_net_class.tolist()
        # logger.info(f"{image_net_class=}")

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
        payload = InferenceResponse(
            outputs=[
                ResponseOutput(
                    name="int",
                    shape=[batch_shape],
                    datatype="INT32",
                    data=output,
                    parameters=Parameters(times=batch_times, content_type="np"),
                )
            ],
            model_name=self.name,
            parameters=Parameters(type_of="int"),
        )
        logger.info(f"request counter:\n{self.request_counter}\n")
        logger.info(f"batch counter:\n{self.batch_counter}\n")
        return payload

import os
import sys
import time
import yaml
import numpy as np
from typing import List, Tuple, Dict, Any
import re
from jinja2 import Environment, FileSystemLoader
from PIL import Image
import asyncio
import subprocess
from datasets import load_dataset
import tqdm
from multiprocessing import Queue

from barazmoon import Data
from barazmoon import MLServerAsyncGrpc

from kubernetes import config
from kubernetes import client

try:
    config.load_kube_config()
    kube_config = client.Configuration().get_default_copy()
except AttributeError:
    kube_config = client.Configuration()
    kube_config.assert_hostname = False
client.Configuration.set_default(kube_config)
kube_api = client.api.core_v1_api.CoreV1Api()

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))
from experiments.utils.constants import NAMESPACE, ROUTER_PATH, QUEUE_PATH

from experiments.utils import logger


def get_cpu_model_name():
    command = "lscpu"
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout.strip()
    lines = output.split("\n")

    model_name = None
    for line in lines:
        if line.startswith("Model name:"):
            model_name = line.split(":", 1)[1].strip()
            break

    return model_name


def get_pod_name(node_name: str, orchestrator=False):
    pod_regex = f"{node_name}.*"
    pods_list = kube_api.list_namespaced_pod(NAMESPACE)
    pod_names = []
    for pod_name in pods_list.items:
        pod_name = pod_name.metadata.name
        if orchestrator and re.match(pod_regex, pod_name) and "svc" in pod_name:
            return pod_name
        if re.match(pod_regex, pod_name) and "svc" not in pod_name:
            pod_names.append(pod_name)
    return pod_names


def setup_node(
    node_name: str,
    cpu_request: str,
    memory_request: str,
    model_variant: str,
    max_batch_size: str,
    max_batch_time: str,
    replica: int,
    node_path: str,
    use_threading: bool,
    num_interop_threads: int,
    num_threads: int,
    distrpution_time: int,
    from_storage=bool,
    no_engine=False,
    debug_mode=False,
    drop_limit=1000,
    logs_enabled: bool = True,
):
    # TODO HERE add if else here to check with model or not
    logger.info("-" * 25 + " setting up the node with following config" + "-" * 25)
    logger.info("\n")
    # central pipeline case
    if max_batch_size is None and max_batch_time is None:
        svc_vars = {
            "name": node_name,
            "cpu_request": cpu_request,
            "memory_request": memory_request,
            "cpu_limit": cpu_request,
            "memory_limit": memory_request,
            "model_variant": model_variant,
            "replicas": replica,
            "no_engine": str(no_engine),
            "distrpution_time": distrpution_time,
            "use_threading": use_threading,
            "num_interop_threads": num_interop_threads,
            "num_threads": num_threads,
            "drop_limit": drop_limit,
            "logs_enabled": logs_enabled,
        }
    else:
        # profiling pipeline case
        svc_vars = {
            "name": node_name,
            "cpu_request": cpu_request,
            "memory_request": memory_request,
            "cpu_limit": cpu_request,
            "memory_limit": memory_request,
            "model_variant": model_variant,
            "max_batch_size": max_batch_size,
            "max_batch_time": max_batch_time,
            "replicas": replica,
            "distrpution_time": distrpution_time,
            "no_engine": str(no_engine),
            "use_threading": use_threading,
            "num_interop_threads": num_interop_threads,
            "num_threads": num_threads,
            "drop_limit": drop_limit,
            "logs_enabled": logs_enabled,
        }
    environment = Environment(loader=FileSystemLoader(node_path))
    if from_storage:
        template_file_name = "node-template.yaml"
    else:
        template_file_name = "node-template-with-model.yaml"
    svc_template = environment.get_template(template_file_name)
    content = svc_template.render(svc_vars)
    logger.info(content)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
        """
    os.system(command)
    logger.info("-" * 25 + f" waiting to make sure the node is up " + "-" * 25)
    logger.info("\n")
    check_node_loaded(node_name=node_name)


def setup_router(
    pipeline_name: str,
    node_names: List[str],
    distrpution_time: int,
    debug_mode: bool = False,
    drop_limit: int = 1000,
    logs_enabled: bool = True,
):
    logger.info("-" * 25 + " setting up the node with following config" + "-" * 25)
    logger.info("\n")
    if debug_mode:
        node_names = list(map(lambda l: f"{l}-debug", node_names))
    model_lists = str(node_names)
    model_lists = model_lists.replace("'", '"')
    svc_vars = {
        "name": pipeline_name,
        "cpu_request": 4,
        "memory_request": "8Gi",
        "cpu_limit": 16,
        "memory_limit": "32Gi",
        "replicas": 1,
        "distrpution_time": distrpution_time,
        "model_lists": model_lists,
        "drop_limit": drop_limit,
        "logs_enabled": logs_enabled,
    }
    environment = Environment(loader=FileSystemLoader(ROUTER_PATH))
    svc_template = environment.get_template("node-template.yaml")
    content = svc_template.render(svc_vars)
    logger.info(content)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
        """
    os.system(command)
    logger.info("-" * 25 + f" waiting to make sure the node is up " + "-" * 25)
    logger.info("\n")
    check_node_loaded(node_name="router")


def setup_queues(
    node_names: List[str],
    max_batch_sizes: int,
    max_batch_times: int,
    distrpution_time: int,
    logs_enabled: bool,
    debug_mode: bool = False,
    drop_limit: int = 1000,
):
    # TODO
    # in a for loop
    for node_index, model_name in enumerate(node_names):
        last_node = "True" if node_index + 1 == len(model_name) else "False"
        max_batch_size = max_batch_sizes[node_index]
        max_batch_time = max_batch_times[node_index]
        setup_queue(
            model_name=model_name,
            max_batch_size=max_batch_size,
            max_batch_time=max_batch_time,
            last_node=last_node,
            distrpution_time=distrpution_time,
            debug_mode=debug_mode,
            drop_limit=drop_limit,
            logs_enabled=logs_enabled,
        )


def setup_queue(
    model_name: str,
    max_batch_size: int,
    max_batch_time: int,
    last_node: bool,
    distrpution_time: int,
    debug_mode: bool = False,
    drop_limit: int = 1000,
    logs_enabled: bool = True,
):
    # TODO
    # in a for loop
    logger.info("-" * 25 + " setting up the node with following config" + "-" * 25)
    logger.info("\n")
    model_name = f"{model_name}-debug" if debug_mode else model_name
    svc_vars = {
        "max_batch_size": max_batch_size,
        "max_batch_time": max_batch_time,
        "cpu_request": 4,
        "memory_request": "8Gi",
        "cpu_limit": 16,
        "memory_limit": "32Gi",
        "replicas": 1,
        "distrpution_time": 120,
        "model_name": model_name,
        "last_node": last_node,
        "drop_limit": drop_limit,
        "logs_enabled": logs_enabled,
    }
    queue_path = f"{QUEUE_PATH}-debug" if debug_mode else QUEUE_PATH
    environment = Environment(loader=FileSystemLoader(queue_path))
    svc_template = environment.get_template("node-template.yaml")
    content = svc_template.render(svc_vars)
    logger.info(content)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
        """
    os.system(command)
    logger.info("-" * 25 + f" waiting to make sure the node is up " + "-" * 25)
    logger.info("\n")
    check_node_loaded(node_name="queue-" + model_name)


def setup_seldon_pipeline(
    pipeline_name: str,
    cpu_request: Tuple[str],
    memory_request: Tuple[str],
    model_variant: Tuple[str],
    max_batch_size: Tuple[str],
    max_batch_time: Tuple[str],
    replica: Tuple[int],
    use_threading: Tuple[bool],
    num_interop_threads: Tuple[int],
    num_threads: Tuple[int],
    pipeline_path: str,
    num_nodes: int,
):
    logger.info("-" * 25 + " setting up the node with following config" + "-" * 25)
    logger.info("\n")
    # TODO add num nodes logic here
    svc_vars = {"name": pipeline_name}
    for node_id in range(num_nodes):
        node_index = node_id + 1
        svc_vars.update(
            {
                f"cpu_request_{node_index}": cpu_request[node_id],
                f"memory_request_{node_index}": memory_request[node_id],
                f"cpu_limit_{node_index}": cpu_request[node_id],
                f"memory_limit_{node_index}": memory_request[node_id],
                f"model_variant_{node_index}": model_variant[node_id],
                f"max_batch_size_{node_index}": max_batch_size[node_id],
                f"max_batch_time_{node_index}": max_batch_time[node_id],
                f"replicas_{node_index}": replica[node_id],
                f"use_threading_{node_index}": use_threading[node_id],
                f"num_interop_threads_{node_index}": num_interop_threads[node_id],
                f"num_threads_{node_index}": num_threads[node_id],
            }
        )
    environment = Environment(loader=FileSystemLoader(pipeline_path))
    svc_template = environment.get_template("pipeline-template.yaml")
    content = svc_template.render(svc_vars)
    logger.info(content)
    command = f"""cat <<EOF | kubectl apply -f -
{content}
        """
    os.system(command)
    logger.info("-" * 25 + f" waiting to make sure the node is up " + "-" * 25)
    logger.info("\n")
    logger.info("-" * 25 + f" model pod {pipeline_name} successfuly set up " + "-" * 25)
    logger.info("\n")
    # extract model model container names
    model_container = yaml.safe_load(content)
    model_names = list(
        map(
            lambda l: l["spec"]["containers"][0]["name"],
            model_container["spec"]["predictors"][0]["componentSpecs"],
        )
    )
    # checks if the pods are ready each 5 seconds
    loop_timeout = 5
    while True:
        models_loaded, svc_loaded, pipeline_loaded = False, False, False
        logger.info(f"waited for {loop_timeout} to check if the pods are up")
        time.sleep(loop_timeout)
        model_pods = kube_api.list_namespaced_pod(
            namespace=NAMESPACE, label_selector=f"seldon-deployment-id={pipeline_name}"
        )
        all_model_pods = []
        all_conainers = []
        for pod in model_pods.items:
            if pod.status.phase == "Running":
                all_model_pods.append(True)
                pod_name = pod.metadata.name
                for model_name in model_names:
                    if model_name in pod_name:
                        container_name = model_name
                        break
                logs = kube_api.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=NAMESPACE,
                    container=container_name,
                )
                logger.info(logs)
                if "Uvicorn running on http://0.0.0.0:600" in logs:
                    all_conainers.append(True)
                else:
                    all_conainers.append(False)
            else:
                all_model_pods.append(False)
        logger.info(f"all_model_pods: {all_model_pods}")
        if all(all_model_pods):
            models_loaded = True
        else:
            continue
        logger.info(f"all_containers: {all_conainers}")
        if all(all_model_pods):
            pipeline_loaded = True
        else:
            continue
        svc_pods = kube_api.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"seldon-deployment-id={pipeline_name}-{pipeline_name}",
        )
        for pod in svc_pods.items:
            if pod.status.phase == "Running":
                svc_loaded = True
            for container_status in pod.status.container_statuses:
                if container_status.ready:
                    pipeline_loaded = True
                else:
                    continue
            else:
                continue
        if models_loaded and svc_loaded and pipeline_loaded:
            logger.info("model container completely loaded!")
            break


def setup_router_pipeline(
    pipeline_name: str,
    node_names: List[str],
    cpu_request: Tuple[str],
    memory_request: Tuple[str],
    model_variant: Tuple[str],
    max_batch_size: Tuple[str],
    max_batch_time: Tuple[str],
    replica: Tuple[int],
    use_threading: Tuple[bool],
    num_interop_threads: Tuple[int],
    num_threads: Tuple[int],
    pipeline_path: str,
    timeout: int,
    num_nodes: int,
    distrpution_time: int,
    debug_mode: bool = False,
):
    logger.info("-" * 25 + " setting up the node with following config" + "-" * 25)
    logger.info("\n")
    for node_id, node_name in zip(range(num_nodes), node_names):
        node_path = os.path.join(pipeline_path, "nodes", node_name)
        setup_node(
            node_name=node_name,
            cpu_request=cpu_request[node_id],
            memory_request=memory_request[node_id],
            model_variant=model_variant[node_id],
            max_batch_size=max_batch_size[node_id],
            max_batch_time=max_batch_time[node_id],
            replica=replica[node_id],
            node_path=node_path,
            no_engine=True,
            use_threading=use_threading[node_id],
            # HACK for now we set the number of requests
            # proportional to the the number threads
            num_interop_threads=cpu_request[node_id],
            num_threads=cpu_request[node_id],
            distrpution_time=distrpution_time,
            debug_mode=debug_mode,
        )
    setup_router(
        pipeline_name=pipeline_name,
        node_names=node_names,
        distrpution_time=distrpution_time,
        debug_mode=debug_mode,
    )


def setup_central_pipeline(
    pipeline_name: str,
    node_names: List[str],
    cpu_request: Tuple[str],
    memory_request: Tuple[str],
    model_variant: Tuple[str],
    max_batch_size: Tuple[str],
    max_batch_time: Tuple[str],
    replica: Tuple[int],
    use_threading: Tuple[bool],
    num_interop_threads: Tuple[int],
    num_threads: Tuple[int],
    pipeline_path: str,
    timeout: int,
    num_nodes: int,
    distrpution_time: int,
    from_storage: List[bool],
    debug_mode: bool = False,
    drop_limit: int = 1000,
    logs_enabled: bool = True,
):
    logger.info("-" * 25 + " setting up the node with following config" + "-" * 25)
    logger.info("\n")
    for node_id, node_name in zip(range(num_nodes), node_names):
        node_sub_path = f"{node_name}-debug" if debug_mode else node_name
        node_path = os.path.join(pipeline_path, "nodes", node_sub_path)
        setup_node(
            node_name=node_name,
            cpu_request=cpu_request[node_id],
            memory_request=memory_request[node_id],
            model_variant=model_variant[node_id],
            max_batch_size=None,
            max_batch_time=None,
            replica=replica[node_id],
            node_path=node_path,
            no_engine=True,
            use_threading=use_threading[node_id],
            # HACK for now we set the number of requests
            # proportional to the the number threads
            num_interop_threads=cpu_request[node_id],
            num_threads=cpu_request[node_id],
            distrpution_time=distrpution_time,
            drop_limit=drop_limit,
            logs_enabled=logs_enabled,
            from_storage=from_storage[node_id],
        )
    queue_names = list(map(lambda l: "queue-" + l, node_names))
    setup_queues(
        node_names=node_names,
        max_batch_sizes=max_batch_size,
        max_batch_times=max_batch_time,
        distrpution_time=distrpution_time,
        debug_mode=debug_mode,
        drop_limit=drop_limit,
        logs_enabled=logs_enabled,
    )
    setup_router(
        pipeline_name=pipeline_name,
        node_names=queue_names,
        distrpution_time=distrpution_time,
        debug_mode=debug_mode,
        drop_limit=drop_limit,
        logs_enabled=logs_enabled,
    )


def load_data(data_type: str, pipeline_path: str, node_type: str = "first"):
    if data_type == "audio":
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
        )
        data = ds[0]["audio"]["array"]
        data_shape = [len(data)]
    elif data_type == "text":
        input_sample_path = os.path.join(pipeline_path, "input-sample.txt")
        input_sample_shape_path = os.path.join(pipeline_path, "input-sample-shape.json")
        with open(input_sample_path, "r") as openfile:
            data = openfile.read()
            # with open(input_sample_shape_path, 'r') as openfile:
            #     data_shape = json.load(openfile)
            #     data_shape = data_shape['data_shape']
            data_shape = [1]
    elif data_type == "image":
        input_sample_path = os.path.join(pipeline_path, "input-sample.JPEG")
        data = Image.open(input_sample_path)
        if node_type == "first":
            data_shape = list(np.array(data).shape)
        elif node_type == "second":
            data_shape = [list(np.array(data).shape)]
        else:
            raise ValueError(f"Invalid data type: {node_type}")
        data = np.array(data).flatten()
    # custom_parameters =  "[\"{'yolo': {'arrival': 1680913322.8364007, 'serving': 1680913322.92951}}\"]"
    data_1 = Data(
        data=data,
        data_shape=data_shape
        # custom_parameters={'times': str(custom_parameters)},
    )

    # Data list
    data = []
    data.append(data_1)
    return data


def check_load_test(pipeline_name: str, data_type: str, pipeline_path: str, model: str):
    node_type = "first"
    if pipeline_name == "resnet-human":
        node_type = "second"
    data = load_data(
        data_type=data_type, pipeline_path=pipeline_path, node_type=node_type
    )
    loop_timeout = 5
    while True:
        logger.info(
            f"waited for {loop_timeout} seconds to check for successful request"
        )
        time.sleep(loop_timeout)
        _, _, response = load_test(
            pipeline_name=pipeline_name,
            model=model,
            data=data,
            data_type=data_type,
            workload=[2],
        )
        if "failed" not in response[0][0].keys():
            return True


def warm_up(
    pipeline_name: str,
    data_type: str,
    model: str,
    pipeline_path: str,
    warm_up_duration: int,
):
    node_type = "first"
    if pipeline_name == "resnet-human":
        node_type = "second"
    workload = [1] * warm_up_duration
    data = load_data(
        data_type=data_type, pipeline_path=pipeline_path, node_type=node_type
    )
    load_test(
        pipeline_name=pipeline_name,
        model=model,
        data=data,
        data_type=data_type,
        workload=workload,
    )


def remove_pipeline(pipeline_name: str):
    os.system(f"kubectl delete seldondeployment {pipeline_name} -n default")
    # TEMP TODO until fixing the server problem
    os.system(f"kubectl delete seldondeployment --all -n default")
    os.system(f"kubectl delete deployments --all -n default")
    os.system(f"kubectl delete replicaset --all -n default")
    os.system(f"kubectl delete pods --all -n default")
    os.system(
        "kubectl get services | grep -v kubernetes | awk '{print $1}' | xargs kubectl delete service -n default"
    )
    logger.info("-" * 50 + f" pipeline {pipeline_name} successfuly removed " + "-" * 50)
    logger.info("\n")


def load_test(
    pipeline_name: str,
    data_type: str,
    model: str,
    workload: List[int],
    data: List[Data],
    namespace: str = "default",
    mode: str = "step",
    benchmark_duration=1,
    queue: Queue = None,
) -> Tuple[int, int, List[List[Dict[str, Any]]]]:
    start_time = time.time()

    endpoint = "localhost:32000"
    deployment_name = pipeline_name
    namespace = "default"
    metadata = [("seldon", deployment_name), ("namespace", namespace)]
    load_tester = MLServerAsyncGrpc(
        endpoint=endpoint,
        metadata=metadata,
        workload=workload,
        model=model,
        data=data,
        mode=mode,  # options - step, equal, exponential
        data_type=data_type,
        benchmark_duration=benchmark_duration,
    )
    responses = asyncio.run(load_tester.start())
    end_time = time.time()

    # remove ouput for image inputs/outpus (yolo)
    # as they make logs very heavy
    for second_response in responses:
        for response in second_response:
            response["outputs"] = []
    if queue is not None:
        queue.put([start_time, end_time, responses])
    return start_time, end_time, responses


def check_node_loaded(node_name: str, silent_mode: bool = False) -> bool:
    if not silent_mode:
        logger.info("-" * 25 + f" model pod {node_name} successfuly set up " + "-" * 25)
        logger.info("\n")
    # checks if the pods are ready each 5 seconds
    check_interval = 5
    while True:
        models_loaded, container_loaded = False, False
        model_pods = kube_api.list_namespaced_pod(
            namespace=NAMESPACE, label_selector=f"seldon-deployment-id={node_name}"
        )
        all_model_pods = []
        all_conainers = []
        for pod in model_pods.items:
            if pod.status.phase == "Running":
                all_model_pods.append(True)
                logs = kube_api.read_namespaced_pod_log(
                    name=pod.metadata.name, namespace=NAMESPACE, container=node_name
                )
                if not silent_mode:
                    logger.info(logs)
                if "Uvicorn running on http://0.0.0.0:6000" in logs:
                    all_conainers.append(True)
                else:
                    all_conainers.append(False)
            else:
                all_model_pods.append(False)
        if not silent_mode:
            logger.info(f"all_model_pods: {all_model_pods}")
        if all(all_model_pods) and all_model_pods != []:
            models_loaded = True
        else:
            if not silent_mode:
                logger.info(f"waited for {check_interval} to check if the pods are up")
                for _ in tqdm.tqdm(range(check_interval)):
                    time.sleep(1)
            else:
                time.sleep(check_interval)
        if not silent_mode:
            logger.info(f"all_containers: {all_conainers}")
        if all(all_conainers):
            container_loaded = True
        else:
            if not silent_mode:
                logger.info(f"waited for {check_interval} to check if the pods are up")
                for _ in tqdm.tqdm(range(check_interval)):
                    time.sleep(1)
            else:
                time.sleep(check_interval)
        if models_loaded and container_loaded:
            if not silent_mode:
                logger.info("model container completely loaded!")
            return True
        if not silent_mode:
            logger.info(f"waited for {check_interval} to check if the pods are up")
            for _ in tqdm.tqdm(range(check_interval)):
                time.sleep(1)
        else:
            time.sleep(check_interval)


def check_node_up(node_name: str) -> bool:
    model_pods = kube_api.list_namespaced_pod(
        namespace=NAMESPACE, label_selector=f"seldon-deployment-id={node_name}"
    )
    return model_pods.items != []


def is_terminating(node_name: str) -> bool:
    pods = kube_api.list_namespaced_pod(
        namespace=NAMESPACE, label_selector=f"seldon-deployment-id={node_name}"
    )
    return pods.items[0].status.phase == "Terminating"

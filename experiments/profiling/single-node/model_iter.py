"""
Iterate through all possible combination
of models
"""
import os
import time
import json
import yaml
from typing import Union
import click
import sys
import csv
from tqdm import tqdm
import shutil
from multiprocessing import Queue, Process
from barazmoon.twitter import twitter_workload_generator

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..", "..")))

from experiments.utils.pipeline_operations import (
    load_data,
    warm_up,
    check_load_test,
    load_test,
    remove_pipeline,
    setup_node,
    get_pod_name,
)
from experiments.utils.constants import (
    PIPLINES_PATH,
    NODE_PROFILING_CONFIGS_PATH,
    NODE_PROFILING_RESULTS_PATH,
    KEY_CONFIG_FILENAME,
)

from experiments.utils.prometheus import PromClient

prom_client = PromClient()

from experiments.utils import logger


def experiments(
    pipeline_name: str, node_name: str, config: dict, node_path: str, data_type: str
):
    model_variants = config["model_variants"]
    max_batch_sizes = config["max_batch_size"]
    max_batch_times = config["max_batch_time"]
    cpu_requests = config["cpu_request"]
    memory_requests = config["memory_request"]
    replicas = config["replicas"]
    series = config["series"]
    series_meta = config["series_meta"]
    workload_type = config["workload_type"]
    workload_config = config["workload_config"]

    logs_enabled = config["logs_enabled"]
    distrpution_time = config["distrpution_time"]

    if workload_type == "static":
        loads_to_test = workload_config["loads_to_test"]
        load_duration = workload_config["load_duration"]
    elif workload_type == "twitter":
        loads_to_test = []
        for w_config in workload_config:
            start = w_config["start"]
            end = w_config["end"]
            load_to_test = start + "-" + end
            loads_to_test.append(load_to_test)
        workload = twitter_workload_generator(loads_to_test[0])
        load_duration = len(workload)
    else:
        raise ValueError(f"Invalid workload type: {workload_type}")

    mode = config["mode"]
    benchmark_duration = config["benchmark_duration"]
    use_threading = config["use_threading"]
    # for now set to the number of CPU
    num_interop_threads = config["num_interop_threads"]
    num_threads = config["num_threads"]
    if "no_engine" in config.keys():
        no_engine = config["no_engine"]
    else:
        no_engine = False
    timeout = config["timeout"]
    # TOOD Add cpu type, gpu type
    # TODO Better solution instead of nested for loops
    # TODO Also add the random - maybe just use Tune
    remove_pipeline(pipeline_name=pipeline_name)
    for model_variant in model_variants:
        for max_batch_size in max_batch_sizes:
            for max_batch_time in max_batch_times:
                for cpu_request in cpu_requests:
                    for memory_request in memory_requests:
                        for replica in replicas:
                            # for num_interop_thread in num_interop_threads:
                            #     for num_thread in num_threads:
                            for load in loads_to_test:
                                # for rep in range(repetition):
                                logger.info(
                                    "-" * 25
                                    + f" starting repetition experiment "
                                    + "-" * 25
                                )
                                logger.info("\n")
                                experiments_exist, experiment_id = key_config_mapper(
                                    pipeline_name=pipeline_name,
                                    node_name=node_name,
                                    cpu_request=cpu_request,
                                    memory_request=memory_request,
                                    model_variant=model_variant,
                                    max_batch_size=max_batch_size,
                                    max_batch_time=max_batch_time,
                                    load=load,
                                    load_duration=load_duration,
                                    series=series,
                                    series_meta=series_meta,
                                    replica=replica,
                                    no_engine=no_engine,
                                    mode=mode,
                                    data_type=data_type,
                                    benchmark_duration=benchmark_duration,
                                )
                                if not experiments_exist:
                                    setup_node(
                                        node_name=node_name,
                                        cpu_request=cpu_request,
                                        memory_request=memory_request,
                                        model_variant=model_variant,
                                        max_batch_size=max_batch_size,
                                        max_batch_time=max_batch_time,
                                        replica=replica,
                                        node_path=node_path,
                                        no_engine=no_engine,
                                        use_threading=use_threading,
                                        # HACK for now we set the number of requests
                                        # proportional to the the number threads
                                        num_interop_threads=cpu_request,
                                        num_threads=cpu_request,
                                        distrpution_time=distrpution_time,
                                        logs_enabled=logs_enabled,
                                    )
                                    logger.info("Checking if the model is up ...")
                                    logger.info("\n")
                                    # check if the model is up or not
                                    check_load_test(
                                        pipeline_name=node_name,
                                        model=node_name,
                                        data_type=data_type,
                                        pipeline_path=node_path,
                                    )
                                    logger.info("model warm up ...")
                                    logger.info("\n")
                                    # warm_up_duration = 10
                                    # warm_up(
                                    #     pipeline_name=node_name,
                                    #     model=node_name,
                                    #     data_type=data_type,
                                    #     pipeline_path=node_path,
                                    #     warm_up_duration=warm_up_duration,
                                    # )
                                    logger.info(
                                        "-" * 25 + f"starting load test " + "-" * 25
                                    )
                                    logger.info("\n")
                                    if workload_type == "static":
                                        workload = [load] * load_duration
                                    node_type = "first"
                                    if node_name == "resnet-human":
                                        node_type = "second"
                                    data = load_data(
                                        data_type=data_type,
                                        pipeline_path=node_path,
                                        node_type=node_type,
                                    )
                                    # start_time = time.time()
                                    # output_queue = Queue()
                                    try:
                                        # kwargs = {
                                        #     "pipeline_name": node_name,
                                        #     "model": node_name,
                                        #     "data_type": data_type,
                                        #     "data": data,
                                        #     "workload": workload,
                                        #     "mode": mode,
                                        #     "namespace": "default",
                                        #     # "no_engine": no_engine,
                                        #     "benchmark_duration": benchmark_duration,
                                        #     "queue": output_queue,
                                        # }
                                        # p = Process(target=load_test, kwargs=kwargs)
                                        # p.start()
                                        # while True:
                                        #     time.sleep(1)
                                        #     if p.is_alive():
                                        #         if time.time() - start_time > timeout:
                                        #             print("finished by cap")
                                        #             start_time_experiment = start_time
                                        #             end_time_experiment = time.time()
                                        #             responses = []
                                        #             p.terminate()
                                        #             break
                                        #     else:
                                        #         print("finished on time")
                                        #         (
                                        #             start_time_experiment,
                                        #             end_time_experiment,
                                        #             responses,
                                        #         ) = output_queue.get()
                                        #         p.join()
                                        #         break
                                        (
                                            start_time_experiment,
                                            end_time_experiment,
                                            responses,
                                        ) = load_test(
                                            pipeline_name=node_name,
                                            model=node_name,
                                            data_type=data_type,
                                            data=data,
                                            workload=workload,
                                            mode=mode,
                                            namespace="default",
                                            # load_duration=load_duration,
                                            # no_engine=no_engine,
                                            benchmark_duration=benchmark_duration,
                                        )
                                        logger.info(
                                            "-" * 25 + "saving the report" + "-" * 25
                                        )
                                        logger.info("\n")
                                        save_report(
                                            experiment_id=experiment_id,
                                            responses=responses,
                                            node_name=node_name,
                                            start_time_experiment=start_time_experiment,
                                            end_time_experiment=end_time_experiment,
                                            series=series,
                                            no_engine=no_engine,
                                        )
                                    except UnboundLocalError:
                                        logger.info("Impossible experiment!")
                                        logger.info(
                                            "skipping to the next experiment ..."
                                        )
                                    wait_time = 1
                                    logger.info(f"waiting for: {wait_time} seconds")
                                    for _ in tqdm(range(20)):
                                        time.sleep((wait_time) / 20)
                                    remove_pipeline(pipeline_name=node_name)
                                else:
                                    logger.info(
                                        "experiment with the same set of varialbes already exists"
                                    )
                                    logger.info("skipping to the next experiment ...")
                                    continue


def key_config_mapper(
    pipeline_name: str,
    node_name: str,
    cpu_request: str,
    memory_request: str,
    model_variant: str,
    max_batch_size: str,
    max_batch_time: str,
    load: Union[int, str],
    load_duration: int,
    series: int,
    series_meta: str,
    replica: int,
    no_engine: bool = True,
    mode: str = "step",
    data_type: str = "audio",
    benchmark_duration=1,
):
    dir_path = os.path.join(NODE_PROFILING_RESULTS_PATH, "series", str(series))
    file_path = os.path.join(dir_path, KEY_CONFIG_FILENAME)
    header = [
        "experiment_id",
        "pipeline_name",
        "node_name",
        "model_variant",
        "cpu_request",
        "memory_request",
        "max_batch_size",
        "max_batch_time",
        "load",
        "load_duration",
        "series",
        "series_meta",
        "replicas",
        "no_engine",
        "mode",
        "data_type",
        "benchmark_duration",
    ]
    row_to_add = {
        "experiment_id": None,
        "pipeline_name": pipeline_name,
        "model_variant": model_variant,
        "node_name": node_name,
        "cpu_request": cpu_request,
        "memory_request": memory_request,
        "max_batch_size": max_batch_size,
        "max_batch_time": max_batch_time,
        "load": load,
        "load_duration": load_duration,
        "series": series,
        "series_meta": series_meta,
        "replicas": replica,
        "no_engine": no_engine,
        "mode": mode,
        "data_type": data_type,
        "benchmark_duration": benchmark_duration,
    }
    experiments_exist = False
    if not os.path.exists(file_path):
        # os.makedirs(dir_path)
        with open(file_path, "w", newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(header)
        experiment_id = 1
    else:
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                # add logic of experiment exists
                file_row_dict = {}
                if line_count != 0:
                    for key, value in zip(header, row):
                        if key in [
                            "pipeline_name",
                            "node_name",
                            "max_batch_size",
                            "max_batch_time",
                            "memory_request",
                            "model_variant",
                            "memory_request",
                            "cpu_request",
                            "series_meta",
                            "mode",
                            "data_type",
                        ]:
                            file_row_dict[key] = value
                        elif key in [
                            "experiment_id",
                            "load",
                            "load_duration",
                            "series",
                            "replicas",
                            "benchmark_duration",
                        ]:
                            file_row_dict[key] = int(value)
                        elif key in ["no_engine"]:
                            file_row_dict[key] = eval(value)
                    dict_items_equal = []
                    for header_item in header:
                        if header_item == "experiment_id":
                            continue
                        if row_to_add[header_item] == file_row_dict[header_item]:
                            dict_items_equal.append(True)
                        else:
                            dict_items_equal.append(False)
                    if all(dict_items_equal):
                        experiments_exist = True
                        break
                line_count += 1
        experiment_id = line_count

    if not experiments_exist:
        row_to_add.update({"experiment_id": experiment_id})
        with open(file_path, "a") as row_writer:
            dictwriter_object = csv.DictWriter(row_writer, fieldnames=header)
            dictwriter_object.writerow(row_to_add)
            row_writer.close()

    return experiments_exist, experiment_id


def save_report(
    experiment_id: int,
    responses: str,
    node_name: str,
    start_time_experiment: float,
    end_time_experiment: float,
    namespace: str = "default",
    series: int = 0,
    no_engine: bool = False,
):
    results = {
        "cpu_usage_count": [],
        "time_cpu_usage_count": [],
        "cpu_usage_rate": [],
        "time_cpu_usage_rate": [],
        "cpu_throttled_count": [],
        "time_cpu_throttled_count": [],
        "cpu_throttled_rate": [],
        "time_cpu_throttled_rate": [],
        "memory_usage": [],
        "time_memory_usage": [],
        "throughput": [],
        "time_throughput": [],
        "responses": responses,
        "start_time_experiment": start_time_experiment,
        "end_time_experiment": end_time_experiment,
    }
    # TODO experiments id system
    duration = (end_time_experiment - start_time_experiment) // 60 + 1
    rate = int(end_time_experiment - start_time_experiment)
    save_path = os.path.join(
        NODE_PROFILING_RESULTS_PATH, "series", str(series), f"{experiment_id}.json"
    )
    # TODO add list of pods in case of replicas
    pod_name = get_pod_name(node_name=node_name)[0]

    cpu_usage_count, time_cpu_usage_count = prom_client.get_cpu_usage_count(
        pod_name=pod_name,
        namespace="default",
        duration=int(duration),
        container=node_name,
    )
    cpu_usage_rate, time_cpu_usage_rate = prom_client.get_cpu_usage_rate(
        pod_name=pod_name,
        namespace="default",
        duration=int(duration),
        container=node_name,
        rate=rate,
    )

    cpu_throttled_count, time_cpu_throttled_count = prom_client.get_cpu_throttled_count(
        pod_name=pod_name,
        namespace="default",
        duration=int(duration),
        container=node_name,
    )
    cpu_throttled_rate, time_cpu_throttled_rate = prom_client.get_cpu_throttled_rate(
        pod_name=pod_name,
        namespace="default",
        duration=int(duration),
        container=node_name,
        rate=rate,
    )

    memory_usage, time_memory_usage = prom_client.get_memory_usage(
        pod_name=pod_name,
        namespace="default",
        container=node_name,
        duration=int(duration),
        need_max=False,
    )

    throughput, time_throughput = prom_client.get_request_per_second(
        pod_name=pod_name,
        namespace="default",
        duration=int(duration),
        container=node_name,
        rate=rate,
    )

    results["cpu_usage_count"] = cpu_usage_count
    results["time_cpu_usage_count"] = time_cpu_usage_count
    results["cpu_usage_rate"] = cpu_usage_rate
    results["time_cpu_usage_rate"] = time_cpu_usage_rate

    results["cpu_throttled_count"] = cpu_throttled_count
    results["time_cpu_throttled_count"] = time_cpu_throttled_count
    results["cpu_throttled_rate"] = cpu_throttled_rate
    results["time_cpu_throttled_rate"] = time_cpu_throttled_rate

    results["memory_usage"] = memory_usage
    results["time_memory_usage"] = time_memory_usage

    results["throughput"] = throughput
    results["time_throughput"] = time_throughput

    with open(save_path, "w") as outfile:
        outfile.write(json.dumps(results))

    logger.info(f"results have been sucessfully saved in:\n{save_path}")


@click.command()
@click.option("--config-name", required=True, type=str, default="1-test")
def main(config_name: str):
    config_path = os.path.join(NODE_PROFILING_CONFIGS_PATH, f"{config_name}.yaml")
    with open(config_path, "r") as cf:
        config = yaml.safe_load(cf)
    pipeline_name = config["pipeline_name"]
    node_name = config["node_name"]
    data_type = config["data_type"]
    series = config["series"]

    # pipeline path based on pipeline type [central | distributed] queues
    central_queue = config["central_queue"]
    pipeline_type = "mlserver-centralized" if central_queue else "mlserver-final"
    pipeline_path = os.path.join(
        PIPLINES_PATH, pipeline_type, pipeline_name, "seldon-core-version"
    )
    node_path = os.path.join(pipeline_path, "nodes", node_name)

    dir_path = os.path.join(NODE_PROFILING_RESULTS_PATH, "series", str(series))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        dest_config_path = os.path.join(dir_path, "0.yaml")
        shutil.copy(config_path, dest_config_path)
    else:
        num_configs = 0
        # Iterate directory
        for file in os.listdir(dir_path):
            # check only text files
            if file.endswith(".yaml"):
                num_configs += 1
        dest_config_path = os.path.join(dir_path, f"{num_configs}.yaml")
        shutil.copy(config_path, dest_config_path)

    experiments(
        pipeline_name=pipeline_name,
        node_name=node_name,
        config=config,
        node_path=node_path,
        data_type=data_type,
    )


if __name__ == "__main__":
    main()

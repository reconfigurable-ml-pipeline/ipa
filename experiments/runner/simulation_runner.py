"""
Iterate through all possible combination
of pipelines
"""
import os
import json
import yaml
import click
import sys
import shutil
from typing import List

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from experiments.utils.constants import (
    FINAL_CONFIGS_PATH,
    FINAL_RESULTS_PATH,
    ACCURACIES_PATH,
)

from optimizer import SimAdapter
from experiments.utils.simulation_operations import generate_simulated_pipeline
from experiments.utils.workload import make_workload
from experiments.utils.misc import Int64Encoder


def find_initial_config(
    node_names: List[str],
    config: dict,
):
    model_variants = []
    max_batch_sizes = []
    cpu_requests = []
    memory_requests = []
    replicas = []
    for node_config in config["nodes"]:
        model_variants.append(node_config["model_variants"])
        max_batch_sizes.append(node_config["max_batch_size"])
        cpu_requests.append(node_config["cpu_request"])
        memory_requests.append(node_config["memory_request"])
        replicas.append(node_config["replicas"])
    config = {}
    for node_index, node_name in enumerate(node_names):
        config[node_name] = {}
        config[node_name]["replicas"] = int(replicas[node_index])
        config[node_name]["batch"] = max_batch_sizes[node_index]
        config[node_name]["cpu"] = int(cpu_requests[node_index])
        config[node_name]["variant"] = model_variants[node_index]
    return config


@click.command()
@click.option("--config-name", required=True, type=str, default="base-allocation-video")
@click.option(
    "--type-of",
    required=True,
    type=click.Choice(["experiment", "adaptation"]),
    default="adaptation",
)
def main(config_name: str, type_of: str):
    """loading system configs

    Args:
        config_name (str): configuration for an e2e experiment
    """
    # ----------- 1. loading system configs -------------
    config_path = os.path.join(FINAL_CONFIGS_PATH, f"{config_name}.yaml")
    with open(config_path, "r") as cf:
        config = yaml.safe_load(cf)
    metaseries = config["metaseries"]
    series = config["series"]

    dir_path = os.path.join(
        FINAL_RESULTS_PATH, "metaseries", str(metaseries), "series", str(series)
    )
    save_path = os.path.join(dir_path, "adaptation_log.json")
    pipeline_name = config["pipeline_name"]
    node_names = [config["node_name"] for config in config["nodes"]]
    adaptation_interval = config["adaptation_interval"]

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

    # ----------- 2. loading profiling configs -------------
    with open(config_path, "r") as cf:
        config = yaml.safe_load(cf)
    with open(ACCURACIES_PATH, "r") as cf:
        accuracies = yaml.safe_load(cf)

    # profiling config
    number_tasks = config["number_tasks"]
    profiling_series = config["profiling_series"]
    model_name = config["model_name"]
    task_name = config["task_name"]
    initial_active_model = config["initial_active_model"]
    initial_cpu_allocation = config["initial_cpu_allocation"]
    initial_replica = config["initial_replica"]
    initial_batch = config["initial_batch"]
    scaling_cap = config["scaling_cap"]
    batching_cap = config["batching_cap"]
    pipeline_name = config["pipeline_name"]
    only_measured_profiles = config["only_measured_profiles"]
    profiling_load = config["profiling_load"]

    # pipeline config
    num_state_limit = config["num_state_limit"]
    optimization_method = config["optimization_method"]
    allocation_mode = config["allocation_mode"]
    threshold = config["threshold"]
    sla_factor = config["sla_factor"]
    accuracy_method = config["accuracy_method"]
    normalize_accuracy = config["normalize_accuracy"]

    # pipeline accuracy
    pipeline_accuracies = accuracies[pipeline_name]

    # optimizer
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]

    # baselines [only scaling | only switching]
    baseline_mode = config["baseline_mode"]

    # read the initial config
    initial_config = find_initial_config(config=config, node_names=node_names)

    # reference latency for generating pipeline model
    reference_latency = config["reference_latency"]  # p99 | avg
    reference_throughput = config["reference_throughput"]
    latency_margin = config["latency_margin"]
    throughput_margin = config["throughput_margin"]
    # replica_factor = config["replica_factor"]

    pipeline = generate_simulated_pipeline(
        number_tasks=number_tasks,
        profiling_series=profiling_series,
        model_names=model_name,
        task_names=task_name,
        initial_active_model=initial_active_model,
        allocation_mode=allocation_mode,
        initial_cpu_allocation=initial_cpu_allocation,
        initial_replica=initial_replica,
        initial_batch=initial_batch,
        threshold=threshold,
        sla_factor=sla_factor,
        accuracy_method=accuracy_method,
        normalize_accuracy=normalize_accuracy,
        pipeline_accuracies=pipeline_accuracies,
        only_measured_profiles=only_measured_profiles,
        profiling_load=profiling_load,
        reference_latency=reference_latency,
        reference_throughput=reference_throughput,
        latency_margin=latency_margin,
        throughput_margin=throughput_margin,
    )

    # ----------- 3. loading predictor configs -------------
    monitoring_duration = config["monitoring_duration"]
    predictor_type = config["predictor_type"]
    backup_predictor_type = config["backup_predictor_type"]
    backup_predictor_duration = config["backup_predictor_duration"]

    # whether to use the simulation mode or not
    simulation_mode = config["simulation_mode"]
    if not simulation_mode:
        raise ValueError("wrong config chosen, this is a real experiment config")

    # teleport mode check
    if config["teleport_mode"] and config["simulation_mode"]:
        raise ValueError("teleport model is not available in simulation mode")

    # should be inside of experiments
    adapter = SimAdapter(
        pipeline_name=pipeline_name,
        pipeline=pipeline,
        node_names=node_names,
        adaptation_interval=adaptation_interval,
        optimization_method=optimization_method,
        allocation_mode=allocation_mode,
        only_measured_profiles=only_measured_profiles,
        scaling_cap=scaling_cap,
        batching_cap=batching_cap,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        num_state_limit=num_state_limit,
        monitoring_duration=monitoring_duration,
        predictor_type=predictor_type,
        baseline_mode=baseline_mode,
        backup_predictor_type=backup_predictor_type,
        backup_predictor_duration=backup_predictor_duration,
        # replica_factor=replica_factor
    )

    _, workload = make_workload(config=config)

    # ----------- 3. Running an experiment series -------------
    # 1. Setup the pipeline
    # 2. Makes two processes for experiment and adapter
    # 3. Run both processes at the same time
    # 4. Join both processes

    if type_of == "adaptation":
        # 2. process two the pipeline adapter
        adapter.start_adaptation(workload=workload, initial_config=initial_config)
        with open(save_path, "w") as outfile:
            # outfile.write(json.dumps(convert_values_to_strings(adapter.monitoring.adaptation_report)))
            # outfile.write(json.dumps(adapter.monitoring.adaptation_report))
            outfile.write(
                json.dumps(adapter.monitoring.adaptation_report, cls=Int64Encoder)
            )


if __name__ == "__main__":
    main()

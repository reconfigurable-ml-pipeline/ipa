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

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from experiments.utils.prometheus import PromClient

prom_client = PromClient()

from experiments.utils.constants import (
    FINAL_CONFIGS_PATH,
    FINAL_RESULTS_PATH,
    ACCURACIES_PATH,
)

from optimizer import Adapter
from experiments.utils.simulation_operations import generate_simulated_pipeline
from experiments.utils.workload import make_workload


@click.command()
@click.option("--config-name", required=True, type=str, default="audio-qa-3")
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

    # name resuls zero for consistency with the profiling parser
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
    series = config["series"]
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
    central_queue = config["central_queue"]

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

    # whether if it is in debug mode or not with contaienrs logs
    debug_mode = config["debug_mode"]

    # optimizer
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]

    # baselines [only scaling | only switching]
    baseline_mode = config["baseline_mode"]

    # predicted load
    baseline_mode = config["baseline_mode"]

    # load safety margin from the predictor
    predictor_margin = config["predictor_margin"]

    # teleport mode
    teleport_mode = config["teleport_mode"]
    teleport_interval = config["teleport_interval"]
    if teleport_mode:
        _, workload = make_workload(config=config, teleport_interval=teleport_interval)
    if teleport_mode and config["simulation_mode"]:
        raise ValueError("teleport model is not available in simulation mode")

    # reference latency for generating pipeline model
    reference_latency = config["reference_latency"]  # p99 | avg
    reference_throughput = config["reference_throughput"]
    latency_margin = config["latency_margin"]
    throughput_margin = config["throughput_margin"]

    # read models from storage or container
    from_storage = config["from_storage"]

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

    # should be inside of experiments
    adapter = Adapter(
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
        backup_predictor_type=backup_predictor_type,
        baseline_mode=baseline_mode,
        central_queue=central_queue,
        debug_mode=debug_mode,
        predictor_margin=predictor_margin,
        teleport_mode=teleport_mode,
        teleport_interval=teleport_interval,
        from_storage=from_storage,
    )

    # ----------- 3. Running an experiment series -------------
    # 1. Setup the pipeline
    # 2. Makes two processes for experiment and adapter
    # 3. Run both processes at the same time
    # 4. Join both processes

    if type_of == "adaptation":
        # 2. process two the pipeline adapter
        if teleport_mode:
            adapter.start_adaptation(workload=workload)
        else:
            adapter.start_adaptation()
        with open(save_path, "w") as outfile:
            outfile.write(json.dumps(adapter.monitoring.adaptation_report))


if __name__ == "__main__":
    main()

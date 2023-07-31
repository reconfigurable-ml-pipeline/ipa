import pandas as pd
import click
import time
import os
import sys
import shutil
import yaml
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from optimizer import Optimizer
from experiments.utils.simulation_operations import generate_simulated_pipeline

from experiments.utils.constants import (
    PIPELINE_SIMULATION_CONFIGS_PATH,
    PIPELINE_SIMULATION_RESULTS_PATH,
    ACCURACIES_PATH,
)

from experiments.utils import logger

config_key_mapper = "key_config_mapper.csv"


@click.command()
@click.option("--config-name", required=True, type=str, default="audio-qa")
def main(config_name: str):
    config_path = os.path.join(PIPELINE_SIMULATION_CONFIGS_PATH, f"{config_name}.yaml")
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
    complete_profile = config["complete_profile"]
    only_measured_profiles = config["only_measured_profiles"]
    profiling_load = config["profiling_load"]
    random_sample = config["random_sample"]

    # pipeline config
    arrival_rate = config["arrival_rate"]
    num_state_limit = config["num_state_limit"]
    generate = config["generate"]
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
    )

    optimizer = Optimizer(
        pipeline=pipeline,
        allocation_mode=allocation_mode,
        complete_profile=complete_profile,
        only_measured_profiles=only_measured_profiles,
        random_sample=random_sample,
        baseline_mode=baseline_mode,
    )

    # copy generation config
    dir_path = os.path.join(PIPELINE_SIMULATION_RESULTS_PATH, "series", str(series))
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

    # doing the requested type of experiments
    all_states_time = None
    feasible_time = None
    optimal_time = None
    total_time = None
    time_file = open(os.path.join(dir_path, "times.csv"), "w")
    if optimization_method == "gurobi":
        assert generate[0] == "optimal", "only optimal is allowed with gurbi"
    total_time = time.time()
    if "all" in generate:
        all_states_time = time.time()
        # all states
        states = optimizer.all_states(
            check_constraints=False,
            scaling_cap=scaling_cap,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit,
        )
        states.to_markdown(
            os.path.join(dir_path, "readable-all-states.csv"), index=False
        )
        states.to_csv(os.path.join(dir_path, "all-states.csv"), index=False)
        all_states_time = time.time() - all_states_time
        time_file.write(f"all: {all_states_time}\n")
        logger.info(f"all states time: {all_states_time}")
    if "feasible" in generate:
        feasible_time = time.time()
        # all feasibla states
        with_constraints = optimizer.all_states(
            check_constraints=True,
            scaling_cap=scaling_cap,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit,
        )
        # logger.info(f"{with_constraints = }")
        with_constraints.to_markdown(
            os.path.join(dir_path, "readable-with-constraints.csv"), index=False
        )
        with_constraints.to_csv(
            os.path.join(dir_path, "with-constraints.csv"), index=False
        )
        feasible_time = time.time() - feasible_time
        time_file.write(f"feasible_time: {feasible_time}\n")
        logger.info(f"with constraint time: {feasible_time}")
    if "optimal" in generate:
        if optimization_method == "gurobi":
            optimal_time = time.time()
            # optimal states
            optimal = optimizer.optimize(
                optimization_method=optimization_method,
                scaling_cap=scaling_cap,
                batching_cap=batching_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
            )
            # logger.info(f"{optimal = }")
            optimal.to_markdown(
                os.path.join(dir_path, "readable-optimal-gurobi.csv"), index=False
            )
            optimal.to_csv(os.path.join(dir_path, "optimal-gurobi.csv"), index=False)
            optimal_time = time.time() - optimal_time
            time_file.write(f"optimal_time_gurobi: {optimal_time}\n")
            logger.info(f"optimal time gurobi: {optimal_time}")

        if optimization_method == "brute-force":
            optimal_time = time.time()
            # optimal states
            optimal = optimizer.optimize(
                optimization_method=optimization_method,
                scaling_cap=scaling_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
            )
            # logger.info(f"{optimal = }")
            optimal.to_markdown(
                os.path.join(dir_path, "readable-optimal-brute-force.csv"), index=False
            )
            optimal.to_csv(
                os.path.join(dir_path, "optimal-brute-force.csv"), index=False
            )
            optimal_time = time.time() - optimal_time
            time_file.write(f"optimal_time_brute_force: {optimal_time}\n")
            logger.info(f"optimal time brute-force: {optimal_time}")

        if optimization_method == "both":
            optimal_time = time.time()
            # optimal states
            optimal = optimizer.optimize(
                optimization_method="brute-force",
                scaling_cap=scaling_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
            )
            # logger.info(f"{optimal = }")
            optimal.to_markdown(
                os.path.join(dir_path, "readable-optimal-brute-force.csv"), index=False
            )
            optimal.to_csv(
                os.path.join(dir_path, "optimal-brute-force.csv"), index=False
            )
            optimal_time = time.time() - optimal_time
            time_file.write(f"optimal_time_brute_force: {optimal_time}\n")
            logger.info(f"optimal time brute-force: {optimal_time}")

            optimal_time = time.time()
            # optimal states
            optimal = optimizer.optimize(
                optimization_method="gurobi",
                scaling_cap=scaling_cap,
                batching_cap=batching_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
                baseline_mode=baseline_mode,
            )
            # logger.info(f"{optimal = }")
            optimal.to_markdown(
                os.path.join(dir_path, "readable-optimal-gurobi.csv"), index=False
            )
            optimal.to_csv(os.path.join(dir_path, "optimal-gurobi.csv"), index=False)
            optimal_time = time.time() - optimal_time
            time_file.write(f"optimal_time_gurobi: {optimal_time}\n")
            logger.info(f"optimal time gurobi: {optimal_time}")

    total_time = time.time() - total_time
    time_file.write(f"total_time: {total_time}")
    time_file.close()
    logger.info(f"total time spent: {total_time}")


if __name__ == "__main__":
    main()

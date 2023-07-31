import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import time
import json
import yaml
from json import JSONDecodeError
from copy import deepcopy


class Parser:
    def __init__(
        self,
        series_path,
        model_name,
        second_node=False,
        config_key_mapper: str = None,
        type_of="node",
    ) -> None:
        self.series_path = series_path
        if config_key_mapper is not None:
            self.config_path = os.path.join(series_path, config_key_mapper)
        else:
            self.config_path = None
        self.model_name = model_name
        self.second_node = second_node
        self.type_of = type_of
        legal_types = ["node", "pipeline", "router_pipeline", "router_queue_pipeline"]
        if type_of not in legal_types:
            raise ValueError(f"Invalid type: {type_of}")
        if (
            type_of == "pipeline"
            or type_of == "router_pipeline"
            or type_of == "router_queue_pipeline"
        ):
            self.node_orders = list(
                map(
                    lambda l: l["node_name"],
                    list(self.load_configs().values())[0]["nodes"],
                )
            )

    def load_configs(self) -> Dict[str, Dict[str, Any]]:
        config_files = {}
        for file in os.listdir(self.series_path):
            # check only text files
            if file.endswith(".yaml"):
                config_path = os.path.join(self.series_path, file)
                with open(config_path, "r") as cf:
                    config = yaml.safe_load(cf)
                config_files[file] = config
        return config_files

    def key_config_mapper(self) -> pd.DataFrame:
        key_config_mapper = pd.read_csv(self.config_path)
        return key_config_mapper

    def get_experiment_detail(self, experiment_id: int):
        key_config_mapper = self.key_config_mapper()
        return key_config_mapper[key_config_mapper["experiment_id"] == experiment_id]

    def _get_experiments_with_logs(self):
        key_config_mapper = self.key_config_mapper()
        experiments_with_logs = key_config_mapper[
            key_config_mapper["no_engine"] == False
        ]["experiment_id"].tolist()
        return experiments_with_logs

    def get_result_file_names(self) -> List[str]:
        # for final results case
        if self.config_path == None:
            return ["0.json"]
        # for profiling case
        files = []
        key_config_mapper = self.key_config_mapper()
        experiments_keys = list(key_config_mapper["experiment_id"])
        for path in os.listdir(self.series_path):
            if os.path.isfile(os.path.join(self.series_path, path)):
                file_name = path.split(".")[0]
                try:
                    if int(file_name) in experiments_keys:
                        files.append(path)
                except:
                    pass
        return files

    def _read_results(self, selected=None):
        files = self.get_result_file_names()
        results = {}
        for file in files:
            name = file.split(".")[0].split("/")[-1]
            if selected is not None:
                if int(name) in selected:
                    full_path = os.path.join(self.series_path, file)
                    json_file = open(full_path)
                    try:
                        results[name] = json.load(json_file)
                    except JSONDecodeError:
                        pass
            else:
                full_path = os.path.join(self.series_path, file)
                json_file = open(full_path)
                try:
                    results[name] = json.load(json_file)
                except JSONDecodeError:
                    pass
        return results

    def flatten_results(self, per_second_latencies):
        """
        change the results format from
        [[second_1], [second_2], ...]
        to:
        [req_1, req_2, ...]
        """
        flattend_results = []
        num_each_second_requests = []
        for second_results in per_second_latencies:
            num_each_second_requests.append(len(second_results))
            for request_result in second_results:
                flattend_results.append(request_result)
        return num_each_second_requests, flattend_results

    def _node_latency_calculator(self, results: Dict[Dict, Any]):
        client_to_model = []
        model_latencies = []
        model_to_client = []
        e2e_latencies = []
        latencies = {
            "client_to_model": [],
            "model_latencies": [],
            "model_to_client": [],
            "e2e_latencies": [],
        }
        timeout_count = 0
        for result in results:
            try:
                times = result["times"]
                request_times = times["request"]
                model_times = times["models"][self.model_name]
                client_to_model_latency = (
                    model_times["arrival"] - request_times["sending"]
                )
                model_latency = model_times["serving"] - model_times["arrival"]
                model_to_client_latency = (
                    request_times["arrival"] - model_times["serving"]
                )
                e2e_latencies.append(
                    request_times["arrival"] - request_times["sending"]
                )
                client_to_model.append(client_to_model_latency)
                model_latencies.append(model_latency)
                model_to_client.append(model_to_client_latency)
                latencies = {
                    "client_to_model": client_to_model,
                    "model_latencies": model_latencies,
                    "model_to_client": model_to_client,
                    "e2e_latencies": e2e_latencies,
                }

            except KeyError:
                timeout_count += 1
        return latencies, timeout_count

    def _pipeline_latency_calculator(
        self, results: Dict[Dict, Any], keep_lost: bool = False
    ):
        latencies = {
            "client_to_pipeline": [],
            "pipeline_to_client": [],
        }
        for index, model in enumerate(self.node_orders):
            latencies[f"task_{index}_model"] = []
            if index < len(self.node_orders) - 1:
                latencies[f"task_{index}_to_task_{index+1}_latencies"] = []
        timeout_count = 0
        for result in results:
            try:
                times = result["times"]
                request_times = times["request"]
                model_times = times["models"]
                for index, model in enumerate(self.node_orders):
                    if index == 0:
                        latencies["client_to_pipeline"].append(
                            model_times[model]["arrival"] - request_times["sending"]
                        )
                    if index == len(self.node_orders) - 1:
                        latencies["pipeline_to_client"].append(
                            request_times["arrival"] - model_times[model]["serving"]
                        )
                    latencies[f"task_{index}_model"].append(
                        model_times[model]["serving"] - model_times[model]["arrival"]
                    )
                    if index < len(self.node_orders) - 1:
                        latencies[f"task_{index}_to_task_{index+1}_latencies"].append(
                            model_times[self.node_orders[index + 1]]["arrival"]
                            - model_times[model]["serving"]
                        )
            except KeyError:
                timeout_count += 1
                if keep_lost:
                    for latency_item, _ in latencies.items():
                        latencies[latency_item].append(None)
        return latencies, timeout_count

    def _router_pipeline_latency_calculator(
        self, results: Dict[Dict, Any], keep_lost: bool = False
    ):
        latencies = {
            "client_to_router": [],
            "router_to_task_0": [],
        }

        for index, model in enumerate(self.node_orders):
            latencies[f"task_{index}_model"] = []
            if index < len(self.node_orders) - 1:
                latencies[f"task_{index}_to_task_{index+1}"] = []

        latencies[f"task_{len(self.node_orders) - 1}_to_router"] = []
        latencies["router_to_client"] = []
        latencies["e2e"] = []

        timeout_count = 0
        for result in results:
            try:
                times = result["times"]
                request_times = times["request"]
                model_times = times["models"]
                latencies["client_to_router"].append(
                    model_times["router"]["arrival"] - request_times["sending"]
                )
                latencies["router_to_client"].append(
                    request_times["arrival"] - model_times["router"]["serving"]
                )
                latencies["e2e"].append(
                    request_times["arrival"] - request_times["sending"]
                )
                for index, model in enumerate(self.node_orders):
                    if index == 0:
                        latencies["router_to_task_0"].append(
                            model_times[model]["arrival"]
                            - model_times["router"]["arrival"]
                        )
                    latencies[f"task_{index}_model"].append(
                        model_times[model]["serving"] - model_times[model]["arrival"]
                    )
                    if index == len(self.node_orders) - 1:
                        latencies[f"task_{len(self.node_orders) - 1}_to_router"].append(
                            model_times["router"]["serving"]
                            - model_times[model]["serving"]
                        )
                    if index < len(self.node_orders) - 1:
                        latencies[f"task_{index}_to_task_{index+1}"].append(
                            model_times[self.node_orders[index + 1]]["arrival"]
                            - model_times[model]["serving"]
                        )
            except KeyError:
                timeout_count += 1
                if keep_lost:
                    for latency_item, _ in latencies.items():
                        latencies[latency_item].append(None)
        return latencies, timeout_count

    def _router_queue_pipeline_latency_calculator(
        self, results: Dict[Dict, Any], keep_lost: bool = False
    ):
        latencies = {
            "client_to_router": [],
            "router_to_task_0_queue": [],
            "former_router_to_task_0": [],
        }

        for index, model in enumerate(self.node_orders):
            latencies[f"task_{index}_queue_to_task_{index}"] = []
            latencies[f"task_{index}_model"] = []
            latencies[f"task_{index}_to_task_{index}_queue"] = []
            if index < len(self.node_orders) - 1:
                latencies[f"task_{index}_queue_to_task_{index+1}_queue"] = []

        latencies[f"task_{len(self.node_orders) - 1}_queue_to_router"] = []
        latencies["router_to_client"] = []
        latencies["e2e"] = []

        timeout_count = 0
        for result in results:
            try:
                times = result["times"]
                request_times = times["request"]
                model_times = times["models"]
                latencies["client_to_router"].append(
                    model_times["router"]["arrival"] - request_times["sending"]
                )
                latencies["router_to_client"].append(
                    request_times["arrival"] - model_times["router"]["serving"]
                )
                latencies["e2e"].append(
                    request_times["arrival"] - request_times["sending"]
                )
                for index, model in enumerate(self.node_orders):
                    if index == 0:
                        latencies["router_to_task_0_queue"].append(
                            model_times[f"queue-{model}"]["arrival"]
                            - model_times["router"]["arrival"]
                        )
                        latencies["former_router_to_task_0"].append(
                            model_times[model]["arrival"]
                            - model_times["router"]["arrival"]
                        )
                    latencies[f"task_{index}_model"].append(
                        model_times[model]["serving"] - model_times[model]["arrival"]
                    )
                    latencies[f"task_{index}_queue_to_task_{index}"].append(
                        model_times[model]["arrival"]
                        - model_times[f"queue-{model}"]["arrival"]
                    )
                    latencies[f"task_{index}_to_task_{index}_queue"].append(
                        model_times[f"queue-{model}"]["serving"]
                        - model_times[model]["serving"]
                    )
                    if index == len(self.node_orders) - 1:
                        latencies[
                            f"task_{len(self.node_orders) - 1}_queue_to_router"
                        ].append(
                            model_times["router"]["serving"]
                            - model_times[f"queue-{model}"]["serving"]
                        )
                    if index < len(self.node_orders) - 1:
                        latencies[f"task_{index}_queue_to_task_{index+1}_queue"].append(
                            model_times[f"queue-{self.node_orders[index + 1]}"][
                                "arrival"
                            ]
                            - model_times[f"queue-{model}"]["serving"]
                        )
            except KeyError:
                timeout_count += 1
                if keep_lost:
                    for latency_item, _ in latencies.items():
                        latencies[latency_item].append(None)
        return latencies, timeout_count

    def latency_calculator(self, results: Dict[Dict, Any], log=None, keep_lost=False):
        """symmetric input meaning:
        per each input at the first node of the pipeline
        we only have one output going to the second node of the
        pipeline
        """
        if self.type_of == "node":
            return self._node_latency_calculator(results)
        elif self.type_of == "pipeline":
            return self._pipeline_latency_calculator(results, keep_lost=keep_lost)
        elif self.type_of == "router_pipeline":
            return self._router_pipeline_latency_calculator(
                results, keep_lost=keep_lost
            )
        elif self.type_of == "router_queue_pipeline":
            return self._router_queue_pipeline_latency_calculator(
                results, keep_lost=keep_lost
            )

    def metric_summary(self, metric, values):
        summary = {}
        if values is not None:
            values = list(filter(lambda x: x is not None, values))
        if values != [] and values != None:
            try:
                summary[f"{metric}_avg"] = np.average(values)
            except TypeError:
                pass
            summary[f"{metric}_p99"] = np.percentile(values, 99)
            summary[f"{metric}_p95"] = np.percentile(values, 95)
            summary[f"{metric}_p50"] = np.percentile(values, 50)
            summary[f"{metric}_var"] = np.var(values)
            summary[f"{metric}_max"] = max(values)
            summary[f"{metric}_min"] = min(values)
        else:
            summary[f"{metric}_avg"] = None
            summary[f"{metric}_p99"] = None
            summary[f"{metric}_p95"] = None
            summary[f"{metric}_p50"] = None
            summary[f"{metric}_var"] = None
            summary[f"{metric}_max"] = None
            summary[f"{metric}_min"] = None
        return summary

    def latency_summary(self, latencies):
        summary = {}
        for metric_name, values in latencies.items():
            summary.update(self.metric_summary(metric=metric_name, values=values))
        return summary

    def result_processing(self):
        log = None
        selected = None
        results = self._read_results(selected)
        final_dataframe = []
        for experiment_id, result in results.items():
            processed_exp = {"experiment_id": int(experiment_id)}
            _, flattened_results = self.flatten_results(
                results[str(experiment_id)]["responses"]
            )
            if log is not None:
                latencies, timeout_count = self.latency_calculator(
                    flattened_results, log[experiment_id]
                )
            else:
                latencies, timeout_count = self.latency_calculator(
                    flattened_results, log
                )
            latencies = self.latency_summary(latencies)
            processed_exp.update(latencies)
            processed_exp["start_time"] = time.ctime(result["start_time_experiment"])
            processed_exp["end_time"] = time.ctime(result["end_time_experiment"])
            processed_exp["duration"] = round(
                result["end_time_experiment"] - result["start_time_experiment"]
            )
            processed_exp["timeout_count"] = timeout_count
            skipped_metrics = [
                "time_cpu_usage_count",
                "time_cpu_usage_rate",
                "time_cpu_throttled_count",
                "time_cpu_throttled_rate",
                "time_memory_usage",
                "time_throughput",
                "responses",
                "start_time_experiment",
                "end_time_experiment",
            ]
            if self.type_of == "node":
                for metric, values in result.items():
                    if metric in skipped_metrics:
                        continue
                    processed_exp.update(
                        self.metric_summary(metric=metric, values=values)
                    )
                final_dataframe.append(processed_exp)
            elif (
                self.type_of == "pipeline"
                or self.type_of == "router_pipeline"
                or self.type_of == "router_queue_pipeline"
            ):
                # the adapatation capability
                if self.config_path is not None:
                    for index, model in enumerate(self.node_orders):
                        for pod_name, pod_values in result[model].items():
                            pod_index = 1
                            for metric, values in pod_values.items():
                                if metric in skipped_metrics:
                                    continue
                                processed_exp.update(
                                    self.metric_summary(
                                        metric=f"task_{index}_{metric}", values=values
                                    )
                                )
                            pod_index += 1
                final_dataframe.append(processed_exp)
        return pd.DataFrame(final_dataframe)

    def per_request_result_processing(self, experiment_id=0):
        log = None
        selected = None
        results = self._read_results(selected)
        # for experiment_id, result in results.items():
        # processed_exp = {"experiment_id": int(experiment_id)}
        _, flattened_results = self.flatten_results(
            results[str(experiment_id)]["responses"]
        )
        latencies, _ = self.latency_calculator(flattened_results, log, keep_lost=True)
        return latencies

    def per_second_result_processing(self, experiment_id=0):
        log = None
        selected = None
        results = self._read_results(selected)
        # for experiment_id, result in results.items():
        # processed_exp = {"experiment_id": int(experiment_id)}
        num_request_per_seconds, flattened_results = self.flatten_results(
            results[str(experiment_id)]["responses"]
        )
        latencies, _ = self.latency_calculator(flattened_results, log, keep_lost=True)
        temp_latency = deepcopy(latencies)
        per_second_stats = []
        for i in range(len(num_request_per_seconds)):
            temp_dict = {}
            for key in temp_latency:
                temp_dict[key] = temp_latency[key][: num_request_per_seconds[i]]
                temp_latency[key] = temp_latency[key][num_request_per_seconds[i] :]
            per_second_stats.append(temp_dict)
        stats = []
        for item in per_second_stats:
            stats.append(self.latency_summary(item))
        timeout_per_second = []
        for item in per_second_stats:
            if self.type_of == "pipeline":
                num_nones = len(
                    list(filter(lambda x: x is None, item["client_to_pipeline"]))
                )
            if (
                self.type_of == "router_pipeline"
                or self.type_of == "router_queue_pipeline"
            ):
                num_nones = len(
                    list(filter(lambda x: x is None, item["client_to_router"]))
                )
            timeout_per_second.append(num_nones)
        return timeout_per_second, pd.DataFrame(stats)

    def table_maker(
        self,
        experiment_ids: List[int],
        metadata_columns: List[str],
        results_columns: List[str],
    ):
        # extract full data
        results = self.result_processing()
        metadata = self.key_config_mapper()
        # retrieve rows
        selected_results = results[results["experiment_id"].isin(experiment_ids)]
        selected_metadata = metadata[metadata["experiment_id"].isin(experiment_ids)]
        merged_results = selected_metadata.merge(selected_results)
        columns = metadata_columns + results_columns
        output = merged_results[columns]
        return output


class AdaptationParser:
    def __init__(self, series_path, model_name, type_of) -> None:
        self.series_path = series_path
        self.loader = Parser(
            series_path=series_path,
            config_key_mapper=None,
            model_name=model_name,
            type_of=type_of,
        )

    def load_configs(self):
        return self.loader.load_configs()

    def result_processing(self):
        return self.loader.result_processing()

    def per_second_result_processing(self):
        return self.loader.per_second_result_processing()

    def read_results(self):
        return self.loader._read_results()

    def flatten_results(self, per_second_latencies):
        return self.loader.flatten_results(per_second_latencies)

    def latency_calculator(self, results: Dict[Dict, Any]):
        return self.loader.latency_calculator(results=results)

    def load_adaptation_log(self) -> Dict[str, Any]:
        adaptation_file = os.path.join(self.series_path, "adaptation_log.json")
        with open(adaptation_file, "r") as input_file:
            adaptation_log = json.load(input_file)
        return adaptation_log

    def points_with_change(
        self, adaptation_log: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[bool]]:
        node_changes = {}
        for node in self.loader.node_orders:
            node_changes[node] = []
        for node in self.loader.node_orders:
            last_config = {}
            for _, config in adaptation_log["timesteps"].items():
                if last_config == {}:
                    last_config = deepcopy(config["config"][node])
                    continue
                for config_knob, config_knob_value in config["config"][node].items():
                    if last_config[config_knob] != config_knob_value:
                        node_changes[node].append(True)
                        break
                else:  # no-break
                    node_changes[node].append(False)
                last_config = deepcopy(config["config"][node])
        return node_changes

    def series_changes(self, adaptation_log: Dict[str, Dict[str, Any]]):
        changes = {
            "time_interval": [],
            "accuracy_objective": [],
            "resource_objective": [],
            "batch_objective": [],
            "objective": [],
            "monitored_load": [],
            "predicted_load": [],
            "nodes": {},
            "total_load": [],
        }
        for node_name in self.loader.node_orders:
            changes["nodes"][node_name] = {
                "cpu": [],
                "replicas": [],
                "batch": [],
                "variant": [],
                "latency": [],
                "accuracy": [],
                "throughput": [],
            }
        try:  # for backward compatibility with older experiments
            changes["recieved_load"] = adaptation_log["metadata"]["recieved_load"]
            changes["sla"] = adaptation_log["metadata"]["sla"]
            try:  # backward compatibility
                changes["stage_wise_slas"] = adaptation_log["metadata"][
                    "stage_wise_slas"
                ]
            except:
                pass
            try:  # backward compatibility
                changes["base_allocations"] = adaptation_log["metadata"][
                    "base_allocations"
                ]
            except:
                pass
            for _, state in adaptation_log["timesteps"].items():
                changes["time_interval"].append(state["time_interval"])
                try:  # backward compatibility
                    changes["accuracy_objective"].append(state["accuracy_objective"])
                    changes["resource_objective"].append(state["resource_objective"])
                    changes["batch_objective"].append(state["batch_objective"])
                except KeyError:
                    pass
                changes["objective"].append(state["objective"])
                changes["monitored_load"].append(state["monitored_load"])
                changes["predicted_load"].append(state["predicted_load"])
                for node_name in self.loader.node_orders:
                    for metric, _ in state["config"][node_name].items():
                        if metric == "batch":
                            value = int(state["config"][node_name][metric])
                        else:
                            value = state["config"][node_name][metric]
                        changes["nodes"][node_name][metric].append(value)
        except KeyError:
            pass
        return changes


    def series_changes_gurobi(self, adaptation_log: Dict[str, Dict[str, Any]]):
        changes = {
            "duration": [],
        }
        for _, state in adaptation_log["timesteps"].items():
            changes["duration"].append(state["duration"])
        return changes
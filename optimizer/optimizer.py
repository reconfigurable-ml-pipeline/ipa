import random
import math
from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import itertools
from copy import deepcopy
from .models import Pipeline


class Optimizer:
    def __init__(
        self,
        pipeline: Pipeline,
        allocation_mode: str,
        complete_profile: bool,
        only_measured_profiles: bool,
        random_sample: bool,
        baseline_mode: Optional[str] = None,
    ) -> None:
        """_summary_

        Args:
            pipeline (Pipeline): pipeline objecit for optimization
            allocation_mode (str): allocation mode for cpu usage,
                fix | base | variable
                fix: stays on the initiial CPU allocation
                base: finding the base allocation explained in the paper
                variable: search through the cpu allocation as a configuration knob
            complete_profile (bool): whether to log the complete result or not
            only_measured_profiles (bool): only profiled based on the measured latency/throughput
                profiles and not using regression models
        """
        self.pipeline = pipeline
        self.allocation_mode = allocation_mode
        self.complete_profile = complete_profile
        self.only_measured_profiles = only_measured_profiles
        self.random_sample = random_sample
        self.baseline_mode = baseline_mode

    def accuracy_objective(self) -> float:
        """
        objective function of the pipeline
        """
        accuracy_objective = self.pipeline.pipeline_accuracy
        return accuracy_objective

    def resource_objective(self) -> float:
        """
        objective function of the pipeline
        """
        resource_objective = self.pipeline.cpu_usage
        return resource_objective

    def batch_objective(self) -> float:
        """
        batch objecive of the pipeline
        """
        max_batch = 0
        for task in self.pipeline.inference_graph:
            max_batch += task.batch
        return max_batch

    def objective(self, alpha: float, beta: float, gamma: float) -> Dict[str, float]:
        """
        objective function of the pipeline
        """
        objectives = {}
        objectives["accuracy_objective"] = alpha * self.accuracy_objective()
        objectives["resource_objective"] = beta * self.resource_objective()
        objectives["batch_objective"] = gamma * self.batch_objective()
        objectives["objective"] = (
            objectives["accuracy_objective"]
            - objectives["resource_objective"]
            - objectives["batch_objective"]
        )
        return objectives

    def constraints(self, arrival_rate: int) -> bool:
        """
        whether the constraints are met or not
        """
        if self.sla_is_met() and self.can_sustain_load(arrival_rate=arrival_rate):
            return True
        return False

    def pipeline_latency_upper_bound(self, stage, variant_name) -> float:
        # maximum number for latency of a node in
        # a pipeline for calculating the M variable
        max_model = 0
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            if task.name == stage:
                task.model_switch(variant_name)
                task.change_batch(max(task.batches))
                max_model = task.model_latency
        return max_model

    def latency_parameters(
        self, only_measured_profiles
    ) -> Union[
        Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, Dict[str, float]]]
    ]:
        # latency parameters of regression models
        #  of all cases nested dictionary
        # for gorubi solver
        # [stage_name][variant]
        # or
        # [stage_name][variant][batch]
        # if only_measured_profiles:
        # HACK for now do this for both measured and unmeasured cases
        model_latencies_parameters = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            model_latencies_parameters[task.name] = {}
            for variant_name in task.variant_names:
                model_latencies_parameters[task.name][variant_name] = {}
                task.model_switch(variant_name)
                for batch_size in task.batches:
                    model_latencies_parameters[task.name][variant_name][batch_size] = {}
                    task.change_batch(batch_size)
                    model_latencies_parameters[task.name][variant_name][
                        batch_size
                    ] = task.model_latency
        # extract all batches profiles for filling out missing batches
        # with very big values to make them consistent for Gurobi
        batches_profiles = list(
            map(
                lambda l: l,
                list(
                    map(
                        lambda l: list(l.values()),
                        list(model_latencies_parameters.values()),
                    )
                ),
            )
        )
        all_batches_profiles = []
        for batch_prfoile in batches_profiles:
            all_batches_profiles += batch_prfoile
        distinct_batches = []
        for model_batch in all_batches_profiles:
            for batch in model_batch:
                if batch not in distinct_batches:
                    distinct_batches.append(batch)
        # add the big value for model with missing latency
        dummy_latency = 1000
        for stage, variants in model_latencies_parameters.items():
            for variant_name, variant_profile in variants.items():
                for batch in distinct_batches:
                    if batch not in variant_profile.keys():
                        model_latencies_parameters[stage][variant_name][
                            batch
                        ] = dummy_latency
            # else:
            # HACK gurobi does not support cubic equations so for now
            # we will use the same method as only measured profiles to store all the
            # latency values staticly
            pass
            # model_latencies_parameters = {}
            # inference_graph = deepcopy(self.pipeline.inference_graph)
            # for task in inference_graph:
            #     model_latencies_parameters[task.name] = {}
            #     for variant_name in task.variant_names:
            #         model_latencies_parameters[task.name][variant_name] = {}
            #         task.model_switch(variant_name)
            #         # for batch_size in task.batches:
            #         # task.change_batch(batch_size)
            #         model_latencies_parameters[task.name][
            #             variant_name
            #         ] = task.latency_model_params
        return model_latencies_parameters

    def throughput_parameters(self) -> Dict[str, Dict[str, List[float]]]:
        # throughputs of all cases nested dictionary
        # for gorubi solver
        # [stage_name][variant][batch]
        model_throughputs = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            model_throughputs[task.name] = {}
            for variant_name in task.variant_names:
                model_throughputs[task.name][variant_name] = {}
                task.model_switch(variant_name)
                for batch_size in task.batches:
                    model_throughputs[task.name][variant_name][batch_size] = {}
                    task.change_batch(batch_size)
                    model_throughputs[task.name][variant_name][
                        batch_size
                    ] = task.throughput
        # extract all batches profiles for filling out missing batches
        # with very small values to make them consistent for Gurobi
        batches_profiles = list(
            map(
                lambda l: l,
                list(map(lambda l: list(l.values()), list(model_throughputs.values()))),
            )
        )
        all_batches_profiles = []
        for batch_prfoile in batches_profiles:
            all_batches_profiles += batch_prfoile
        distinct_batches = []
        for model_batch in all_batches_profiles:
            for batch in model_batch:
                if batch not in distinct_batches:
                    distinct_batches.append(batch)
        # add the small value for model with missing throughputs
        dummy_throughput = 0.00001
        for stage, variants in model_throughputs.items():
            for variant_name, variant_profile in variants.items():
                for batch in distinct_batches:
                    if batch not in variant_profile.keys():
                        model_throughputs[stage][variant_name][batch] = dummy_throughput
        return distinct_batches, model_throughputs

    def accuracy_parameters(self) -> Dict[str, Dict[str, float]]:
        # accuracies of all cases nested dictionary
        # for gorubi solver
        # [stage_name][variant]
        model_accuracies = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            model_accuracies[task.name] = {}
            for variant_name in task.variant_names:
                model_accuracies[task.name][variant_name] = {}
                task.model_switch(variant_name)
                model_accuracies[task.name][variant_name] = task.accuracy
        return model_accuracies

    def base_allocations(self):
        # base allocation of all cases nested dictionary
        # for gorubi solver
        # [stage_name][variant]
        base_allocations = {}
        inference_graph = deepcopy(self.pipeline.inference_graph)
        for task in inference_graph:
            if self.pipeline.gpu_mode:
                base_allocations[task.name] = {
                    key: value.gpu for (key, value) in task.base_allocations.items()
                }
            else:
                base_allocations[task.name] = {
                    key: value.cpu for (key, value) in task.base_allocations.items()
                }
        return base_allocations

    def all_states(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        check_constraints: bool,
        arrival_rate: int,
        num_state_limit: int = None,
    ) -> pd.DataFrame:
        """generate all the possible states based on profiling data

        Args:
            check_constraints (bool, optional): whether to check the
                objective function contraint or not. Defaults to False.
            scaling_cap (int, optional): maximum number of allowed horizontal
                scaling for each node. Defaults to 2.
            alpha (float, optional): accuracy ojbective weight.
                Defaults to 1.
            beta (float, optional): resource usage
                objective weigth. Defaults to 1.
            gamma (float, optional): batch size
                objective batch. Defaults to 1.
            arrival_rate (int, optional): arrival rate into
                the pipeline. Defaults to None.
            state_limit (int, optional): whether to generate a
                fixed number of state. Defaults to None.

        Returns:
            pd.DataFrame: all the states of the pipeline
        """
        if num_state_limit is not None:
            state_counter = 0
        variant_names = []
        replicas = []
        batches = []
        allocations = []
        for task in self.pipeline.inference_graph:
            variant_names.append(task.variant_names)
            replicas.append(np.arange(1, scaling_cap + 1))
            batches.append(task.batches)
            if self.allocation_mode == "variable":
                if task.gpu_mode:
                    allocations.append(task.resource_allocations_gpu_mode)
                else:
                    allocations.append(task.resource_allocations_cpu_mode)
            elif self.allocation_mode == "fix":
                allocations.append([task.initial_allocation])
            elif self.allocation_mode == "base":
                pass
            else:
                raise ValueError(f"Invalid allocation_mode: {self.allocation_mode}")

        variant_names = list(itertools.product(*variant_names))
        replicas = list(itertools.product(*replicas))
        batches = list(itertools.product(*batches))
        if self.allocation_mode != "base":
            allocations = list(itertools.product(*allocations))
            all_combinations = itertools.product(
                *[variant_names, replicas, batches, allocations]
            )
        else:
            all_combinations = itertools.product(*[variant_names, replicas, batches])

        if self.random_sample:
            all_combinations = random.sample(list(all_combinations), num_state_limit)

        # generate states header format
        states = []

        for combination in all_combinations:
            try:  # Not all models profiles are available under all batch sizes
                for task_id_i in range(self.pipeline.num_nodes):
                    # change config knobs (model_variant, batch, scale)
                    self.pipeline.inference_graph[task_id_i].model_switch(
                        active_variant=combination[0][task_id_i]
                    )
                    self.pipeline.inference_graph[task_id_i].re_scale(
                        replica=combination[1][task_id_i]
                    )
                    self.pipeline.inference_graph[task_id_i].change_batch(
                        batch=combination[2][task_id_i]
                    )
                    if self.allocation_mode != "base":
                        self.pipeline.inference_graph[task_id_i].change_allocation(
                            active_allocation=combination[3][task_id_i]
                        )
                ok_to_add = False
                if check_constraints:
                    if self.constraints(arrival_rate=arrival_rate):
                        ok_to_add = True
                else:
                    ok_to_add = True
                if ok_to_add:
                    state = {}
                    if self.complete_profile:
                        for task_id_j in range(self.pipeline.num_nodes):
                            # record all stats under this configs
                            state[
                                f"task_{task_id_j}_latency"
                            ] = self.pipeline.inference_graph[task_id_j].latency
                            state[
                                f"task_{task_id_j}_throughput"
                            ] = self.pipeline.inference_graph[task_id_j].throughput
                            state[
                                f"task_{task_id_j}_throughput_all_replicas"
                            ] = self.pipeline.inference_graph[
                                task_id_j
                            ].throughput_all_replicas
                            state[
                                f"task_{task_id_j}_accuracy"
                            ] = self.pipeline.inference_graph[task_id_j].accuracy
                            state[
                                f"task_{task_id_j}_measured"
                            ] = self.pipeline.inference_graph[task_id_j].measured
                            state[
                                f"task_{task_id_j}_cpu_all_replicas"
                            ] = self.pipeline.inference_graph[
                                task_id_j
                            ].cpu_all_replicas
                            state[
                                f"task_{task_id_j}_gpu_all_replicas"
                            ] = self.pipeline.inference_graph[
                                task_id_j
                            ].gpu_all_replicas
                        state["pipeline_accuracy"] = self.pipeline.pipeline_accuracy
                        state["pipeline_latency"] = self.pipeline.pipeline_latency
                        state["pipeline_throughput"] = self.pipeline.pipeline_throughput
                        state["pipeline_cpu"] = self.pipeline.pipeline_cpu
                        state["pipeline_gpu"] = self.pipeline.pipeline_gpu
                        state["alpha"] = alpha
                        state["beta"] = beta
                        state["gamma"] = gamma
                        state["accuracy_objective"] = self.accuracy_objective()
                        state["resource_objective"] = self.resource_objective()
                        state["batch_objective"] = self.batch_objective()

                    for task_id_j in range(self.pipeline.num_nodes):
                        # record all stats under this configs
                        state[
                            f"task_{task_id_j}_variant"
                        ] = self.pipeline.inference_graph[task_id_j].active_variant
                        state[f"task_{task_id_j}_cpu"] = self.pipeline.inference_graph[
                            task_id_j
                        ].cpu
                        state[f"task_{task_id_j}_gpu"] = self.pipeline.inference_graph[
                            task_id_j
                        ].gpu
                        state[
                            f"task_{task_id_j}_batch"
                        ] = self.pipeline.inference_graph[task_id_j].batch
                        state[
                            f"task_{task_id_j}_replicas"
                        ] = self.pipeline.inference_graph[task_id_j].replicas

                    state["objective"] = self.objective(
                        alpha=alpha, beta=beta, gamma=gamma
                    )
                    states.append(state)
                    if num_state_limit is not None:
                        state_counter += 1
                        # print(f"state {state_counter} added")
                        if state_counter == num_state_limit:
                            break
            except StopIteration:
                pass
        return pd.DataFrame(states)

    def brute_force(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int = None,
    ) -> pd.DataFrame:
        states = self.all_states(
            check_constraints=True,
            scaling_cap=scaling_cap,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit,
        )
        optimal = states[states["objective"] == states["objective"].max()]
        return optimal

    def gurobi_optmizer(
        self,
        scaling_cap: int,
        batching_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int,
    ) -> pd.DataFrame:
        """generate all the possible states based on profiling data

        Args:
            check_constraints (bool, optional): whether to check the
                objective function contraint or not. Defaults to False.
            scaling_cap (int, optional): maximum number of allowed horizontal
                scaling for each node. Defaults to 2.
            alpha (float, optional): accuracy ojbective weight.
                Defaults to 1.
            beta (float, optional): resource usage
                objective weigth. Defaults to 1.
            arrival_rate (int, optional): arrival rate into
                the pipeline. Defaults to None.
            sla (float, optional): end to end service level agreement
                of pipeline. Defaults to None.
            baseline: baseline approach [scaling | switch]
        Returns:
            pd.DataFrame: all the states of the pipeline
        """
        self.only_measured_profiles = (
            True  # HACK for now handle both cases through using pre-calculated profiles
        )
        sla = self.pipeline.sla
        variant_names = []
        replicas = []
        batches = []
        assert (
            self.allocation_mode == "base"
        ), "currrently only base mode is supported with Gurobi"
        for task in self.pipeline.inference_graph:
            variant_names.append(task.variant_names)
            replicas.append(np.arange(1, scaling_cap + 1))
            batches.append(task.batches)
        # if self.only_measured_profiles:
        #     batching_cap = max(batches[0])

        def func_l(batch: int, params: Dict[str, float]) -> float:
            """using parameters of fitted models

            Args:
                batch: batch size
                params: parameters of the linear model

            Returns:
                latency
            """
            # HACK gurobi does not support quadratic terms and it seeems
            # it isn't feasible to use them now so we don't use this for now
            # TODO change this to the model itself
            coefficients = params["coefficients"]
            intercept = params["intercept"]
            latency = (
                coefficients[2] * (batch**2)
                + coefficients[1] * batch
                + coefficients[0]
                + intercept[0]
            )
            return latency

        def func_q(batch, arrival_rate):
            """queueing latency

            Args:
                batch: batch size
                params: parameters of the linear model

            Returns:
                latency
            """
            if arrival_rate == 0:
                return 0  # just handling the zero load case
            queue = (batch - 1) / arrival_rate
            return queue

        # defining groubipy model for descision problem
        model = gp.Model("pipeline")

        # stages
        stages = self.pipeline.stage_wise_task_names
        stages_variants = self.pipeline.stage_wise_available_variants

        # coefficients
        base_allocations = self.base_allocations()
        accuracy_parameters = self.accuracy_parameters()
        if self.only_measured_profiles:
            distinct_batches, throughput_parameters = self.throughput_parameters()
            latency_parameters = self.latency_parameters(
                only_measured_profiles=self.only_measured_profiles
            )
            distinct_batches = [
                distinct_batch
                for distinct_batch in distinct_batches
                if distinct_batch <= batching_cap
            ]
        else:
            latency_parameters = self.latency_parameters(
                only_measured_profiles=self.only_measured_profiles
            )

        # sets
        gurobi_variants = []
        gurobi_replicas = []
        gurobi_batches = []
        for stage_index, stage_name in enumerate(stages):
            gurobi_variants += [
                (stage_name, variant) for variant in variant_names[stage_index]
            ]
            gurobi_replicas += [stage_name]
            if self.only_measured_profiles:
                gurobi_batches += [stage_name]
            else:
                gurobi_batches += [stage_name]

        # variables
        i = model.addVars(gurobi_variants, name="i", vtype=GRB.BINARY)
        n_lb = 1
        b_lb = 1
        if self.only_measured_profiles:
            b = model.addVars(
                gurobi_batches, distinct_batches, name="b", vtype=GRB.BINARY
            )
            aux_batch = model.addVars(
                gurobi_variants, distinct_batches, name="aux", vtype=GRB.BINARY
            )
        else:
            b = model.addVars(
                gurobi_batches, name="b", vtype=GRB.INTEGER, lb=b_lb, ub=batching_cap
            )
            # variables for enforcing only power of 2s
            batch_sizes = [2**i for i in range(int(math.log2(batching_cap)) + 1)]
            batch_size_indicator = model.addVars(stages, batch_sizes, vtype=GRB.BINARY)
        n = model.addVars(
            gurobi_replicas, name="n", vtype=GRB.INTEGER, lb=n_lb, ub=scaling_cap
        )
        model.update()

        # constraints
        if self.only_measured_profiles:
            # throughput constraint
            # trick based on the following answer
            # https://support.gurobi.com/hc/en-us/community/posts/360077892211-How-Indicator-constraint-can-be-triggered-with-multiple-variables-?input_string=How%20I%20can%20add%20an%20indicator%20cons
            for stage in stages:
                for variant in stages_variants[stage]:
                    for batch in distinct_batches:
                        model.addGenConstrAnd(
                            aux_batch[stage, variant, batch],
                            [i[stage, variant], b[stage, batch]],
                            "andconstr-batch-variant",
                        )
                        model.addConstr(
                            (aux_batch[stage, variant, batch] == 1)
                            >> (
                                n[stage] * throughput_parameters[stage][variant][batch]
                                >= arrival_rate
                            )
                        )
            # latency constraint
            # for stage in stages:
            #     for variant in stages_variants[stage]:
            #         for batch in distinct_batches:
            #             model.addConstr(
            #                 (b[stage, batch] == 1) >>
            #                 (latency_parameters[stage][variant][batch] * i[stage, variant] +
            #                 func_q(b[stage, batch], queue_parameters[stage]) <= sla), name='latency')
            # for stage in stages:
            #     for variant in stages_variants[stage]:
            #         for batch in distinct_batches:
            model.addQConstr(
                (
                    gp.quicksum(
                        latency_parameters[stage][variant][batch]
                        * i[stage, variant]
                        * b[stage, batch]
                        # + func_q(b[stage, batch], queue_parameters[stage])
                        + func_q(b[stage, batch], arrival_rate)
                        for stage in stages
                        for variant in stages_variants[stage]
                        for batch in distinct_batches
                    )
                    <= sla
                ),
                name="latency",
            )
            # add the constraint of batches, only one batch get selected per model servers
            model.addConstrs(
                (
                    gp.quicksum(b[stage, batch] for batch in distinct_batches) == 1
                    for stage in stages
                ),
                name="single-batch",
            )
        else:
            # throughput constraint
            # upper bound trick based on
            # https://support.gurobi.com/hc/en-us/community/posts/12996185241105-How-to-add-quadratic-constraint-in-conditional-indicator-constraints
            for stage in stages:
                for variant in stages_variants[stage]:
                    M = (
                        arrival_rate * self.pipeline_latency_upper_bound(stage, variant)
                        - n_lb * b_lb
                    )
                    model.addQConstr(
                        (
                            (
                                arrival_rate
                                * func_l(b[stage], latency_parameters[stage][variant])
                                - n[stage] * b[stage]
                            )
                            <= M * (1 - i[stage, variant])
                        ),
                        f"throughput-{stage}-{variant}",
                    )
            # latency constraint
            model.addQConstr(
                (
                    gp.quicksum(
                        func_l(b[stage], latency_parameters[stage][variant])
                        * i[stage, variant]
                        # + func_q(b[stage], queue_parameters[stage])
                        + func_q(b[stage], arrival_rate)
                        for stage in stages
                        for variant in stages_variants[stage]
                    )
                    <= sla
                ),
                name="latency",
            )

            # Add constraints to ensure that only one value is selected
            for stage in stages:
                model.addConstr(
                    gp.quicksum(
                        batch_size_indicator[stage, batch_size]
                        for batch_size in batch_sizes
                    )
                    == 1
                )

            # Add constraints to enforce the indicator variables
            for stage in stages:
                for batch_size in batch_sizes:
                    model.addConstr(
                        b[stage]
                        >= batch_size
                        - (max(batch_sizes) - min(batch_sizes))
                        * (1 - batch_size_indicator[stage, batch_size])
                    )
                    model.addConstr(
                        b[stage]
                        <= batch_size
                        + (max(batch_sizes) - min(batch_sizes))
                        * (1 - batch_size_indicator[stage, batch_size])
                    )

        if self.baseline_mode == "scale":
            model.addConstrs(
                (
                    i[task.name, task.active_variant] == 1
                    for task in self.pipeline.inference_graph
                ),
                name="only-scale-task",
            )
        elif self.baseline_mode == "switch":
            model.addConstrs(
                (
                    n[task.name] == task.replicas
                    for task in self.pipeline.inference_graph
                ),
                name="only-switch-task",
            )
        elif self.baseline_mode == "switch-scale":
            # no batch but ours TODO
            pass
        # elif self.baseline_mode == "switch":
        # only switch TODO
        # pass
        # elif self.baseline_mode == "sclae":
        # only scale TODO
        # pass
        # one variant constraint
        model.addConstrs(
            (
                gp.quicksum(i[stage, variant] for variant in stages_variants[stage])
                == 1
                for stage in stages
            ),
            name="one_model",
        )
        # objectives
        if self.pipeline.accuracy_method == "multiply":
            raise NotImplementedError(
                (
                    "multiplication accuracy objective is not implemented",
                    "yet for Grubi due to quadratic limitation of Gurobi",
                )
            )
        elif self.pipeline.accuracy_method == "sum":
            accuracy_objective = gp.quicksum(
                accuracy_parameters[stage][vairant] * i[stage, vairant]
                for stage in stages
                for vairant in stages_variants[stage]
            )
        elif self.pipeline.accuracy_method == "average":
            accuracy_objective = gp.quicksum(
                accuracy_parameters[stage][vairant]
                * i[stage, vairant]
                * (1 / len(stages))
                for stage in stages
                for vairant in stages_variants[stage]
            )
        else:
            raise ValueError(f"Invalid accuracy method {self.pipeline.accuracy_method}")

        resource_objective = gp.quicksum(
            base_allocations[stage][vairant] * n[stage] * i[stage, vairant]
            for stage in stages
            for vairant in stages_variants[stage]
        )
        if self.only_measured_profiles:
            batch_objective = gp.quicksum(
                batch * b[stage, batch]
                for batch in distinct_batches
                for stage in stages
            )
        else:
            batch_objective = gp.quicksum(b[stage] for stage in stages)

        # update the model
        model.setObjective(
            alpha * accuracy_objective
            - beta * resource_objective
            - gamma * batch_objective,
            GRB.MAXIMIZE,
        )

        # Parameters for retrieving more than one solution
        model.Params.PoolSearchMode = 2
        model.Params.LogToConsole = 0
        # model.Params.PoolSolutions = 10**8
        model.Params.PoolSolutions = num_state_limit
        model.Params.PoolGap = 0.0

        model.update()

        # Solve bilinear model
        model.params.NonConvex = 2
        model.optimize()
        model.write("unmeasured.lp")
        # model.display()
        # model.printStatus()

        # generate states header format
        states = []

        for solution_count in range(model.SolCount):
            model.Params.SolutionNumber = solution_count
            all_vars = {v.varName: v.Xn for v in model.getVars()}
            i_var_output = {
                key: round(value) for key, value in all_vars.items() if "i[" in key
            }
            n_var_output = {
                key: round(value) for key, value in all_vars.items() if "n[" in key
            }
            b_var_output = {
                key: round(value) for key, value in all_vars.items() if "b[" in key
            }

            i_output = {}  # i_output[stage] <- variant
            for stage in stages:
                i_output[stage] = {}
                for variant in stages_variants[stage]:
                    result = [
                        value
                        for key, value in i_var_output.items()
                        if stage in key and variant in key
                    ][0]
                    if result == 1:
                        i_output[stage] = variant

            n_output = {}  # n_output[stage]
            for stage in stages:
                result = [value for key, value in n_var_output.items() if stage in key][
                    0
                ]
                n_output[stage] = result

            if self.only_measured_profiles:
                b_output = {}  # b_output[stage]
                for stage in stages:
                    result = [
                        value for key, value in b_var_output.items() if stage in key
                    ]
                    for index, batch in enumerate(distinct_batches):
                        if result[index] == 1:
                            b_output[stage] = batch
                            break
            else:
                b_output = {}  # b_output[stage]
                for stage in stages:
                    result = [
                        value for key, value in b_var_output.items() if stage in key
                    ][0]
                    b_output[stage] = result

            # set models, replication and batch of inference graph
            for task_id, stage in enumerate(stages):
                self.pipeline.inference_graph[task_id].model_switch(i_output[stage])
                self.pipeline.inference_graph[task_id].re_scale(n_output[stage])
                self.pipeline.inference_graph[task_id].change_batch(b_output[stage])

            # generate states data
            state = {}

            if self.complete_profile:
                for task_id_j in range(self.pipeline.num_nodes):
                    # record all stats under this configs
                    state[f"task_{task_id_j}_latency"] = self.pipeline.inference_graph[
                        task_id_j
                    ].latency
                    state[
                        f"task_{task_id_j}_throughput"
                    ] = self.pipeline.inference_graph[task_id_j].throughput
                    state[
                        f"task_{task_id_j}_throughput_all_replicas"
                    ] = self.pipeline.inference_graph[task_id_j].throughput_all_replicas
                    state[f"task_{task_id_j}_accuracy"] = self.pipeline.inference_graph[
                        task_id_j
                    ].accuracy
                    state[f"task_{task_id_j}_measured"] = self.pipeline.inference_graph[
                        task_id_j
                    ].measured
                    state[
                        f"task_{task_id_j}_cpu_all_replicas"
                    ] = self.pipeline.inference_graph[task_id_j].cpu_all_replicas
                    state[
                        f"task_{task_id_j}_gpu_all_replicas"
                    ] = self.pipeline.inference_graph[task_id_j].gpu_all_replicas
                state["pipeline_accuracy"] = self.pipeline.pipeline_accuracy
                state["pipeline_latency"] = self.pipeline.pipeline_latency
                state["pipeline_throughput"] = self.pipeline.pipeline_throughput
                state["pipeline_cpu"] = self.pipeline.pipeline_cpu
                state["pipeline_gpu"] = self.pipeline.pipeline_gpu
                state["alpha"] = alpha
                state["beta"] = beta
                state["gamma"] = gamma
                state["accuracy_objective"] = self.accuracy_objective()
                state["resource_objective"] = self.resource_objective()
                state["batch_objective"] = self.batch_objective()

            for task_id_j in range(self.pipeline.num_nodes):
                # record all stats under this configs
                state[f"task_{task_id_j}_variant"] = self.pipeline.inference_graph[
                    task_id_j
                ].active_variant
                state[f"task_{task_id_j}_cpu"] = self.pipeline.inference_graph[
                    task_id_j
                ].cpu
                state[f"task_{task_id_j}_gpu"] = self.pipeline.inference_graph[
                    task_id_j
                ].gpu
                state[f"task_{task_id_j}_batch"] = self.pipeline.inference_graph[
                    task_id_j
                ].batch
                state[f"task_{task_id_j}_replicas"] = self.pipeline.inference_graph[
                    task_id_j
                ].replicas

            state.update(self.objective(alpha=alpha, beta=beta, gamma=gamma))
            states.append(state)
        return pd.DataFrame(states)

    def optimize(
        self,
        optimization_method: str,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int = None,
        batching_cap: int = None,
    ) -> pd.DataFrame:
        if optimization_method == "brute-force":
            optimal = self.brute_force(
                scaling_cap=scaling_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
            )
        elif optimization_method == "gurobi":
            optimal = self.gurobi_optmizer(
                scaling_cap=scaling_cap,
                batching_cap=batching_cap,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                arrival_rate=arrival_rate,
                num_state_limit=num_state_limit,
            )
        else:
            raise ValueError(f"Invalid optimization_method: {optimization_method}")
        return optimal

    def can_sustain_load(self, arrival_rate: int) -> bool:
        """
        whether the existing config can sustain a load
        """
        for task in self.pipeline.inference_graph:
            if arrival_rate > task.throughput_all_replicas:
                return False
        return True

    def sla_is_met(self) -> bool:
        return self.pipeline.pipeline_latency < self.pipeline.sla

    def find_load_bottlenecks(self, arrival_rate: int) -> List[int]:
        """
        whether the existing config can sustain a load
        """
        if self.can_sustain_load(arrival_rate=arrival_rate):
            raise ValueError(f"The load can be sustained! no bottleneck!")
        bottlenecks = []
        for task_id, task in enumerate(self.pipeline.inference_graph):
            if arrival_rate > task.throughput_all_replicas:
                bottlenecks.append(task_id)
        return bottlenecks

    def get_one_answer(self) -> Dict:
        """
        Optimizer should return only one feasible answer
        TODO maybe based on the previous state
        TODO find a format for returning each stage's
        active model, replication and batch size
        """
        pass

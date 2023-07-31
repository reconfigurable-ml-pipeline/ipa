from typing import Dict, List, Union
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import math


class ResourceAllocation:
    def __init__(self, cpu: float = 0, gpu: float = 0) -> None:
        # For now only one type CPU/GPU allocation is allowed
        if cpu != 0 and gpu != 0:
            raise ValueError("For now only one of the CPU or GPU allocation is allowed")
        self.cpu = cpu
        self.gpu = gpu


class Profile:
    def __init__(
        self,
        batch: int,
        latency: float,
        measured: bool = True,
        measured_throughput=None,
    ) -> None:
        self.batch = batch
        self.latency = latency
        self.measured = measured
        if measured_throughput is not None:
            self.measured_throughput = measured_throughput

    @property
    def throughput(self):
        if self.measured:
            throughput = self.measured_throughput
        else:
            throughput = (1 / self.latency) * self.batch
        return throughput

    def __eq__(self, other):
        if not isinstance(other, int):
            raise TypeError("batch size variables should be int")
        if other == self.batch:
            return True
        return False


class Model:
    def __init__(
        self,
        name: str,
        resource_allocation: ResourceAllocation,
        measured_profiles: List[Profile],
        only_measured_profiles: bool,
        accuracy: float,
    ) -> None:
        self.resource_allocation = resource_allocation
        self.measured_profiles = measured_profiles
        self.measured_profiles.sort(key=lambda profile: profile.batch)
        self.accuracy = accuracy / 100
        self.name = name
        self.only_measured_profiles = only_measured_profiles
        self.profiles, self.latency_model_params = self.regression_model()

    def regression_model(self) -> Union[List[Profile], Dict[str, float]]:
        """
        interapolate the latency for unknown batch sizes
        """
        train_x = np.array(
            list(map(lambda l: l.batch, self.measured_profiles))
        ).reshape(-1, 1)
        train_y = np.array(
            list(map(lambda l: l.latency, self.measured_profiles))
        ).reshape(-1, 1)
        if self.only_measured_profiles:
            all_x = train_x
        else:
            all_x = np.arange(self.min_batch, self.max_batch + 1)
        # HACK all the data from the latency model and not using
        # measured data
        # test_x = all_x[~np.isin(all_x, train_x)].reshape(-1, 1)
        test_x = all_x.reshape(-1, 1)
        profiles = []
        if self.only_measured_profiles:
            for index, x, y in zip(
                range(len(all_x)), train_x.reshape(-1), train_y.reshape(-1)
            ):
                profiles.append(
                    Profile(
                        batch=x,
                        latency=self.measured_profiles[index].latency,
                        measured=True,
                        measured_throughput=self.measured_profiles[
                            index
                        ].measured_throughput,
                    )
                )
            model_parameters = {"coefficients": None, "intercept": None}
        else:
            poly_features = PolynomialFeatures(degree=2)
            train_x_poly = poly_features.fit_transform(train_x)
            test_x_poly = poly_features.transform(test_x)

            latency_model = LinearRegression()
            latency_model.fit(train_x_poly, train_y)

            test_y = latency_model.predict(test_x_poly)

            # TODO add a hueristic to remove the <0 latency values
            # we set polynomial as reference but for small values
            # polynomial will result into negative values
            # if there is a negative values in the polynomial results
            # we fill it with linear model resutls
            # test_x = all_x.reshape(-1, 1)
            latency_model_linear = LinearRegression()
            latency_model_linear.fit(train_x, train_y)
            test_y_linear = latency_model_linear.predict(test_x)

            for index, lateny in enumerate(test_y):
                if lateny < 0:
                    test_y[index] = test_y_linear[index]

            predicted_profiles = []
            for index, x, y in zip(
                range(len(all_x)), test_x.reshape(-1), test_y.reshape(-1)
            ):
                predicted_profiles.append(
                    Profile(
                        batch=x, latency=y, measured=False, measured_throughput=None
                    )
                )
            profiles: List[Profile] = predicted_profiles
            profiles.sort(key=lambda profile: profile.batch)

            # Extract coefficients and intercept
            coefficients = latency_model.coef_[0]
            intercept = latency_model.intercept_

            model_parameters = {"coefficients": coefficients, "intercept": intercept}

            # HACK only power of twos for now
            # if not self.only_measured_profiles:
            selected_profiles_indices = [
                2**i - 1 for i in range(int(math.log2(len(profiles))) + 1)
            ]
            profiles = [
                profiles[index]
                for index in selected_profiles_indices
                if index < len(profiles)
            ]

        return profiles, model_parameters

    @property
    def profiled_batches(self):
        batches = [profile.batch for profile in self.measured_profiles]
        return batches

    @property
    def min_batch(self):
        return min(self.profiled_batches)

    @property
    def max_batch(self):
        return max(self.profiled_batches)

    @property
    def max_batch(self):
        return max(self.profiled_batches)


class Task:
    def __init__(
        self,
        name: str,
        available_model_profiles: List[Model],
        active_variant: str,
        active_allocation: ResourceAllocation,
        replica: int,
        batch: int,
        allocation_mode: str,
        threshold: int,
        sla_factor: int,
        normalize_accuracy: bool,
        gpu_mode: False,
    ) -> None:
        self.available_model_profiles = available_model_profiles
        self.active_variant = active_variant
        self.active_allocation = active_allocation
        self.initial_allocation = active_allocation
        self.replicas = replica
        self.batch = batch
        self.replicas = replica
        self.gpu_mode = gpu_mode
        self.normalize_accuracy = normalize_accuracy
        self.threshold = threshold
        self.name = name
        self.sla_factor = sla_factor
        self.allocation_mode = allocation_mode

        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == active_variant:
                if self.gpu_mode:
                    if self.active_allocation.gpu == variant.resource_allocation.gpu:
                        self.active_variant_index = variant_index
                        break
                else:
                    if self.active_allocation.cpu == variant.resource_allocation.cpu:
                        self.active_variant_index = variant_index
                        break
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {active_variant} and allocation"
                f" of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}"
            )

    def remove_model_profiles_by_name(self, model_name: str):
        self.available_model_profiles = [
            profile
            for profile in self.available_model_profiles
            if profile.name != model_name
        ]

    def get_all_models_by_name(self, model_name: str):
        return [
            profile
            for profile in self.available_model_profiles
            if profile.name == model_name
        ]

    def add_model_profile(self, model: Model):
        self.available_model_profiles.append(model)

    def add_model_profiles(self, model: List[Model]):
        self.available_model_profiles += model

    def model_switch(self, active_variant: str) -> None:
        """
        changes variant under specific allocation
        """
        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == active_variant:
                if self.gpu_mode:
                    if self.active_allocation.gpu == variant.resource_allocation.gpu:
                        self.active_variant_index = variant_index
                        self.active_variant = active_variant
                        break
                else:
                    if self.active_allocation.cpu == variant.resource_allocation.cpu:
                        self.active_variant_index = variant_index
                        self.active_variant = active_variant
                        break
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {active_variant} and allocation"
                f"of cpu: {self.active_allocation.cpu} and gpu: {self.active_allocation.gpu}"
            )

        if self.allocation_mode == "base":
            self.set_to_base_allocation()

    @property
    def num_variants(self):
        return len(self.variant_names)

    @property
    def sla(self) -> Dict[str, ResourceAllocation]:
        models = {key: [] for key in self.variant_names}
        # 1. filter out models
        for model_variant in self.variant_names:
            for allocation in self.available_model_profiles:
                if allocation.name == model_variant:
                    models[model_variant].append(allocation)
        # 2. find variant SLA
        model_slas = {}
        for model, allocation in models.items():
            # finding sla of each model
            # sla is latency of minimum batch
            # under minimum resource multiplied by
            # a given scaling factor
            # since allocations are sorted the first
            # one will be the one with maximum resource req
            sla = allocation[-1].profiles[0].latency * self.sla_factor
            model_slas[model] = sla
        task_sla = (sum(model_slas.values()) / len(model_slas.values())) * 5
        # task_sla = min(model_slas.values())
        return task_sla

    @property
    def base_allocations(self) -> Dict[str, ResourceAllocation]:
        if self.allocation_mode != "base":
            return None
        models = {key: [] for key in self.variant_names}
        # TOOD change here
        # 1. filter out models
        for model_variant in self.variant_names:
            for allocation in self.available_model_profiles:
                if allocation.name == model_variant:
                    models[model_variant].append(allocation)
        base_allocation = {}
        check_both = {}
        check_sla = {}
        check_throughput = {}
        for model_variant, allocations in models.items():
            check_both[model_variant] = {}
            check_sla[model_variant] = {}
            check_throughput[model_variant] = {}
            # finding the minimum allocation that can respond
            # to the threshold
            # the profiles are sorted therefore therefore
            # we iterate from the first profile
            profiled_batches = allocations[0].profiled_batches
            for allocation in allocations:
                # check if the max batch size throughput
                # can reponsd to the threshold
                check_both[model_variant][allocation.resource_allocation.cpu] = {}
                check_sla[model_variant][allocation.resource_allocation.cpu] = {}
                check_throughput[model_variant][allocation.resource_allocation.cpu] = {}
                # for profile_index in range(len(profiled_batches)-1, -1, -1):
                for profile_index in range(0, len(profiled_batches)):
                    if (
                        allocation.profiles[profile_index].throughput >= self.threshold
                        and allocation.profiles[profile_index].latency <= self.sla
                    ):
                        base_allocation[model_variant] = deepcopy(
                            allocation.resource_allocation
                        )
                        check_both[model_variant][allocation.resource_allocation.cpu][
                            profiled_batches[profile_index]
                        ] = True
                    else:
                        check_both[model_variant][allocation.resource_allocation.cpu][
                            profiled_batches[profile_index]
                        ] = False
                    if allocation.profiles[profile_index].throughput >= self.threshold:
                        check_throughput[model_variant][
                            allocation.resource_allocation.cpu
                        ][profiled_batches[profile_index]] = True
                    else:
                        check_throughput[model_variant][
                            allocation.resource_allocation.cpu
                        ][profiled_batches[profile_index]] = False
                    if allocation.profiles[profile_index].latency <= self.sla:
                        check_sla[model_variant][allocation.resource_allocation.cpu][
                            profiled_batches[profile_index]
                        ] = True
                    else:
                        check_sla[model_variant][allocation.resource_allocation.cpu][
                            profiled_batches[profile_index]
                        ] = False
        allocation_num_sustains = {}
        for model, allocations in check_both.items():
            allocation_num_sustains[model] = {}
            for allocation, batch_can_sustain in allocations.items():
                allocation_num_sustains[model][allocation] = sum(
                    batch_can_sustain.values()
                )
                # TODO 1. add node orders
                # 2. make the heuristic
                # 3. a test
                # 4. if worked, document up
        variant_orders = list(self.variants_accuracies.keys())  # TODO to be fixed
        base_allocation = {}
        indicator = 0
        # former_varaint_indicator = 0
        sample_allocation = list(allocation_num_sustains[variant_orders[0]].keys())
        indicator_to_allocation = {
            key: value
            for key, value in zip(range(len(sample_allocation)), sample_allocation)
        }
        for model in variant_orders:
            allocation_num_sustain = allocation_num_sustains[model]
            base_allocation[model] = None
            while base_allocation[model] == None:
                if indicator > len(sample_allocation):
                    base_allocation[model] = None
                    break
                if model == variant_orders[0]:
                    if allocation_num_sustain[indicator_to_allocation[indicator]] != 0:
                        base_allocation[model] = ResourceAllocation(
                            cpu=indicator_to_allocation[indicator]
                        )
                    else:
                        indicator += 1
                        continue
                else:
                    # if indicator == len(sample_allocation) - 1:
                    #     base_allocation[model] = None
                    #     break
                    if (
                        indicator != len(sample_allocation) - 1
                        and allocation_num_sustain[
                            indicator_to_allocation[indicator + 1]
                        ]
                        > allocation_num_sustain[indicator_to_allocation[indicator]]
                    ):
                        indicator += 1
                    if allocation_num_sustain[indicator_to_allocation[indicator]] == 0:
                        if indicator == len(sample_allocation) - 1:
                            base_allocation[model] = None
                            break
                        else:
                            indicator += 1
                            continue
                    else:
                        base_allocation[model] = ResourceAllocation(
                            cpu=indicator_to_allocation[indicator]
                        )
        for model_variant, allocation in base_allocation.items():
            if allocation == None:
                raise ValueError(
                    f"No responsive model profile to threshold {self.threshold}"
                    f" or model sla {self.sla} was found"
                    f" for model variant {model_variant} "
                    "consider either changing the the threshold or "
                    f"sla factor {self.sla_factor}"
                )
        return base_allocation

    def set_to_base_allocation(self):
        self.change_allocation(
            active_allocation=self.base_allocations[self.active_variant]
        )

    def change_allocation(self, active_allocation: ResourceAllocation) -> None:
        """
        change allocation of a specific variant
        """
        for variant_index, variant in enumerate(self.available_model_profiles):
            if variant.name == self.active_variant:
                if self.gpu_mode:
                    if active_allocation.gpu == variant.resource_allocation.gpu:
                        self.active_variant_index = variant_index
                        self.active_allocation = active_allocation
                        break
                else:
                    if active_allocation.cpu == variant.resource_allocation.cpu:
                        self.active_variant_index = variant_index
                        self.active_allocation = active_allocation
                        break
        else:  # no-break
            raise ValueError(
                f"no matching profile for the variant {self.active_variant} and allocation"
                f"of cpu: {active_allocation.cpu} and gpu: {active_allocation.gpu}"
            )

    def re_scale(self, replica) -> None:
        self.replicas = replica

    def change_batch(self, batch) -> None:
        self.batch = batch

    @property
    def variants_accuracies(self) -> Dict[str, float]:
        """create all the accuracies for each task

        Returns:
            Dict[str, float]: variant accuracies
        """
        variants_accuracies = {}
        for profile in self.available_model_profiles:
            variants_accuracies[profile.name] = profile.accuracy
        variants_accuracies = dict(
            sorted(variants_accuracies.items(), key=lambda l: l[1])
        )
        return variants_accuracies

    @property
    def variants_accuracies_normalized(self) -> Dict[str, float]:
        """create normalized accuracies for each task

        Returns:
            Dict[str, float]: varaint accuracies
        """
        variants = []
        accuracies = []
        for variant, accuracy in self.variants_accuracies.items():
            variants.append(variant)
            accuracies.append(accuracy)
        variants = [variant for _, variant in sorted(zip(accuracies, variants))]
        accuracies.sort()
        if len(accuracies) == 1:
            accuracies_normalized = [1]
        else:
            accuracies_normalized = (
                np.arange(len(accuracies)) / (len(accuracies) - 1)
            ).tolist()
        variants_accuracies_normalized = {
            variant: accuracy_normalized
            for variant, accuracy_normalized in zip(variants, accuracies_normalized)
        }
        return variants_accuracies_normalized

    @property
    def active_model(self) -> Model:
        return self.available_model_profiles[self.active_variant_index]

    @property
    def latency_model_params(self) -> Model:
        return self.available_model_profiles[
            self.active_variant_index
        ].latency_model_params

    @property
    def cpu(self) -> int:
        if self.gpu_mode:
            raise ValueError("The node is on gpu mode")
        else:
            return self.active_model.resource_allocation.cpu

    @property
    def gpu(self) -> float:
        if self.gpu_mode:
            return self.active_model.resource_allocation.gpu
        else:
            return 0

    @property
    def cpu_all_replicas(self) -> int:
        if self.gpu_mode:
            raise ValueError("The node is on gpu mode")
        else:
            return self.active_model.resource_allocation.cpu * self.replicas

    @property
    def gpu_all_replicas(self) -> float:
        if self.gpu_mode:
            return self.active_model.resource_allocation.gpu * self.replicas
        return 0

    @property
    def queue_latency(self) -> float:
        # TODO TEMP
        queue_latency = 0
        return queue_latency

    @property
    def model_latency(self) -> float:
        latency = next(
            filter(
                lambda profile: profile.batch == self.batch, self.active_model.profiles
            )
        ).latency
        return latency

    @property
    def latency(self) -> float:
        latency = self.model_latency + self.queue_latency
        return latency

    @property
    def throughput(self) -> float:
        throughput = next(
            filter(
                lambda profile: profile.batch == self.batch, self.active_model.profiles
            )
        ).throughput
        return throughput

    @property
    def measured(self) -> bool:
        measured = next(
            filter(
                lambda profile: profile.batch == self.batch, self.active_model.profiles
            )
        ).measured
        return measured

    @property
    def throughput_all_replicas(self):
        return self.throughput * self.replicas

    @property
    def accuracy(self):
        if self.normalize_accuracy:
            return self.variants_accuracies_normalized[self.active_variant]
        else:
            return self.active_model.accuracy

    @property
    def variant_names(self):
        return list(set(map(lambda l: l.name, self.available_model_profiles)))

    @property
    def batches(self):
        batches = list(map(lambda l: l.batch, self.active_model.profiles))
        return batches

    @property
    def resource_allocations_cpu_mode(self):
        cpu_allocations = list(
            set(
                list(
                    map(
                        lambda l: l.resource_allocation.cpu,
                        self.available_model_profiles,
                    )
                )
            )
        )
        resource_allocations = list(
            map(lambda l: ResourceAllocation(cpu=l), cpu_allocations)
        )
        return resource_allocations

    @property
    def resource_allocations_gpu_mode(self):
        gpu_allocations = list(
            set(
                list(
                    map(
                        lambda l: l.resource_allocation.gpu,
                        self.available_model_profiles,
                    )
                )
            )
        )
        resource_allocations = list(
            map(lambda l: ResourceAllocation(gpu=l), gpu_allocations)
        )
        return resource_allocations


class Pipeline:
    def __init__(
        self,
        inference_graph: List[Task],
        gpu_mode: bool,
        sla_factor: int,
        accuracy_method: str,
        normalize_accuracy: bool,
    ) -> None:
        self.inference_graph: List[Task] = inference_graph
        self.gpu_mode = gpu_mode
        self.sla_factor = sla_factor
        self.accuracy_method = accuracy_method
        self.normalize_accuracy = normalize_accuracy
        if not self.gpu_mode:
            for task in self.inference_graph:
                if task.gpu_mode:
                    raise ValueError(
                        f"pipeline is deployed on cpu",
                        f"but task {task.name} is on gpu",
                    )

    def add_task(self, task: Task):
        self.inference_graph.append(task)

    def remove_task(self):
        self.inference_graph.pop()

    @property
    def stage_wise_throughput(self):
        throughputs = list(
            map(lambda l: l.throughput_all_replicas, self.inference_graph)
        )
        return throughputs

    @property
    def stage_wise_latencies(self):
        latencies = list(map(lambda l: l.latency, self.inference_graph))
        return latencies

    @property
    def sla(self):
        sla = sum(map(lambda l: l.sla, self.inference_graph))
        return sla

    @property
    def stage_wise_slas(self):
        slas = dict(map(lambda l: (l.name, l.sla), self.inference_graph))
        return slas

    @property
    def stage_wise_accuracies(self):
        latencies = list(map(lambda l: l.accuracy, self.inference_graph))
        return latencies

    @property
    def stage_wise_replicas(self):
        replicas = list(map(lambda l: l.replicas, self.inference_graph))
        return replicas

    @property
    def stage_wise_cpu(self):
        cpu = []
        for task in self.inference_graph:
            if not task.gpu_mode:
                cpu.append(task.cpu_all_replicas)
            else:
                cpu.append(0)
        return cpu

    @property
    def stage_wise_gpu(self):
        gpu = []
        for task in self.inference_graph:
            if task.gpu_mode:
                gpu.append(task.gpu_all_replicas)
            else:
                gpu.append(0)
        return gpu

    @property
    def stage_wise_task_names(self):
        task_names = []
        for task in self.inference_graph:
            task_names.append(task.name)
        return task_names

    @property
    def stage_wise_available_variants(self):
        task_names = {}
        for task in self.inference_graph:
            task_names[task.name] = task.variant_names
        return task_names

    @property
    def pipeline_cpu(self):
        return sum(self.stage_wise_cpu)

    @property
    def pipeline_gpu(self):
        return sum(self.stage_wise_gpu)

    @property
    def pipeline_latency(self):
        return sum(self.stage_wise_latencies)

    @property
    def pipeline_accuracy(self):
        tasks_accuracies = {}
        for task in self.inference_graph:
            acive_variant = task.active_variant
            if self.normalize_accuracy:
                accuracy = task.variants_accuracies_normalized[acive_variant]
            else:
                accuracy = task.variants_accuracies[acive_variant]
            tasks_accuracies[acive_variant] = accuracy
        if self.accuracy_method == "multiply":
            accuracy = 1
            for task, task_accuracy in tasks_accuracies.items():
                accuracy *= task_accuracy
        elif self.accuracy_method == "sum":
            accuracy = 0
            for task, task_accuracy in tasks_accuracies.items():
                accuracy += task_accuracy
        elif self.accuracy_method == "average":
            accuracy = 0
            for task, task_accuracy in tasks_accuracies.items():
                accuracy += task_accuracy
            accuracy /= len(self.inference_graph)
        return accuracy

    @property
    def pipeline_throughput(self):
        return min(self.stage_wise_throughput)

    @property
    def cpu_usage(self):
        return sum(self.stage_wise_cpu)

    @property
    def gpu_usage(self):
        return sum(self.stage_wise_gpu)

    @property
    def num_nodes(self):
        return len(self.inference_graph)

    def visualize(self):
        pass

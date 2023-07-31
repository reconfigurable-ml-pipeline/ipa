import os
import sys
from models import Model, ResourceAllocation, Profile, Task, Pipeline
from optimizer import Optimizer

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

from experiments.utils.constants import PIPELINE_SIMULATION_MOCK_PATH


sla_factor = 10
normalize_accuracy = True
threshold = 5
allocation_mode = "base"
gpu_mode = False
optimization_method = "gurobi"
accuracy_method = "sum"
scaling_cap = 2
sla = 5
arrival_rate = 10
alpha = 0.5
beta = 0.5
gamma = 0.5
num_state_limit = 100000000

# ---------- first task ----------
task_a_model_1 = Model(
    name="yolo5n",
    resource_allocation=ResourceAllocation(cpu=1),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5,
)

task_a_model_2 = Model(
    name="yolo5n",
    resource_allocation=ResourceAllocation(cpu=2),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5,
)

task_a_model_3 = Model(
    name="yolo5s",
    resource_allocation=ResourceAllocation(cpu=1),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8,
)

task_a_model_4 = Model(
    name="yolo5s",
    resource_allocation=ResourceAllocation(cpu=2),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8,
)

task_a = Task(
    name="crop",
    available_model_profiles=[
        task_a_model_1,
        task_a_model_2,
        task_a_model_3,
        task_a_model_4,
    ],
    active_variant="yolo5s",
    active_allocation=ResourceAllocation(cpu=2),
    replica=2,
    batch=1,
    sla_factor=sla_factor,
    normalize_accuracy=normalize_accuracy,
    threshold=threshold,
    allocation_mode=allocation_mode,
    gpu_mode=gpu_mode,
)

# ---------- second task ----------
task_b_model_1 = Model(
    name="resnet18",
    resource_allocation=ResourceAllocation(cpu=1),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5,
)

task_b_model_2 = Model(
    name="resnet18",
    resource_allocation=ResourceAllocation(cpu=2),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5,
)

task_b_model_3 = Model(
    name="resnet34",
    resource_allocation=ResourceAllocation(cpu=1),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8,
)

task_b_model_4 = Model(
    name="resnet34",
    resource_allocation=ResourceAllocation(cpu=2),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8,
)

task_b = Task(
    name="classification",
    available_model_profiles=[
        task_b_model_1,
        task_b_model_2,
        task_b_model_3,
        task_b_model_4,
    ],
    active_variant="resnet34",
    active_allocation=ResourceAllocation(cpu=2),
    replica=2,
    batch=1,
    sla_factor=sla_factor,
    normalize_accuracy=normalize_accuracy,
    threshold=threshold,
    allocation_mode=allocation_mode,
    gpu_mode=gpu_mode,
)

inference_graph = [task_a, task_b]

pipeline = Pipeline(
    inference_graph=inference_graph,
    gpu_mode=gpu_mode,
    sla_factor=sla_factor,
    accuracy_method=accuracy_method,
    normalize_accuracy=normalize_accuracy,
)

optimizer = Optimizer(pipeline=pipeline, allocation_mode=allocation_mode)

print(f"{pipeline.stage_wise_throughput = }")
print(f"{pipeline.stage_wise_latencies = }")
print(f"{pipeline.stage_wise_replicas = }")
print(f"{pipeline.stage_wise_cpu = }")
print(f"{pipeline.stage_wise_gpu = }")
print(f"{pipeline.cpu_usage = }")
print(f"{pipeline.gpu_usage = }")
print(f"{pipeline.pipeline_latency = }")

print(f"{optimizer.can_sustain_load(arrival_rate=4) = }")
print(f"{optimizer.find_load_bottlenecks(arrival_rate=30) = }")
print(f"{optimizer.objective(alpha=alpha, beta=beta, gamma=gamma) = }")

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
print(f"{states = }")
states.to_markdown(os.path.join(PIPELINE_SIMULATION_MOCK_PATH, "all-states.csv"))

# all feasibla states
all_states = optimizer.all_states(
    check_constraints=True,
    scaling_cap=scaling_cap,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    arrival_rate=arrival_rate,
    num_state_limit=num_state_limit,
)
print(f"{all_states = }")
all_states.to_markdown(os.path.join(PIPELINE_SIMULATION_MOCK_PATH, "feasible.csv"))

optimization_method = "brute-force"
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
optimal.to_markdown(
    os.path.join(PIPELINE_SIMULATION_MOCK_PATH, f"optimal_{optimization_method}.csv")
)

optimization_method = "gurobi"
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
optimal.to_markdown(
    os.path.join(PIPELINE_SIMULATION_MOCK_PATH, f"optimal_{optimization_method}.csv")
)

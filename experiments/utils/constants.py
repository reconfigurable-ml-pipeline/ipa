import os
from .obj import setup_obj_store

# defined by the user
# TODO make the user cc a variable
PROJECT_PATH = "~/ipa"
OBJ_PATH = "/home/cc/my_mounting_point/"  # object store path
KEY_CONFIG_FILENAME = "key_config_mapper.csv"
NAMESPACE = "default"

# base DATA folder path and object sore path
DATA_PATH = os.path.join(PROJECT_PATH, "data")
OBJ_DATA_PATH = os.path.join(OBJ_PATH, "data")
# PIPELINES_FOLDER = "mlserver-final"
# PIPELINES_FOLDER = "mlserver-centralized"

# pipelines path
# PIPLINES_PATH = os.path.join(PROJECT_PATH, "pipelines", PIPELINES_FOLDER)
PIPLINES_PATH = os.path.join(PROJECT_PATH, "pipelines")
CONFIGS_PATH = os.path.join(DATA_PATH, "configs")

# router path
ROUTER_PATH = os.path.join(PIPLINES_PATH, "router")

# queue path
QUEUE_PATH = os.path.join(PIPLINES_PATH, "queue")

# logging constants
LOGGING_LEVEL: str = "info"
LOG_TO = "print"

# accuracies files
ACCURACIES_PATH = os.path.join(CONFIGS_PATH, "accuracies.yaml")

# models metadata file
MODELS_METADATA_PATH = os.path.join(CONFIGS_PATH, "models-metadata.yaml")

# profiling configs
PROFILING_CONFIGS_PATH = os.path.join(CONFIGS_PATH, "profiling")
NODE_PROFILING_CONFIGS_PATH = os.path.join(PROFILING_CONFIGS_PATH, "nodes")
PIPELINE_PROFILING_CONFIGS_PATH = os.path.join(PROFILING_CONFIGS_PATH, "pipelines")

# pipeline simulation config path
PIPELINE_SIMULATION_CONFIGS_PATH = os.path.join(CONFIGS_PATH, "pipeline-simulation")

# pipeline simulation config path
FINAL_CONFIGS_PATH = os.path.join(CONFIGS_PATH, "final")

# lstm load predictor model path
LSTM_PATH = os.path.join(DATA_PATH, "lstm")
LSTM_INPUT_SIZE = 12

# triton folders
# TRITON_PROFILING_PATH = os.path.join(PROFILING_CONFIGS_PATH, "triton")
# TRITON_PROFILING_CONFIGS_PATH = os.path.join(TRITON_PROFILING_PATH, "configs")
# TRITON_PROFILING_TEMPLATES_PATH = os.path.join(TRITON_PROFILING_PATH, "templates")

# results noraml path
RESULTS_PATH = os.path.join(DATA_PATH, "results")
PROFILING_RESULTS_PATH = os.path.join(RESULTS_PATH, "profiling")
FIGURES_PATH = os.path.join(DATA_PATH, "figures")
PIPELINE_SIMULATION_RESULTS_PATH = os.path.join(RESULTS_PATH, "pipeline-simulation")
FINAL_RESULTS_PATH = os.path.join(RESULTS_PATH, "final")
PIPELINE_SIMULATION_MOCK_PATH = os.path.join(
    PIPELINE_SIMULATION_RESULTS_PATH, "mock-simulation"
)
NODE_PROFILING_RESULTS_PATH = os.path.join(PROFILING_RESULTS_PATH, "nodes")
# NODE_PROFILING_RESULTS_TRITON_PATH = os.path.join(NODE_PROFILING_RESULTS_PATH, "triton")
PIPELINE_PROFILING_RESULTS_PATH = os.path.join(PROFILING_RESULTS_PATH, "pipelines")

# results object storage path
OBJ_RESULTS_PATH = os.path.join(OBJ_DATA_PATH, "results")
OBJ_PROFILING_RESULTS_PATH = os.path.join(OBJ_RESULTS_PATH, "profiling")
OBJ_PIPELINE_SIMULATION_RESULTS_PATH = os.path.join(RESULTS_PATH, "pipeline-simulation")
OBJ_FINAL_RESULTS_PATH = os.path.join(OBJ_RESULTS_PATH, "final")
OBJ_PIPELINE_PROFILING_RESULTS_PATH = os.path.join(
    OBJ_PROFILING_RESULTS_PATH, "pipelines"
)

# generated baesd on the users' path

TEMP_MODELS_PATH = os.path.join(DATA_PATH, "model-temp")
DATASETS = os.path.join(DATA_PATH, "datasets")
# KUBE_YAMLS_PATH = os.path.join(DATA_PATH, "yamls")
# PIPELINES_MODELS = os.path.join(DATA_PATH, "pipeline-test-meta")


def create_dirs():
    """
    create directories if they don't exist
    """
    if not os.path.exists(TEMP_MODELS_PATH):
        os.makedirs(TEMP_MODELS_PATH)
    # if not os.path.exists(KUBE_YAMLS_PATH):
    #     os.makedirs(KUBE_YAMLS_PATH)
    # if not os.path.exists(PIPELINES_MODELS):
    #     os.makedirs(PIPELINES_MODELS)
    # if not os.path.exists(PIPLINES_PATH):
    #     os.makedirs(PIPLINES_PATH)
    if not os.path.exists(LSTM_PATH):
        os.makedirs(LSTM_PATH)
    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)
    if not os.path.exists(CONFIGS_PATH):
        os.makedirs(CONFIGS_PATH)
    if not os.path.exists(PROFILING_CONFIGS_PATH):
        os.makedirs(PROFILING_CONFIGS_PATH)
    if not os.path.exists(NODE_PROFILING_CONFIGS_PATH):
        os.makedirs(NODE_PROFILING_CONFIGS_PATH)
    if not os.path.exists(PIPELINE_PROFILING_CONFIGS_PATH):
        os.makedirs(PIPELINE_PROFILING_CONFIGS_PATH)
    if not os.path.exists(PIPELINE_SIMULATION_CONFIGS_PATH):
        os.makedirs(PIPELINE_SIMULATION_CONFIGS_PATH)
    if not os.path.exists(FINAL_CONFIGS_PATH):
        os.makedirs(FINAL_CONFIGS_PATH)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    if not os.path.exists(PROFILING_RESULTS_PATH):
        os.makedirs(PROFILING_RESULTS_PATH)
    if not os.path.exists(PIPELINE_SIMULATION_RESULTS_PATH):
        os.makedirs(PIPELINE_SIMULATION_RESULTS_PATH)
    if not os.path.exists(FINAL_RESULTS_PATH):
        os.makedirs(FINAL_RESULTS_PATH)
    if not os.path.exists(NODE_PROFILING_RESULTS_PATH):
        os.makedirs(NODE_PROFILING_RESULTS_PATH)
    if not os.path.exists(NODE_PROFILING_RESULTS_PATH):
        os.makedirs(NODE_PROFILING_RESULTS_PATH)
    # if not os.path.exists(NODE_PROFILING_RESULTS_TRITON_PATH):
    #     os.makedirs(NODE_PROFILING_RESULTS_TRITON_PATH)
    if not os.path.exists(PIPELINE_PROFILING_RESULTS_PATH):
        os.makedirs(PIPELINE_PROFILING_RESULTS_PATH)
    if not os.path.exists(OBJ_RESULTS_PATH):
        setup_obj_store()
        os.makedirs(OBJ_RESULTS_PATH)
    if not os.path.exists(OBJ_PROFILING_RESULTS_PATH):
        setup_obj_store()
        os.makedirs(OBJ_PROFILING_RESULTS_PATH)
    if not os.path.exists(OBJ_PIPELINE_PROFILING_RESULTS_PATH):
        setup_obj_store()
        os.makedirs(OBJ_PIPELINE_PROFILING_RESULTS_PATH)
    if not os.path.exists(DATASETS):
        os.makedirs(DATASETS)

# prometheus client
PROMETHEUS = "http://localhost:30090"

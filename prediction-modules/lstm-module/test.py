import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from tensorflow.keras.models import load_model

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from barazmoon.twitter import twitter_workload_generator

from experiments.utils.constants import PROJECT_PATH, LSTM_PATH, LSTM_INPUT_SIZE
from train import get_x_y  # , input_size

fig_path = os.path.join(
    PROJECT_PATH, "prediction-modules", "lstm-module", "lstm_prediction.png"
)

model = load_model(LSTM_PATH)
last_day = 21 * 24 * 3600
workload = twitter_workload_generator(f"0-{last_day}", damping_factor=5)
workload = list(filter(lambda x: x != 0, workload))  # for removing missing hours
hour = 60 * 60
day = hour * 24

# pick up the untrained part of the dataset
test_idx = 18 * day
test_data = workload[test_idx : test_idx + 2 * hour]

# test_idx = 1862800
# test_end = 1863300
# test_data = workload[test_idx : test_end]

test_x, test_y = get_x_y(test_data)
test_x = tf.convert_to_tensor(
    np.array(test_x).reshape((-1, LSTM_INPUT_SIZE, 1)), dtype=tf.float32
)
prediction = model.predict(test_x)

plt.plot(list(range(len(test_y))), list(test_y), label="real values")
plt.plot(list(range(len(test_y))), list(prediction), label="predictions")
plt.xlabel("time (minute)")
plt.ylabel("load (RPS)")
plt.legend()
plt.savefig(fig_path)

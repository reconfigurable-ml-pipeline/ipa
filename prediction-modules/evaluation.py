import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

from barazmoon.twitter import twitter_workload_generator

from experiments.utils.constants import PROJECT_PATH, LSTM_PATH, LSTM_INPUT_SIZE


def smape(y_true, y_pred):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        actual (np.ndarray or list): Array of actual values.
        predicted (np.ndarray or list): Array of predicted values.

    Returns:
        float: The SMAPE value.
    """

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    return np.mean(numerator / denominator) * 100


fig_path = os.path.join(
    PROJECT_PATH, "prediction-modules", "lstm-module", "lstm_prediction.png"
)


# TODO put it in somewhere centralized
def get_x_y(data):
    """
    For each 60 seconds it taeks the max of last 60 seconds
    and returns an output with length of len(data)/60 that
    each entry is the maximum rps in each aggregated 60 seconds
    x: series of max of every 1 minute
    y: target of the 10 minutes
    """
    x = []
    y = []
    history_seconds = 120
    step = 10
    for i in range(0, len(data) - history_seconds, step):
        t = data[i : i + history_seconds]
        for j in range(0, len(t), step):
            x.append(max(t[j : j + step]))
        y.append(max(data[i + history_seconds : i + history_seconds + 2 * step]))
    return x, y


# ----------------------- LSTM -----------------------

model = load_model(LSTM_PATH)
# first_day = 15
# last_day = 21 * 24 * 3600
first_day = 1296000
last_day = first_day + 20 * 60

workload = twitter_workload_generator(f"{first_day}-{last_day}", damping_factor=1)
workload = list(filter(lambda x: x != 0, workload))  # for removing missing hours
hour = 60 * 60
day = hour * 24

# pick up the untrained part of the dataset
# test_idx = 18 * day
# test_data = workload[test_idx : test_idx + 2 * hour]
test_data = workload

# test_idx = 1862800
# test_end = 1863300
# test_data = workload[test_idx : test_end]

test_x, test_y = get_x_y(test_data)
test_x = tf.convert_to_tensor(
    np.array(test_x).reshape((-1, LSTM_INPUT_SIZE, 1)), dtype=tf.float32
)
lstm_prediction = model.predict(test_x)

print(
    f"lstm mean_absolute_percentage_error: {mean_absolute_percentage_error(y_true=test_y, y_pred=lstm_prediction)}"
)
print(
    f"lstm Symmetric Mean Absolute Percentage Error (SMAPE): {smape(y_true=test_y, y_pred=lstm_prediction)}"
)

# ----------------------- ARIMA -----------------------

arima_prediction = []
for x in test_x:
    model = ARIMA(list(x), order=(1, 0, 0))
    model_fit = model.fit()
    pred = int(max(model_fit.forecast(steps=2)))  # max
    arima_prediction.append(pred)

print(
    f"arima mean_absolute_percentage_error: {mean_absolute_percentage_error(y_true=test_y, y_pred=arima_prediction)}"
)
print(
    f"arima Symmetric Mean Absolute Percentage Error (SMAPE): {smape(y_true=test_y, y_pred=arima_prediction)}"
)


# plt.plot(lstm_prediction, label='lstm')
# plt.plot(arima_prediction, label='arima')
# plt.plot(test_y, label='real')
# plt.legend()
# plt.savefig("eval.png")

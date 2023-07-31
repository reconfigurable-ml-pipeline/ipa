import os
import datetime

TESTS_PATH = os.path.dirname(__file__)
SECONDS_PER_DAY = 30 * 60 + 23 * 3600

# twitter dataset extracted from
path = os.path.join(TESTS_PATH, "workload.txt")


def get_workload():
    with open(path, "r") as f:
        data = f.read()
        workload = data.split(" ")
        return_workload = []
        for i, w in enumerate(workload):
            try:
                return_workload.append(int(w))
            except:
                pass
        return return_workload


def parse_date_str(date_str):
    parts = list(map(int, date_str.split(":")))
    if len(parts) == 3:
        # day:hour:minute format
        day, hour, minute = parts
        return datetime.datetime(2023, 1, day, hour, minute)
    else:
        # day:hour:minute:second format
        day, hour, minute, second = parts
        return datetime.datetime(2023, 1, day, hour, minute, second)


def twitter_workload_generator(days, damping_factor=None):
    workload_all = get_workload()
    if ":" not in days:
        # returns by days
        first, end = list(map(int, days.split("-")))
        # first = (first - 1) * seconds_per_day
        # end = end * seconds_per_day
        if damping_factor is not None:
            workload_all = list(map(lambda l: int(l/damping_factor), workload_all))
        return workload_all[first: end]

    else:
        # return by full date
        start_date_str, end_date_str = days.split("-")
        start_date = parse_date_str(start_date_str)
        end_date = parse_date_str(end_date_str)
        start_index = (
            start_date - datetime.datetime(2023, 1, 1, 0, 0, 0)
        ).total_seconds()
        end_index = (end_date - datetime.datetime(2023, 1, 1, 0, 0, 0)).total_seconds()
        if damping_factor is not None:
            workload_all = list(map(lambda l: int(l/damping_factor), workload_all))
        return workload_all[int(start_index) : int(end_index) + 1]

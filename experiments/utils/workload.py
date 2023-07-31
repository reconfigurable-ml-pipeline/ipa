from barazmoon.twitter import twitter_workload_generator


def make_workload(config: dict, teleport_interval: int = None):
    workload_type = config["workload_type"]
    workload_config = config["workload_config"]

    if workload_type == "static":
        loads_to_test = workload_config["loads_to_test"]
        load_duration = workload_config["load_duration"]
        workload = [loads_to_test] * load_duration
    elif workload_type == "twitter":
        # load_duration = 0
        workload = []
        # ----------
        # loads_to_test = []
        for w_config in workload_config:
            damping_factor = w_config["damping_factor"]
            start = w_config["start"]
            if teleport_interval is not None:
                start = str(int(start) + teleport_interval)
            end = w_config["end"]
            if start > end:
                raise ValueError(
                    f"start of workload {start} cannot be larger than {end}"
                )
            load_to_test = start + "-" + end
            # loads_to_test.append(load_to_test)
            workload += twitter_workload_generator(
                load_to_test, damping_factor=damping_factor
            )
        load_duration = len(workload)
    return load_duration, workload

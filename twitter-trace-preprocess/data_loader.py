import bz2
import json
from builder_req import build_workload
import datetime


def create_workload(day="25"):
    # tar = tarfile.open("twitter-2018-04-25.tar")
    # days = set()
    # print(len([i for i in tar.getmembers() if i.name.endswith("bz2")]))
    # my_test_day = day
    # for file in tar.getmembers():
    #     if file.name.endswith("bz2"):
    #         day = file.name.split("/")[2]
    #         if day == my_test_day:
    #             print(file.name)
    #             days.add(file)

    time_series = {}

    file_path = "../../twitter-trace-2018-04/"
    from os import listdir
    from os.path import isfile, join

    days = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    for member in days:
        # f=tar.extractfile(member)
        if "do" in member:
            continue
        with bz2.open(file_path + member, "rt") as bzinput:
            for i, line in enumerate(bzinput):
                tweets = json.loads(line)

                if len(tweets) > 0:
                    if "created_at" in tweets.keys():
                        time = datetime.datetime.fromtimestamp(
                            int(tweets["timestamp_ms"]) // 1000.0
                        ).__str__()
                        if time in time_series.keys():
                            time_series[time] += 1

                        else:
                            time_series[time] = 1
                    if "delete" in tweets.keys():
                        time = datetime.datetime.fromtimestamp(
                            int(tweets["delete"]["timestamp_ms"]) // 1000.0
                        ).__str__()
                        if time in time_series.keys():
                            time_series[time] += 1
                        else:
                            time_series[time] = 1
    print("reading complete")

    with open("data.json", "w") as fp:
        json.dump(time_series, fp, indent=4)
    build_workload()


create_workload()

import json


def build_workload():
    with open("data.json") as json_file:
        data = json.load(json_file)

    times = [0 for i in range(3600 * 24)]

    for key in data.keys():
        time = key.split()[1]
        hour, minute, second = map(int, time.split(":"))
        times[hour * 3600 + minute * 60 + second] = data[key]

    file = open("workload.txt", "w")

    # Saving the array in a text file
    content = str(times)
    file.write(content)
    file.close()

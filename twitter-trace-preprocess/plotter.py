import numpy as np
import matplotlib.pyplot as plt


def preprocess(contents):
    data = contents[0].split(", ")
    for d in range(len(data)):
        if "[" in data[d]:
            data[d] = data[d].replace("[", "")
        if "]" in data[d]:
            data[d] = data[d].replace("]", "")
        data[d] = int(data[d])
    return data


def timer(data):
    starter = 0
    end = 0
    flag_start = False
    for d in range(len(data)):
        if data[d] > 0 and flag_start == False:
            print(data[d], d)
            starter = d
            flag_start = True

        if sum(data[d:]) == 0:
            end = d
            break
    return starter, end


with open("workload.txt") as f:
    contents_workload = f.readlines()

data_workload = preprocess(contents_workload)

with open("workload_delete.txt") as f:
    contents_delete = f.readlines()

data_delete = preprocess(contents_delete)


with open("workload_create.txt") as f:
    contents_create = f.readlines()

data_create = preprocess(contents_create)

sw, ew = timer(data_workload)
sd, ed = timer(data_delete)
sc, ec = timer(data_create)

s, e = min(sw, sd, sc) - 10, max(ew, ed, ec) + 10
x = np.arange(s, e)
plt.plot(x, data_workload[s:e], label="workload")

plt.plot(x, data_delete[s:e], label="delete")

plt.plot(x, data_create[s:e], label="create")
plt.legend(loc="upper right")

plt.savefig("plts.png")

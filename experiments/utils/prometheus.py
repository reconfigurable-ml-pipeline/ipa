import requests
from .constants import PROMETHEUS


class PromClient:
    def __init__(self) -> None:
        pass

    def prom_response_postprocess(self, response):
        try:
            response = response.json()["data"]["result"]
        except:
            print(response.json())
            exit()
        plot_values = [[] for _ in range(len(response))]
        times = [[] for _ in range(len(response))]
        try:
            for val in range(len(response)):
                data = response[val]["values"]
                for d in data:
                    plot_values[val].append((float(d[1])))
                    times[val].append(float(d[0]))
            output = plot_values[0], times[0]
        except IndexError:
            output = None, None
        return output

    def get_memory_usage(
        self,
        pod_name: str,
        namespace: str,
        container: str,
        duration: int,
        need_max: bool = False,
    ):
        query = f"container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{namespace}'}}[{duration}m:1s]"
        if need_max:
            query = f"max_over_time(container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{namespace}'}}[{duration}m])"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_cpu_usage_count(
        self, pod_name: str, namespace: str, container: str, duration: int
    ):
        query = f"container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_cpu_usage_rate(
        self, pod_name: str, namespace: str, container: str, duration: int, rate=120
    ):
        query = f"rate(container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_cpu_throttled_count(
        self, pod_name: str, namespace: str, container: str, duration: int
    ):
        query = f"container_cpu_cfs_throttled_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_cpu_throttled_rate(
        self, pod_name: str, namespace: str, container: str, duration: int, rate=120
    ):
        query = f"rate(container_cpu_cfs_throttled_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_request_per_second(
        self, pod_name, namespace, container, duration, rate=120
    ):
        query = f"rate(model_infer_request_duration_count{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_input_rps(self, pod_name, namespace, container, duration, rate=120):
        query = f"rate(input_requests_count{{pod='{pod_name}', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)


class TritonNodePromClient:
    def __init__(self) -> None:
        pass

    def prom_response_postprocess(self, response):
        try:
            response = response.json()["data"]["result"]
        except:
            print(response.json())
            exit()
        plot_values = [[] for _ in range(len(response))]
        times = [[] for _ in range(len(response))]
        try:
            for val in range(len(response)):
                data = response[val]["values"]
                for d in data:
                    plot_values[val].append((float(d[1])))
                    times[val].append(float(d[0]))
            output = plot_values[0], times[0]
        except IndexError:
            output = None, None
        return output

    def get_memory_usage(
        self,
        pod_name: str,
        namespace: str,
        container: str,
        duration: int,
        need_max: bool = False,
    ):
        query = f"container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{namespace}'}}[{duration}m:1s]"
        if need_max:
            query = f"max_over_time(container_memory_usage_bytes{{pod=~'{pod_name}.*', container='{container}', namespace='{namespace}'}}[{duration}m])"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_cpu_usage_count(
        self, pod_name: str, namespace: str, container: str, duration: int
    ):
        query = f"container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_cpu_usage_rate(
        self, pod_name: str, namespace: str, container: str, duration: int, rate=120
    ):
        query = f"rate(container_cpu_usage_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_cpu_throttled_count(
        self, pod_name: str, namespace: str, container: str, duration: int
    ):
        query = f"container_cpu_cfs_throttled_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_cpu_throttled_rate(
        self, pod_name: str, namespace: str, container: str, duration: int, rate=120
    ):
        query = f"rate(container_cpu_cfs_throttled_seconds_total{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

    def get_request_per_second(
        self, pod_name, namespace, container, duration, rate=120
    ):
        query = f"rate(model_infer_request_duration_count{{pod=~'{pod_name}.*', namespace='{namespace}', container='{container}'}}[{rate}s])[{duration}m:1s]"
        response = requests.get(PROMETHEUS + "/api/v1/query", params={"query": query})
        return self.prom_response_postprocess(response)

"""
Experiment Metrics Collector
============================

Collects metrics during NSGD experiments for:
  1. Paper figures (θ convergence, cost comparison, sensitivity)
  2. Fair comparison with LSTM-PPO using the same observables env.py collects

Each periodic sample computes BOTH:
  - nsgd_cost:      equation (7) from the paper
  - lstm_ppo_reward: exact reward function from env.py _calculate_reward()

Data is persisted automatically to CSV:
  {log_dir}/samples.csv       — periodic aggregates (every sampling_window seconds)
  {log_dir}/request_log.csv   — per-request data (state, cost, theta, latency)
  {log_dir}/monitor.csv       — NSGD state samples (every monitor_interval seconds)

Read in Jupyter:
  import pandas as pd
  samples = pd.read_csv("logs/nsgd/samples.csv")
  requests = pd.read_csv("logs/nsgd/request_log.csv")
"""

import os
import csv
import math
import time
import logging
import threading
from datetime import datetime
from dataclasses import dataclass, asdict, fields

logger = logging.getLogger("experiment-metrics")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PeriodicSample:
    """One periodic sample — same observables as LSTM-PPO env.py."""
    timestamp: float = 0.0
    # ---- env.py observation vector ----
    avg_execution_time: float = 0.0
    throughput: float = 100.0
    requests: int = 0
    replicas: int = 0
    avg_cpu: float = 0.0
    avg_mem: float = 0.0
    # ---- NSGD state (paper equation 7) ----
    cold: int = 0
    idle_on: int = 0
    busy: int = 0
    init_total: int = 0
    init_reserved: int = 0
    nsgd_cost: float = 0.0
    has_rejection: bool = False
    # ---- LSTM-PPO reward (env.py _calculate_reward) ----
    lstm_ppo_reward: float = 0.0
    reward_throughput: float = 0.0
    reward_cpu: float = 0.0
    reward_mem: float = 0.0
    reward_replicas: float = 0.0
    # ---- from proxy ----
    proxy_requests: int = 0
    proxy_rejections: int = 0
    avg_response_latency_ms: float = 0.0
    p99_response_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# LSTM-PPO Reward Function (exact copy from env.py)
# ---------------------------------------------------------------------------

def compute_lstm_ppo_reward(
    throughput: float,
    avg_cpu: float,
    avg_mem: float,
    replicas: int,
    min_pods: int = 1,
) -> dict:
    """
    Exact reward function from env.py _calculate_reward().

    Parameters match env.py observation vector:
      throughput: 0-100 (percentage)
      avg_cpu:    0-1+ (normalized by function CPU request)
      avg_mem:    0-1+ (normalized by function memory request)
      replicas:   integer count of ready pods
      min_pods:   MIN_PODS from env.py (default 1)

    Returns dict with total reward and per-component breakdown.

    Note: the "action unsuccessful" penalty (-100 when scale_value != replicas)
    is omitted because NSGD doesn't use discrete actions. This gives the
    LSTM-PPO reward under the assumption that all scaling actions succeed,
    which is the fair comparison baseline.
    """
    alpha = 0.75
    beta = 0.125
    gamma = 0.125
    phi = 0.25

    r_th = alpha * (throughput ** 2)
    r_cpu = beta * (avg_cpu * 100)
    r_mem = gamma * (avg_mem * 100)
    r_rep = -phi * ((replicas - min_pods) ** 2)

    total = round(r_th + r_cpu + r_mem + r_rep, 2)

    return {
        "total": total,
        "r_throughput": round(r_th, 4),
        "r_cpu": round(r_cpu, 4),
        "r_mem": round(r_mem, 4),
        "r_replicas": round(r_rep, 4),
    }


# ---------------------------------------------------------------------------
# CSV Writer
# ---------------------------------------------------------------------------

REQUEST_LOG_FIELDS = [
    "timestamp",
    "latency_seconds",
    "rejected",
    "status_code",
    "in_flight",
    "ready_pods",
    "not_ready_pods",
    "cold",
    "idle_on",
    "busy",
    "init_free",
    "init_reserved",
    "saturated_at_arrival",
    "is_cold_start",
    "nsgd_cost",
    # perturbed / rounded (used for actual scaling decisions)
    "theta_stock",
    "theta_idle",
    "theta_exp",
    # base / un-perturbed (for convergence plots — Figure 2)
    "theta_base_stock",
    "theta_base_idle",
    "theta_base_exp",
    # algorithm progress
    "iteration",
    "phase",
    "step_in_phase",
    "phase_budget",
    "agent_cost",
    # mode
    "mode",
]

MONITOR_FIELDS = [
    "timestamp",
    "datetime",
    "cold",
    "idle_on",
    "busy",
    "init_total",
    "init_free",
    "init_reserved",
    "nsgd_cost",
    "has_rejection",
    "total_pods",
    "ready_pods",
    "not_ready_pods",
    "inflight_requests",
    "rejected_requests_interval",
    "rejected_requests",
    "latest_cost",
    "cost_delta",
    "iteration",
    "phase",
    "sample_type",
    "event_source",
]


class CSVLogger:
    """Append-only CSV writer. Creates file with headers on first write."""

    def __init__(self, filepath: str, fieldnames: list[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_file(self):
        if self._initialized:
            return
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if os.path.exists(self.filepath):
            with open(self.filepath, newline="") as f:
                first_line = f.readline().strip()
            current_header = ",".join(self.fieldnames)
            if first_line != current_header:
                backup = f"{self.filepath}.schema_mismatch.{int(time.time())}.bak"
                os.rename(self.filepath, backup)
                logger.warning(
                    "CSV schema changed for %s; moved old file to %s",
                    self.filepath,
                    backup,
                )

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        self._initialized = True

    def write_row(self, row: dict):
        with self._lock:
            self._ensure_file()
            with open(self.filepath, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.fieldnames, extrasaction="ignore"
                )
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class ExperimentMetricsCollector:

    def __init__(
        self,
        function_name: str,
        namespace: str,
        pod_watcher,
        in_flight_counter,
        max_replicas: int,
        prometheus_url: str = "http://127.0.0.1:9090",
        sampling_window: int = 30,
        func_cpu_millicores: int = 150,
        func_mem_gbi: float = 0.25,
        nsgd_weights: dict = None,
        log_dir: str = "logs/nsgd",
        min_pods: int = 1,
        monitor_interval: float = 0.1,
        agent_summary_provider=None,
        agent_url: str = None,
        agent_enabled: bool = False,
        agent_timeout_seconds: float = 2.0,
        agent_event_enabled_provider=None,
    ):
        self.function_name = function_name
        self.namespace = namespace
        self.pod_watcher = pod_watcher
        self.in_flight_counter = in_flight_counter
        self.max_replicas = max_replicas
        self.prometheus_url = prometheus_url
        self.sampling_window = sampling_window
        self.func_cpu = func_cpu_millicores
        self.func_mem = func_mem_gbi
        self.log_dir = log_dir
        self.min_pods = min_pods
        self.monitor_interval = monitor_interval
        self.agent_summary_provider = agent_summary_provider
        self.agent_url = agent_url.rstrip("/") if agent_url else None
        self.agent_enabled = agent_enabled
        self.agent_timeout_seconds = agent_timeout_seconds
        self.agent_event_enabled_provider = agent_event_enabled_provider

        self.weights = nsgd_weights or {
            "w_idle_on": 2, "w_busy": 1, "w_init": 5,
            "w_reserved": 100, "w_rej": 200,
        }

        self._prom = None
        self._k8s_metrics_api = None
        self._k8s_apps_api = None
        self._init_clients()

        self._latencies: list[float] = []
        self._latency_lock = threading.Lock()

        self._window_requests = 0
        self._window_rejections = 0
        self._monitor_rejections = 0
        self._total_rejections = 0
        self._window_lock = threading.Lock()

        self._samples: list[dict] = []
        self._samples_lock = threading.Lock()
        self._last_agent_state: tuple[int, int, int, int, int] | None = None
        self._latest_agent_response: dict | None = None
        self._latest_agent_error: str | None = None
        self._agent_client = None

        os.makedirs(log_dir, exist_ok=True)
        sample_fields = [f.name for f in fields(PeriodicSample)]
        self._sample_csv = CSVLogger(
            os.path.join(log_dir, "samples.csv"),
            sample_fields,
        )
        self._request_csv = CSVLogger(
            os.path.join(log_dir, "request_log.csv"),
            REQUEST_LOG_FIELDS,
        )
        self._monitor_csv = CSVLogger(
            os.path.join(log_dir, "monitor.csv"),
            MONITOR_FIELDS,
        )
        logger.info("Experiment logs → %s", log_dir)

        self._thread: threading.Thread = None
        self._monitor_thread: threading.Thread = None
        self._stop_event = threading.Event()

    def _init_clients(self):
        try:
            from prometheus_api_client import PrometheusConnect
            self._prom = PrometheusConnect(
                url=self.prometheus_url, disable_ssl=True
            )
            logger.info("Prometheus connected: %s", self.prometheus_url)
        except ImportError:
            logger.warning("prometheus_api_client not installed")
        except Exception as e:
            logger.warning("Prometheus failed: %s", e)

        try:
            from kubernetes import client
            self._k8s_metrics_api = client.CustomObjectsApi()
            self._k8s_apps_api = client.AppsV1Api()
        except Exception as e:
            logger.warning("K8s metrics API unavailable: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(
            target=self._collection_loop,
            name="experiment-metrics",
            daemon=True,
        )
        self._thread.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="experiment-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(
            "Collector started (window=%ds monitor=%.3fs)",
            self.sampling_window,
            self.monitor_interval,
        )

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        if self._agent_client:
            self._agent_client.close()

    def is_alive(self) -> bool:
        sample_alive = self._thread is not None and self._thread.is_alive()
        monitor_alive = (
            self._monitor_thread is not None
            and self._monitor_thread.is_alive()
        )
        return sample_alive and monitor_alive

    def record_request_latency(self, latency_seconds: float, rejected: bool):
        with self._latency_lock:
            self._latencies.append(latency_seconds)
        with self._window_lock:
            self._window_requests += 1
            if rejected:
                self._window_rejections += 1
                self._monitor_rejections += 1
                self._total_rejections += 1

    def record_request(self, row: dict):
        self._request_csv.write_row(row)

    def record_monitor_event(self, event_source: str):
        """
        Write an immediate monitor snapshot when a state-changing event occurs.

        These rows improve time integration for short-lived states such as
        init_reserved. They do not consume the interval rejection counter; the
        periodic monitor row remains the single source of rejection counts.
        """
        if self._stop_event.is_set():
            return
        try:
            row = self._collect_monitor_row(
                consume_rejections=False,
                sample_type="event",
                event_source=event_source,
            )
            self._monitor_csv.write_row(row)
        except Exception as e:
            logger.debug("Monitor event collection failed: %s", e)

    def get_samples(self, limit: int = 100, since: float = 0.0) -> list[dict]:
        with self._samples_lock:
            if since > 0:
                filtered = [
                    s for s in self._samples if s["timestamp"] > since
                ]
            else:
                filtered = list(self._samples)
            return filtered[-limit:]

    def get_latest(self) -> dict | None:
        with self._samples_lock:
            return self._samples[-1] if self._samples else None

    # ------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------

    def _collection_loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.sampling_window)
            if self._stop_event.is_set():
                return
            try:
                sample = self._collect_sample()
                sample_dict = asdict(sample)

                with self._samples_lock:
                    self._samples.append(sample_dict)
                    if len(self._samples) > 10000:
                        del self._samples[:1000]

                self._sample_csv.write_row(sample_dict)

                logger.info(
                    "Sample: exec=%.3fs tput=%d%% reqs=%d replicas=%d "
                    "cpu=%.2f mem=%.2f nsgd_cost=%.1f ppo_reward=%.1f",
                    sample.avg_execution_time, sample.throughput,
                    sample.requests, sample.replicas,
                    sample.avg_cpu, sample.avg_mem,
                    sample.nsgd_cost, sample.lstm_ppo_reward,
                )
            except Exception as e:
                logger.error("Collection error: %s", e)

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.monitor_interval)
            if self._stop_event.is_set():
                return
            try:
                row = self._collect_monitor_row(
                    consume_rejections=True,
                    sample_type="periodic",
                    event_source="monitor_interval",
                )
                self._send_monitor_events(row)
                self._monitor_csv.write_row(row)
            except Exception as e:
                logger.error("Monitor collection error: %s", e)

    def _agent_events_enabled(self) -> bool:
        if not self.agent_enabled or not self.agent_url:
            return False
        if self.agent_event_enabled_provider:
            try:
                return bool(self.agent_event_enabled_provider())
            except Exception as e:
                logger.debug("Agent event enabled check failed: %s", e)
                return False
        return True

    def _get_agent_client(self):
        if self._agent_client is None:
            import httpx
            self._agent_client = httpx.Client(
                timeout=httpx.Timeout(
                    self.agent_timeout_seconds,
                    connect=min(self.agent_timeout_seconds, 2.0),
                )
            )
        return self._agent_client

    @staticmethod
    def _state_tuple_from_row(row: dict) -> tuple[int, int, int, int, int]:
        return (
            int(row["cold"]),
            int(row["idle_on"]),
            int(row["busy"]),
            int(row["init_total"]),
            int(row["init_reserved"]),
        )

    def _post_agent_event(
        self,
        state: tuple[int, int, int, int, int],
        has_rejected_job: bool,
        timestamp: float,
    ) -> dict | None:
        payload = {
            "state": list(state),
            "has_rejected_job": has_rejected_job,
            "timestamp": timestamp,
        }
        try:
            response = self._get_agent_client().post(
                f"{self.agent_url}/event",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            self._latest_agent_response = result
            self._latest_agent_error = None
            return result
        except Exception as exc:
            self._latest_agent_error = str(exc)
            logger.warning("Monitor agent event failed: %s", exc)
            return None

    def _send_monitor_events(self, row: dict):
        if not self._agent_events_enabled():
            return

        state = self._state_tuple_from_row(row)
        timestamp = float(row["timestamp"])
        rejection_count = int(row.get("rejected_requests_interval") or 0)

        if state != self._last_agent_state:
            self._post_agent_event(
                state,
                has_rejected_job=False,
                timestamp=timestamp,
            )
            self._last_agent_state = state

        for _ in range(rejection_count):
            self._post_agent_event(
                state,
                has_rejected_job=True,
                timestamp=timestamp,
            )

    def _collect_monitor_row(
        self,
        consume_rejections: bool = True,
        sample_type: str = "periodic",
        event_source: str = "monitor_interval",
    ) -> dict:
        counts = self.pod_watcher.get_counts()
        in_flight = self.in_flight_counter.count
        ready = counts["ready"]
        not_ready = counts["not_ready"]
        total = counts["total"]

        busy = min(in_flight, ready)
        idle_on = ready - busy
        blocked = max(0, in_flight - ready)
        init_reserved = min(blocked, not_ready)
        init_total = not_ready
        init_free = not_ready - init_reserved
        cold = self.max_replicas - total

        with self._window_lock:
            if consume_rejections:
                rejected_requests_interval = self._monitor_rejections
                self._monitor_rejections = 0
            else:
                rejected_requests_interval = 0
            rejected_requests = self._total_rejections
        has_rejection = rejected_requests_interval > 0

        w = self.weights
        nsgd_cost = (
            w["w_idle_on"] * idle_on
            + w["w_busy"] * busy
            + w["w_init"] * init_total
            + w["w_reserved"] * init_reserved
            + w["w_rej"] * rejected_requests_interval
        )

        agent = {}
        if self.agent_summary_provider:
            agent = self.agent_summary_provider() or {}

        return {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cold": cold,
            "idle_on": idle_on,
            "busy": busy,
            "init_total": init_total,
            "init_free": init_free,
            "init_reserved": init_reserved,
            "nsgd_cost": nsgd_cost,
            "has_rejection": has_rejection,
            "total_pods": total,
            "ready_pods": ready,
            "not_ready_pods": not_ready,
            "inflight_requests": in_flight,
            "rejected_requests_interval": rejected_requests_interval,
            "rejected_requests": rejected_requests,
            "latest_cost": agent.get("latest_cost"),
            "cost_delta": agent.get("cost_delta"),
            "iteration": agent.get("iteration"),
            "phase": agent.get("phase"),
            "sample_type": sample_type,
            "event_source": event_source,
        }

    def _collect_sample(self) -> PeriodicSample:
        sample = PeriodicSample(timestamp=time.time())

        # Prometheus metrics (same as env.py)
        sample.avg_execution_time = self._query_avg_execution()
        sample.requests = self._query_total_requests()
        sample.throughput = self._query_throughput(sample.requests)
        sample.replicas = self._query_replicas()

        cpu, mem = self._query_pod_resources()
        sample.avg_cpu = cpu
        sample.avg_mem = mem

        # NSGD state
        counts = self.pod_watcher.get_counts()
        in_flight = self.in_flight_counter.count
        ready = counts["ready"]
        not_ready = counts["not_ready"]
        total = counts["total"]

        sample.busy = min(in_flight, ready)
        sample.idle_on = ready - sample.busy
        blocked = max(0, in_flight - ready)
        sample.init_reserved = min(blocked, not_ready)
        sample.init_total = not_ready
        sample.cold = self.max_replicas - total

        # Proxy outcomes collected during this sample window.
        with self._latency_lock:
            latencies = self._latencies.copy()
            self._latencies.clear()

        with self._window_lock:
            sample.proxy_requests = self._window_requests
            sample.proxy_rejections = self._window_rejections
            self._window_requests = 0
            self._window_rejections = 0

        sample.has_rejection = sample.proxy_rejections > 0

        # NSGD cost (equation 7)
        w = self.weights
        sample.nsgd_cost = (
            w["w_idle_on"] * sample.idle_on
            + w["w_busy"] * sample.busy
            + w["w_init"] * sample.init_total
            + w["w_reserved"] * sample.init_reserved
            + w["w_rej"] * sample.proxy_rejections
        )

        # LSTM-PPO reward (env.py _calculate_reward)
        reward = compute_lstm_ppo_reward(
            throughput=sample.throughput,
            avg_cpu=sample.avg_cpu,
            avg_mem=sample.avg_mem,
            replicas=sample.replicas,
            min_pods=self.min_pods,
        )
        sample.lstm_ppo_reward = reward["total"]
        sample.reward_throughput = reward["r_throughput"]
        sample.reward_cpu = reward["r_cpu"]
        sample.reward_mem = reward["r_mem"]
        sample.reward_replicas = reward["r_replicas"]

        if latencies:
            latencies_ms = [l * 1000 for l in latencies]
            sample.avg_response_latency_ms = round(
                sum(latencies_ms) / len(latencies_ms), 2
            )
            latencies_ms.sort()
            p99_idx = max(0, int(len(latencies_ms) * 0.99) - 1)
            sample.p99_response_latency_ms = round(
                latencies_ms[p99_idx], 2
            )

        return sample

    # ------------------------------------------------------------------
    # Prometheus queries (identical to env.py)
    # ------------------------------------------------------------------

    def _query_avg_execution(self) -> float:
        if not self._prom:
            return 0.0
        query = (
            f"(rate(gateway_functions_seconds_sum"
            f"{{function_name='{self.function_name}.{self.namespace}', "
            f"code='200'}}[{self.sampling_window}s]) / "
            f"rate(gateway_functions_seconds_count"
            f"{{function_name='{self.function_name}.{self.namespace}', "
            f"code='200'}}[{self.sampling_window}s]))"
        )
        try:
            data = self._prom.custom_query(query=query)
            if data and "value" in data[0] and len(data[0]["value"]) > 1:
                val = float(data[0]["value"][1])
                return 0.0 if math.isnan(val) else round(val, 3)
        except Exception as e:
            logger.debug("avg_execution query failed: %s", e)
        return 0.0

    def _query_total_requests(self) -> int:
        if not self._prom:
            return 0
        query = (
            f"increase(gateway_function_invocation_total"
            f"{{function_name='{self.function_name}.{self.namespace}'}}"
            f"[{self.sampling_window}s])"
        )
        try:
            data = self._prom.custom_query(query=query)
            total = 0
            if data:
                for d in data:
                    if "value" in d and len(d["value"]) > 1:
                        total += int(float(d["value"][1]))
            return total
        except Exception as e:
            logger.debug("total_requests query failed: %s", e)
        return 0

    def _query_throughput(self, total_requests: int) -> float:
        if not self._prom or total_requests == 0:
            return 100.0
        query = (
            f"increase(gateway_function_invocation_total"
            f"{{code='200', function_name='{self.function_name}.{self.namespace}'}}"
            f"[{self.sampling_window}s])"
        )
        try:
            data = self._prom.custom_query(query=query)
            successful = 0
            if data and "value" in data[0] and len(data[0]["value"]) > 1:
                successful = int(float(data[0]["value"][1]))
            return round((successful / total_requests) * 100, 2)
        except Exception as e:
            logger.debug("throughput query failed: %s", e)
        return 100.0 if total_requests == 0 else 0.0

    def _query_replicas(self) -> int:
        if not self._k8s_apps_api:
            return self.pod_watcher.get_counts()["ready"]
        try:
            deploy = self._k8s_apps_api.read_namespaced_deployment(
                name=self.function_name, namespace=self.namespace
            )
            return deploy.status.ready_replicas or 0
        except Exception as e:
            logger.debug("replicas query failed: %s", e)
            return self.pod_watcher.get_counts()["ready"]

    def _query_pod_resources(self) -> tuple[float, float]:
        if not self._k8s_metrics_api:
            return 0.0, 0.0
        try:
            resource_list = self._k8s_metrics_api.list_namespaced_custom_object(
                "metrics.k8s.io", "v1beta1", self.namespace, "pods"
            )
            pods = [
                pod["containers"][0]["usage"]
                for pod in resource_list.get("items", [])
                if pod.get("metadata", {})
                    .get("labels", {})
                    .get("faas_function") == self.function_name
            ]
            if not pods:
                return 0.0, 0.0

            cpu_total = 0.0
            mem_total = 0.0
            for usage in pods:
                cpu_total += self._parse_cpu(usage.get("cpu", "0"))
                mem_total += self._parse_mem(usage.get("memory", "0"))

            avg_cpu = round((cpu_total / len(pods)) / self.func_cpu, 4)
            avg_mem = round((mem_total / len(pods)) / self.func_mem, 4)
            return avg_cpu, avg_mem
        except Exception as e:
            logger.debug("pod resources query failed: %s", e)
            return 0.0, 0.0

    @staticmethod
    def _parse_cpu(value: str) -> float:
        try:
            if value.endswith("n"):
                return round(int(value[:-1]) / 1e6, 4)
            elif value.endswith("u"):
                return round(int(value[:-1]) / 1e3, 4)
            elif value.endswith("m"):
                return round(int(value[:-1]), 4)
            else:
                return round(float(value) * 1000, 4)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _parse_mem(value: str) -> float:
        try:
            if value.endswith("Ki"):
                return round(int(value[:-2]) / (1024 * 1024), 4)
            elif value.endswith("Mi"):
                return round(int(value[:-2]) / 1024, 4)
            elif value.endswith("Gi"):
                return round(int(value[:-2]), 4)
            else:
                return round(int(value) / (1024 * 1024 * 1024), 4)
        except (ValueError, TypeError):
            return 0.0

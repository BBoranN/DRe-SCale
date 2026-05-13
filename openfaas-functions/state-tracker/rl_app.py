"""
RL Agent Middleware (Passive Observer)
======================================

This is the LSTM-PPO equivalent of the NSGD middleware. It is purely
PASSIVE — it does NOT make scaling decisions. The RL agent (running
separately, e.g. via env.py) does all scaling via the K8s API directly.

What this middleware does:
  - Proxies function calls from the load generator to OpenFaaS
  - Observes the system state (pod counts, in-flight requests)
  - Computes the SAME NSGD cost (equation 7) per request and per sample
  - Logs everything to logs/rl/ in a schema identical to the NSGD logs
    so the same Jupyter notebook can read both datasets for comparison

What this middleware does NOT do:
  - Call any agent
  - Make scaling decisions (no compute_scale_action)
  - Run an expiration controller
  - Apply theta values (no theta to apply)

The CSV schema matches the NSGD middleware exactly. Theta columns are
left as None for RL runs. The 'mode' column is "rl" so the comparison
notebook can filter cleanly.
"""

import os
import time
import logging
import threading
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from pod_watcher import PodWatcher
from experiment_metrics import ExperimentMetricsCollector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENFAAS_GATEWAY = os.getenv("OPENFAAS_GATEWAY", "http://127.0.0.1:8080")

FUNCTION_NAME = os.getenv("FUNCTION_NAME", "matmul")
FUNCTION_NAMESPACE = os.getenv("FUNCTION_NAMESPACE", "openfaas-fn")
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "24"))
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
TRACKER_PORT = int(os.getenv("TRACKER_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "100000"))

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090")
SAMPLING_WINDOW = int(os.getenv("SAMPLING_WINDOW", "30"))
FUNC_CPU_MILLICORES = int(os.getenv("FUNC_CPU_MILLICORES", "150"))
FUNC_MEM_GBI = float(os.getenv("FUNC_MEM_GBI", "0.25"))
EXPERIMENT_LOG_DIR = os.getenv("EXPERIMENT_LOG_DIR", "logs/rl_eval")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rl-tracker")

# Same weights as NSGD config so the cost numbers are directly comparable.
NSGD_WEIGHTS = {
    "w_idle_on": 2, "w_busy": 1, "w_init": 5,
    "w_reserved": 100, "w_rej": 200,
}


# ---------------------------------------------------------------------------
# In-Flight Counter
# ---------------------------------------------------------------------------

class InFlightCounter:
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def observe_and_increment(self) -> int:
        with self._lock:
            pre = self._count
            self._count += 1
            return pre

    def decrement(self) -> int:
        with self._lock:
            self._count = max(0, self._count - 1)
            return self._count


# ---------------------------------------------------------------------------
# State Computation
# ---------------------------------------------------------------------------

@dataclass
class FunctionState:
    timestamp: float
    in_flight: int
    ready_pods: int
    not_ready_pods: int
    total_pods: int
    max_replicas: int
    cold: int = 0
    init_reserved: int = 0
    init_free: int = 0
    busy: int = 0
    idle_on: int = 0
    rejected: bool = False
    saturated_at_arrival: bool = False
    response_status: int = 0

    def __post_init__(self):
        self.cold = self.max_replicas - self.total_pods
        self.busy = min(self.in_flight, self.ready_pods)
        self.idle_on = self.ready_pods - self.busy
        blocked_requests = max(0, self.in_flight - self.ready_pods)
        self.init_reserved = min(blocked_requests, self.not_ready_pods)
        self.init_free = self.not_ready_pods - self.init_reserved
        self.saturated_at_arrival = (
            self.busy + self.init_reserved >= self.max_replicas
        )


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

in_flight_counter = InFlightCounter()
pod_watcher: PodWatcher = None
http_client: httpx.AsyncClient = None
metrics_collector: ExperimentMetricsCollector = None

state_history: list[dict] = []
state_history_lock = threading.Lock()

total_rejected_requests = 0
rejected_lock = threading.Lock()

REJECTION_STATUS_CODES = {429, 500, 503}


# ---------------------------------------------------------------------------
# App Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pod_watcher, http_client, metrics_collector

    pod_watcher = PodWatcher(
        function_name=FUNCTION_NAME,
        namespace=FUNCTION_NAMESPACE,
    )
    pod_watcher.start()
    logger.info("Pod watcher started for %s in %s",
                FUNCTION_NAME, FUNCTION_NAMESPACE)

    metrics_collector = ExperimentMetricsCollector(
        function_name=FUNCTION_NAME,
        namespace=FUNCTION_NAMESPACE,
        pod_watcher=pod_watcher,
        in_flight_counter=in_flight_counter,
        max_replicas=MAX_REPLICAS,
        prometheus_url=PROMETHEUS_URL,
        sampling_window=SAMPLING_WINDOW,
        func_cpu_millicores=FUNC_CPU_MILLICORES,
        func_mem_gbi=FUNC_MEM_GBI,
        nsgd_weights=NSGD_WEIGHTS,
        log_dir=EXPERIMENT_LOG_DIR,
        min_pods=MIN_REPLICAS,
    )
    metrics_collector.start()

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    logger.info("Proxying to: %s", OPENFAAS_GATEWAY)
    logger.info("RL middleware is PASSIVE — no scaling decisions made here.")

    yield

    metrics_collector.stop()
    pod_watcher.stop()
    await http_client.aclose()


app = FastAPI(title="RL Agent Middleware", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Proxy Endpoint
# ---------------------------------------------------------------------------

@app.api_route(
    "/function/{function_name:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_function_call(function_name: str, request: Request):
    """
    Pure passive proxy. Forwards the request to OpenFaaS, observes the
    state at arrival time, computes NSGD cost, and logs everything.

    The RL agent makes scaling decisions independently via the K8s API.
    We just record what the system looks like when each request arrives.
    """
    # 1. Atomic pre-admission snapshot
    pre_admission_count = in_flight_counter.observe_and_increment()

    pod_counts = pod_watcher.get_counts()
    state = FunctionState(
        timestamp=time.time(),
        in_flight=pre_admission_count,
        ready_pods=pod_counts["ready"],
        not_ready_pods=pod_counts["not_ready"],
        total_pods=pod_counts["total"],
        max_replicas=MAX_REPLICAS,
    )

    entry = _record_state(state)

    # 2. Forward to gateway and measure latency
    request_start = time.monotonic()
    is_rejected = False
    response_status = 0

    try:
        body = await request.body()
        target_url = f"{OPENFAAS_GATEWAY}/function/{function_name}"
        forward_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in (
                "host", "content-length", "transfer-encoding"
            )
        }

        response = await http_client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=forward_headers,
        )

        is_rejected = response.status_code in REJECTION_STATUS_CODES
        response_status = response.status_code
        _mark_outcome(
            entry, status=response.status_code, rejected=is_rejected
        )

        response_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in (
                "transfer-encoding", "content-encoding", "content-length"
            )
        }
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    except httpx.TimeoutException:
        logger.warning("Timeout proxying to %s", OPENFAAS_GATEWAY)
        response_status = 504
        _mark_outcome(entry, status=504, rejected=False)
        return Response(content="Gateway timeout", status_code=504)

    except httpx.ConnectError:
        logger.error("Cannot connect to gateway at %s", OPENFAAS_GATEWAY)
        response_status = 502
        _mark_outcome(entry, status=502, rejected=False)
        return Response(
            content=f"Cannot reach gateway at {OPENFAAS_GATEWAY}",
            status_code=502,
        )

    except Exception as e:
        logger.error("Proxy error: %s", e)
        response_status = 502
        _mark_outcome(entry, status=502, rejected=False)
        return Response(content=str(e), status_code=502)

    finally:
        in_flight_counter.decrement()

        latency = time.monotonic() - request_start
        if metrics_collector:
            metrics_collector.record_request_latency(latency, is_rejected)

            # Compute per-request NSGD cost (equation 7)
            w = NSGD_WEIGHTS
            nsgd_cost = (
                w["w_idle_on"] * state.idle_on
                + w["w_busy"] * state.busy
                + w["w_init"] * (state.init_free + state.init_reserved)
                + w["w_reserved"] * state.init_reserved
                + (w["w_rej"] if state.saturated_at_arrival else 0)
            )

            # Same schema as NSGD logs. Theta and algorithm fields are
            # None for RL runs. mode = "rl" so notebook can filter.
            metrics_collector.record_request({
                "timestamp": state.timestamp,
                "latency_seconds": round(latency, 4),
                "rejected": is_rejected,
                "status_code": response_status,
                "in_flight": pre_admission_count,
                "ready_pods": state.ready_pods,
                "not_ready_pods": state.not_ready_pods,
                "cold": state.cold,
                "idle_on": state.idle_on,
                "busy": state.busy,
                "init_free": state.init_free,
                "init_reserved": state.init_reserved,
                "saturated_at_arrival": state.saturated_at_arrival,
                "is_cold_start": state.idle_on == 0 and state.cold > 0,
                "nsgd_cost": nsgd_cost,
                "theta_stock": None,
                "theta_idle": None,
                "theta_exp": None,
                "theta_base_stock": None,
                "theta_base_idle": None,
                "theta_base_exp": None,
                "iteration": None,
                "phase": None,
                "step_in_phase": None,
                "phase_budget": None,
                "agent_cost": None,
                "mode": "rl",
            })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _record_state(state: FunctionState) -> dict:
    entry = asdict(state)
    with state_history_lock:
        state_history.append(entry)
        if len(state_history) > MAX_HISTORY:
            del state_history[: MAX_HISTORY // 10]
    return entry


def _mark_outcome(entry: dict, status: int, rejected: bool):
    global total_rejected_requests
    with state_history_lock:
        entry["response_status"] = status
        entry["rejected"] = rejected
    if rejected:
        with rejected_lock:
            total_rejected_requests += 1


# ---------------------------------------------------------------------------
# State Observation Endpoints
# ---------------------------------------------------------------------------

@app.get("/states/current")
async def get_current_state():
    pod_counts = pod_watcher.get_counts()
    state = FunctionState(
        timestamp=time.time(),
        in_flight=in_flight_counter.count,
        ready_pods=pod_counts["ready"],
        not_ready_pods=pod_counts["not_ready"],
        total_pods=pod_counts["total"],
        max_replicas=MAX_REPLICAS,
    )
    return asdict(state)


@app.get("/states/history")
async def get_state_history(limit: int = 1000, since: float = 0.0):
    with state_history_lock:
        if since > 0:
            filtered = [
                s for s in state_history if s["timestamp"] > since
            ]
        else:
            filtered = list(state_history)
        return filtered[-limit:]


@app.get("/states/summary")
async def get_state_summary():
    pod_counts = pod_watcher.get_counts()
    state = FunctionState(
        timestamp=time.time(),
        in_flight=in_flight_counter.count,
        ready_pods=pod_counts["ready"],
        not_ready_pods=pod_counts["not_ready"],
        total_pods=pod_counts["total"],
        max_replicas=MAX_REPLICAS,
    )
    return {
        "function": FUNCTION_NAME,
        "max_replicas": MAX_REPLICAS,
        "min_replicas": MIN_REPLICAS,
        "mode": "rl",
        "scaling_authority": "external (RL agent via K8s API)",
        "observables": {
            "in_flight_requests": in_flight_counter.count,
            "total_rejected_requests": total_rejected_requests,
            "ready_pods": pod_counts["ready"],
            "not_ready_pods": pod_counts["not_ready"],
            "total_pods": pod_counts["total"],
            "pod_details": pod_watcher.get_pod_details(),
        },
        "derived_states": {
            "cold": state.cold,
            "init_reserved": state.init_reserved,
            "init_free": state.init_free,
            "busy": state.busy,
            "idle_on": state.idle_on,
        },
        "latest_sample": (
            metrics_collector.get_latest()
            if metrics_collector else None
        ),
        "history_size": len(state_history),
        "watcher_alive": pod_watcher.is_alive(),
        "log_dir": EXPERIMENT_LOG_DIR,
    }


@app.get("/states/export")
async def export_history():
    with state_history_lock:
        return JSONResponse(
            content=state_history,
            headers={
                "Content-Disposition": "attachment; filename=state_history.json"
            },
        )


@app.delete("/states/history")
async def clear_history():
    with state_history_lock:
        count = len(state_history)
        state_history.clear()
    return {"cleared": count}


# ---------------------------------------------------------------------------
# Experiment Metrics Endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics/latest")
async def metrics_latest():
    if not metrics_collector:
        return {"error": "Metrics collector not initialized"}
    sample = metrics_collector.get_latest()
    if not sample:
        return {"error": "No samples yet", "sampling_window": SAMPLING_WINDOW}
    return sample


@app.get("/metrics/history")
async def metrics_history(limit: int = 100, since: float = 0.0):
    if not metrics_collector:
        return {"error": "Metrics collector not initialized"}
    return metrics_collector.get_samples(limit=limit, since=since)


@app.get("/metrics/summary")
async def metrics_summary():
    if not metrics_collector:
        return {"error": "Metrics collector not initialized"}

    samples = metrics_collector.get_samples(limit=10000)
    if not samples:
        return {"error": "No samples yet"}

    costs = [s["nsgd_cost"] for s in samples]
    rewards = [s["lstm_ppo_reward"] for s in samples]
    latencies = [
        s["avg_response_latency_ms"] for s in samples
        if s["avg_response_latency_ms"] > 0
    ]
    throughputs = [s["throughput"] for s in samples]
    replicas_list = [s["replicas"] for s in samples]
    cpus = [s["avg_cpu"] for s in samples if s["avg_cpu"] > 0]
    mems = [s["avg_mem"] for s in samples if s["avg_mem"] > 0]
    total_proxy_reqs = sum(s["proxy_requests"] for s in samples)
    total_proxy_rej = sum(s["proxy_rejections"] for s in samples)

    def _stats(values):
        if not values:
            return {"mean": 0, "min": 0, "max": 0, "count": 0}
        return {
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "count": len(values),
        }

    return {
        "agent": "rl",
        "sampling_window_seconds": SAMPLING_WINDOW,
        "total_samples": len(samples),
        "duration_seconds": (
            round(samples[-1]["timestamp"] - samples[0]["timestamp"], 1)
            if len(samples) > 1 else 0
        ),
        "nsgd_cost": _stats(costs),
        "lstm_ppo_reward": _stats(rewards),
        "response_latency_ms": _stats(latencies),
        "throughput_pct": _stats(throughputs),
        "replicas": _stats(replicas_list),
        "avg_cpu_normalized": _stats(cpus),
        "avg_mem_normalized": _stats(mems),
        "csv_files": {
            "samples": os.path.join(EXPERIMENT_LOG_DIR, "samples.csv"),
            "requests": os.path.join(EXPERIMENT_LOG_DIR, "request_log.csv"),
        },
        "proxy_totals": {
            "requests": total_proxy_reqs,
            "rejections": total_proxy_rej,
            "rejection_rate": (
                round(total_proxy_rej / total_proxy_reqs, 4)
                if total_proxy_reqs > 0 else 0
            ),
        },
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent": "rl (external)",
        "watcher_alive": pod_watcher.is_alive(),
        "metrics_collector_alive": (
            metrics_collector.is_alive() if metrics_collector else False
        ),
        "gateway": OPENFAAS_GATEWAY,
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting RL agent middleware on port {TRACKER_PORT}")
    print(f"  Mode:         PASSIVE (RL agent scales externally)")
    print(f"  Proxying to:  {OPENFAAS_GATEWAY}")
    print(f"  Function:     {FUNCTION_NAME} in {FUNCTION_NAMESPACE}")
    print(f"  Replicas:     {MIN_REPLICAS}–{MAX_REPLICAS}")
    print(f"  Prometheus:   {PROMETHEUS_URL}")
    print(f"  Sampling:     every {SAMPLING_WINDOW}s")
    print(f"  Logs:         {EXPERIMENT_LOG_DIR}/")
    print()
    print("CSV files (written automatically, same schema as NSGD logs):")
    print(f"  {EXPERIMENT_LOG_DIR}/samples.csv       (every {SAMPLING_WINDOW}s)")
    print(f"  {EXPERIMENT_LOG_DIR}/request_log.csv   (every request)")
    print()
    print("Run the LSTM-PPO RL agent (env.py) separately. Point its")
    print("workload generator at this proxy:")
    print(f"  http://127.0.0.1:{TRACKER_PORT}/function/{FUNCTION_NAME}")
    print()
    print("Comparison in Jupyter:")
    print("  nsgd  = pd.read_csv('logs/nsgd/samples.csv')")
    print(f"  rl    = pd.read_csv('{EXPERIMENT_LOG_DIR}/samples.csv')")
    print()

    uvicorn.run(
        "rl_app:app",
        host="0.0.0.0",
        port=TRACKER_PORT,
        log_level=LOG_LEVEL.lower(),
        workers=1,
    )
import os
import time
import json
import logging
import threading
import asyncio
import argparse
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from pod_watcher import PodWatcher
from exp_controller import ExpirationController
from agent_call_service import AgentCallService
from experiment_metrics import ExperimentMetricsCollector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENFAAS_GATEWAY = os.getenv("OPENFAAS_GATEWAY", "http://127.0.0.1:8080")
AGENT_URL = os.getenv("AGENT_URL", "http://127.0.0.1:5000")
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "true").lower() in ("1", "true", "yes", "on")
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "2"))

FUNCTION_NAME = os.getenv("FUNCTION_NAME", "matmul")
FUNCTION_NAMESPACE = os.getenv("FUNCTION_NAMESPACE", "openfaas-fn")
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "24"))
TRACKER_PORT = int(os.getenv("TRACKER_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "100000"))

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090")
SAMPLING_WINDOW = int(os.getenv("SAMPLING_WINDOW", "30"))
MONITOR_INTERVAL_SECONDS = float(os.getenv("MONITOR_INTERVAL_SECONDS", "0.1"))
FUNC_CPU_MILLICORES = int(os.getenv("FUNC_CPU_MILLICORES", "150"))
FUNC_MEM_GBI = float(os.getenv("FUNC_MEM_GBI", "0.25"))
DEFAULT_EXPERIMENT_LOG_DIR = "logs/nsgd_const_new"


def parse_experiment_log_dir_arg() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--experiment-log-dir",
        "--experiment_log_dir",
        "--EXPERIMENT_LOG_DIR",
        dest="experiment_log_dir",
    )
    args, _unknown = parser.parse_known_args()
    return args.experiment_log_dir


EXPERIMENT_LOG_DIR = (
    parse_experiment_log_dir_arg()
    or os.getenv("EXPERIMENT_LOG_DIR")
    or DEFAULT_EXPERIMENT_LOG_DIR
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("state-tracker")

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
        self._change_callback = None

    def set_change_callback(self, callback):
        self._change_callback = callback

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def observe_and_increment(self) -> int:
        with self._lock:
            pre = self._count
            self._count += 1
            current = self._count
        self._notify_change("increment", pre, current)
        return pre

    def decrement(self) -> int:
        with self._lock:
            previous = self._count
            self._count = max(0, self._count - 1)
            current = self._count
        self._notify_change("decrement", previous, current)
        return current

    def _notify_change(self, event_type: str, previous: int, current: int):
        if previous == current or not self._change_callback:
            return
        try:
            self._change_callback(
                event_type=event_type,
                old_count=previous,
                new_count=current,
            )
        except Exception as e:
            logger.debug("In-flight change callback failed: %s", e)


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
# Evaluation Mode
# ---------------------------------------------------------------------------

class EvaluationMode:
    """
    Toggle between training and evaluation.

    Training:  agent is called on every request, theta is perturbed
    Evaluation: fixed theta from last training, no agent calls
    """

    def __init__(self):
        self._active = False
        self._theta_step: list | None = None
        self._theta_base: list | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def theta_step(self) -> list | None:
        with self._lock:
            return self._theta_step

    @property
    def theta_base(self) -> list | None:
        with self._lock:
            return self._theta_base

    def enable(self, theta_step: list, theta_base: list):
        with self._lock:
            self._active = True
            self._theta_step = theta_step
            self._theta_base = theta_base
        logger.info(
            "EVALUATION MODE enabled: theta_step=%s theta_base=%s",
            theta_step, theta_base,
        )

    def disable(self):
        with self._lock:
            self._active = False
            self._theta_step = None
            self._theta_base = None
        logger.info("TRAINING MODE resumed")


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

in_flight_counter = InFlightCounter()
pod_watcher: PodWatcher = None
http_client: httpx.AsyncClient = None
agent_call_service: AgentCallService = None
expiration_controller: ExpirationController = None
metrics_collector: ExperimentMetricsCollector = None
eval_mode = EvaluationMode()

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
    global pod_watcher, http_client, agent_call_service
    global expiration_controller, metrics_collector

    pod_watcher = PodWatcher(
        function_name=FUNCTION_NAME,
        namespace=FUNCTION_NAMESPACE,
    )
    pod_watcher.start()
    logger.info("Pod watcher started for %s in %s", FUNCTION_NAME, FUNCTION_NAMESPACE)

    expiration_controller = ExpirationController(
        function_name=FUNCTION_NAME,
        gateway_url=OPENFAAS_GATEWAY,
        pod_watcher=pod_watcher,
        in_flight_counter=in_flight_counter,
        max_replicas=MAX_REPLICAS,
        k_exp=100,
    )
    expiration_controller.start()

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
        monitor_interval=MONITOR_INTERVAL_SECONDS,
        agent_summary_provider=_get_agent_cost_summary,
        agent_url=AGENT_URL,
        agent_enabled=AGENT_ENABLED,
        agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
        agent_event_enabled_provider=lambda: not eval_mode.active,
    )
    in_flight_counter.set_change_callback(
        lambda **kwargs: metrics_collector.record_monitor_event(
            "inflight_" + kwargs.get("event_type", "change")
        )
    )
    pod_watcher.set_change_callback(
        lambda **kwargs: metrics_collector.record_monitor_event(
            "pod_" + kwargs.get("event_type", "change").lower()
        )
    )
    metrics_collector.start()

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    logger.info("Proxying to: %s", OPENFAAS_GATEWAY)

    agent_call_service = AgentCallService(
        agent_url=AGENT_URL,
        timeout_seconds=AGENT_TIMEOUT_SECONDS,
        enabled=AGENT_ENABLED,
    )
    logger.info(
        "Agent calls %s: %s",
        "enabled" if AGENT_ENABLED else "disabled",
        AGENT_URL,
    )

    yield

    metrics_collector.stop()
    expiration_controller.stop()
    await agent_call_service.close()
    pod_watcher.stop()
    await http_client.aclose()


app = FastAPI(title="NSGD State Tracker", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Proxy Endpoint
# ---------------------------------------------------------------------------

@app.api_route(
    "/function/{function_name:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_function_call(function_name: str, request: Request):
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

    # 2. Get theta — either from agent (training) or fixed (evaluation).
    # In training, this reads the current policy without advancing NSGD.
    # The event/cost update is sent after the response outcome is known.
    agent_response = None
    agent_error = None
    theta_step = None
    theta_base = None

    if eval_mode.active:
        theta_step = eval_mode.theta_step
        theta_base = eval_mode.theta_base
        current_mode = "evaluation"
    else:
        agent_response, agent_error = await agent_call_service.get_theta()
        current_mode = "training"

        with state_history_lock:
            entry["theta_response"] = agent_response
            entry["agent_error"] = agent_error

        if agent_response is not None:
            theta_step = agent_response.get("theta_step")
            theta_base = agent_response.get("theta")
        else:
            logger.warning("Agent unavailable: %s", agent_error)

    # 3. Apply scaling decision
    if theta_step:
        handle_agent_response(state, theta_step)

    # 4. Forward to gateway and measure latency
    request_start = time.monotonic()
    is_rejected = False
    response_status = 0

    try:
        body = await request.body()
        target_url = f"{OPENFAAS_GATEWAY}/function/{function_name}"
        forward_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        response = await http_client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=forward_headers,
        )

        is_rejected = response.status_code in REJECTION_STATUS_CODES
        response_status = response.status_code
        _mark_outcome(entry, status=response.status_code, rejected=is_rejected)

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
                + (w["w_rej"] if is_rejected else 0)
            )

            # Safe extraction from theta lists
            def _safe_get(lst, idx):
                if lst and len(lst) > idx:
                    return lst[idx]
                return None

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
                # perturbed / rounded (used for scaling)
                "theta_stock": _safe_get(theta_step, 0),
                "theta_idle": _safe_get(theta_step, 1),
                "theta_exp": _safe_get(theta_step, 2),
                # base / un-perturbed (for Figure 2 convergence plots)
                "theta_base_stock": _safe_get(theta_base, 0),
                "theta_base_idle": _safe_get(theta_base, 1),
                "theta_base_exp": _safe_get(theta_base, 2),
                # algorithm progress
                "iteration": (
                    agent_response.get("iteration")
                    if agent_response else None
                ),
                "phase": (
                    agent_response.get("phase")
                    if agent_response else None
                ),
                "step_in_phase": (
                    agent_response.get("step_in_phase")
                    if agent_response else None
                ),
                "phase_budget": (
                    agent_response.get("phase_budget")
                    if agent_response else None
                ),
                "agent_cost": (
                    agent_response.get("cost")
                    if agent_response else None
                ),
                "mode": current_mode,
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


def compute_scale_action(state: FunctionState, theta_step: list) -> int | None:
    if len(theta_step) < 2:
        raise ValueError(
            "theta_step must contain at least theta_stock and theta_idle"
        )

    pi_stock = int(theta_step[0])
    pi_idle = int(theta_step[1])

    idle_on = state.idle_on
    cold = state.cold
    init_free = state.init_free
    current_total = state.total_pods

    if idle_on == 0 and cold > 0:
        to_spawn = 1 + pi_stock
        to_spawn = min(to_spawn, cold)
        desired = current_total + to_spawn
        return min(desired, state.max_replicas)

    if idle_on > 0:
        idle_after = idle_on - 1
        if idle_after < pi_idle and cold > 0:
            to_spawn = pi_stock - (idle_after + init_free)
            if to_spawn > 0:
                to_spawn = min(to_spawn, cold)
                desired = current_total + to_spawn
                return min(desired, state.max_replicas)

    return None


async def apply_scale_action(desired_replicas: int):
    desired_replicas = max(0, min(desired_replicas, MAX_REPLICAS))
    try:
        response = await http_client.get(
            "http://127.0.0.1:3000/scale",
            params={"replicas": desired_replicas},
        )
        if response.status_code == 200:
            logger.info("Scale action applied: %s replicas", desired_replicas)
        else:
            error_msg = (
                f"API ERROR: Scale returned "
                f"{response.status_code} - {response.text}"
            )
            logger.warning(error_msg)
            print(f"\033[91m{error_msg}\033[0m")
    except Exception as e:
        error_msg = f"EXECUTION ERROR: Scale failed - {e}"
        logger.error(error_msg)
        print(f"\033[91m{error_msg}\033[0m")


def handle_agent_response(state: FunctionState, theta_step: list):
    try:
        scale_to = compute_scale_action(state, theta_step)
        if scale_to is not None:
            task = asyncio.create_task(apply_scale_action(scale_to))
            task.add_done_callback(_log_task_failure)

        if len(theta_step) >= 3 and expiration_controller is not None:
            expiration_controller.set_theta_exp(float(theta_step[2]))
    except Exception as e:
        logger.error("Failed to handle scaling: %s", e)


def _log_task_failure(task: asyncio.Task):
    if task.exception():
        logger.error("Scale task failed: %s", task.exception())


def _get_agent_cost_summary(limit: int = 10) -> dict:
    with state_history_lock:
        snapshot = list(state_history[-limit * 2:])

    observations = []
    for entry in reversed(snapshot):
        response = entry.get("agent_response")
        if not isinstance(response, dict) or response.get("cost") is None:
            continue
        observations.append(response)
        if len(observations) >= limit:
            break

    observations.reverse()
    recent_costs = [obs["cost"] for obs in observations]
    latest = observations[-1] if observations else None
    previous_cost = recent_costs[-2] if len(recent_costs) >= 2 else None
    latest_cost = recent_costs[-1] if recent_costs else None

    return {
        "latest_cost": latest_cost,
        "previous_cost": previous_cost,
        "cost_delta": (
            latest_cost - previous_cost
            if latest_cost is not None and previous_cost is not None
            else None
        ),
        "recent_costs": recent_costs,
        "iteration": latest.get("iteration") if latest else None,
        "phase": latest.get("phase") if latest else None,
        "step_in_phase": latest.get("step_in_phase") if latest else None,
        "phase_budget": latest.get("phase_budget") if latest else None,
        "theta": latest.get("theta") if latest else None,
        "theta_step": latest.get("theta_step") if latest else None,
    }


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
        "mode": "evaluation" if eval_mode.active else "training",
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
        "agent": _get_agent_cost_summary(),
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
# Evaluation Mode Endpoints
# ---------------------------------------------------------------------------

@app.post("/mode/evaluate")
async def start_evaluation():
    """
    Switch to evaluation mode.

    Reads the current un-perturbed theta from the agent via GET /theta,
    then uses it as a fixed policy for all subsequent requests.
    No more /event calls to the agent — cost is measured under the
    learned policy without perturbation noise.

    This is how the paper measures cost for Figures 3 and 4.
    """
    try:
        response = await http_client.get(f"{AGENT_URL}/theta", timeout=5.0)
        if response.status_code != 200:
            return JSONResponse(
                content={"error": f"Agent returned {response.status_code}"},
                status_code=502,
            )
        data = response.json()
        theta = data.get("theta")
        theta_step = data.get("theta_step")

        if not theta or not theta_step:
            return JSONResponse(
                content={"error": "Agent response missing theta", "data": data},
                status_code=502,
            )

        eval_mode.enable(theta_step=theta_step, theta_base=theta)

        return {
            "mode": "evaluation",
            "theta_base": theta,
            "theta_step": theta_step,
            "message": "Evaluation mode active. Send traffic to measure cost.",
        }

    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to fetch theta from agent: {e}"},
            status_code=502,
        )


@app.post("/mode/evaluate/manual")
async def start_evaluation_manual(body: dict):
    """
    Switch to evaluation with manually specified theta.
    Body: {"theta_step": [3, 2, 5], "theta_base": [3.1, 2.0, 5.0]}
    """
    theta_step = body.get("theta_step")
    theta_base = body.get("theta_base", theta_step)
    if not theta_step:
        return JSONResponse(
            content={"error": "theta_step is required"},
            status_code=400,
        )
    eval_mode.enable(theta_step=theta_step, theta_base=theta_base)
    return {
        "mode": "evaluation",
        "theta_step": theta_step,
        "theta_base": theta_base,
    }


@app.post("/mode/train")
async def resume_training():
    """Switch back to training mode. Agent is called on every request."""
    eval_mode.disable()
    return {"mode": "training"}


@app.get("/mode")
async def get_mode():
    return {
        "mode": "evaluation" if eval_mode.active else "training",
        "theta_step": eval_mode.theta_step,
        "theta_base": eval_mode.theta_base,
    }


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
        "sampling_window_seconds": SAMPLING_WINDOW,
        "total_samples": len(samples),
        "duration_seconds": (
            round(samples[-1]["timestamp"] - samples[0]["timestamp"], 1)
            if len(samples) > 1 else 0
        ),
        "nsgd_cost": _stats(costs),
        "response_latency_ms": _stats(latencies),
        "throughput_pct": _stats(throughputs),
        "replicas": _stats(replicas_list),
        "avg_cpu_normalized": _stats(cpus),
        "avg_mem_normalized": _stats(mems),
        "csv_files": {
            "samples": os.path.join(EXPERIMENT_LOG_DIR, "samples.csv"),
            "requests": os.path.join(EXPERIMENT_LOG_DIR, "request_log.csv"),
            "monitor": os.path.join(EXPERIMENT_LOG_DIR, "monitor.csv"),
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
# Health & Debug
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "evaluation" if eval_mode.active else "training",
        "watcher_alive": pod_watcher.is_alive(),
        "metrics_collector_alive": (
            metrics_collector.is_alive() if metrics_collector else False
        ),
        "gateway": OPENFAAS_GATEWAY,
    }


@app.get("/debug/scaling")
async def debug_scaling():
    pod_counts = pod_watcher.get_counts()
    state = FunctionState(
        timestamp=time.time(),
        in_flight=in_flight_counter.count,
        ready_pods=pod_counts["ready"],
        not_ready_pods=pod_counts["not_ready"],
        total_pods=pod_counts["total"],
        max_replicas=MAX_REPLICAS,
    )

    latest_theta = None
    with state_history_lock:
        for entry in reversed(state_history):
            resp = entry.get("agent_response")
            if isinstance(resp, dict) and resp.get("theta_step"):
                latest_theta = resp["theta_step"]
                break

    if latest_theta is None and eval_mode.active:
        latest_theta = eval_mode.theta_step

    scale_to = None
    if latest_theta:
        scale_to = compute_scale_action(state, latest_theta)

    return {
        "state": asdict(state),
        "latest_theta_step": latest_theta,
        "would_scale_to": scale_to,
        "reason": (
            "cold start branch"
            if state.idle_on == 0 and state.cold > 0
            else "proactive branch"
            if (state.idle_on > 0 and latest_theta
                and state.idle_on - 1 < int(latest_theta[1]))
            else "no scaling needed"
            if state.idle_on > 0
            else "saturated"
        ),
        "expiration_timeout": (
            expiration_controller.timeout_seconds
            if expiration_controller else None
        ),
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting NSGD state tracker on port {TRACKER_PORT}")
    print(f"  Proxying to: {OPENFAAS_GATEWAY}")
    print(f"  Tracking:    {FUNCTION_NAME} in {FUNCTION_NAMESPACE}")
    print(f"  Max replicas: {MAX_REPLICAS}")
    print(f"  Prometheus:  {PROMETHEUS_URL}")
    print(f"  Sampling:    every {SAMPLING_WINDOW}s")
    print(f"  Logs:        {EXPERIMENT_LOG_DIR}/")
    print()
    print("CSV files (written automatically):")
    print(f"  {EXPERIMENT_LOG_DIR}/samples.csv       (every {SAMPLING_WINDOW}s)")
    print(f"  {EXPERIMENT_LOG_DIR}/request_log.csv   (every request)")
    print()
    print("Jupyter:")
    print("  import pandas as pd")
    print(f"  samples  = pd.read_csv('{EXPERIMENT_LOG_DIR}/samples.csv')")
    print(f"  requests = pd.read_csv('{EXPERIMENT_LOG_DIR}/request_log.csv')")
    print()
    print("Evaluation mode (after training):")
    print(f"  curl -X POST http://127.0.0.1:{TRACKER_PORT}/mode/evaluate")
    print()

    uvicorn.run(
        "app_with_args:app",
        host="0.0.0.0",
        port=TRACKER_PORT,
        log_level=LOG_LEVEL.lower(),
        workers=1,
    )

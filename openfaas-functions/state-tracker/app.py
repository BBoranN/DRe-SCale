"""
State Tracker Middleware for OpenFaaS Functions
================================================

Sits between your load generator and the OpenFaaS gateway.
Tracks exact replica states from the NSGD paper.

Traffic flow:
    Load Generator (hey) --> State Tracker (:8000) --> OpenFaaS Gateway (:8080)
"""

import os
import time
import json
import logging
import threading
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from pod_watcher import PodWatcher
from exp_controller import ExpirationController

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The REAL gateway URL that this proxy forwards to.
# Since you port-forward the gateway to localhost:8080,
# and this tracker runs on localhost:8000, we forward to 8080.
OPENFAAS_GATEWAY = os.getenv("OPENFAAS_GATEWAY", "http://127.0.0.1:8080")

FUNCTION_NAME = os.getenv("FUNCTION_NAME", "matmul")
FUNCTION_NAMESPACE = os.getenv("FUNCTION_NAMESPACE", "openfaas-fn")
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "24"))
TRACKER_PORT = int(os.getenv("TRACKER_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "100000"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("state-tracker")


# ---------------------------------------------------------------------------
# In-Flight Counter
# ---------------------------------------------------------------------------

class InFlightCounter:
    """
    Exact count of requests currently inside the proxy.

    Why a lock and not asyncio? Because uvicorn can run request handlers
    across threads (even in single-worker mode, the threadpool executor
    handles blocking calls). A threading.Lock is the safest primitive here.
    """

    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def increment(self) -> int:
        with self._lock:
            self._count += 1
            return self._count

    def decrement(self) -> int:
        with self._lock:
            self._count = max(0, self._count - 1)
            return self._count


# ---------------------------------------------------------------------------
# State Computation
# ---------------------------------------------------------------------------
""" 
OLD data class doesn't have rejected status
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

    def __post_init__(self):
        # ---- Step 1: cold = pods that don't exist ----
        self.cold = self.max_replicas - self.total_pods

        # ---- Step 2: busy = ready pods currently serving ----
        # Can't have more busy pods than ready pods OR in-flight requests
        self.busy = min(self.in_flight, self.ready_pods)

        # ---- Step 3: idle_on = ready pods NOT serving ----
        self.idle_on = self.ready_pods - self.busy

        # ---- Step 4: blocked = requests with no ready pod ----
        # These requests are sitting in the gateway queue, waiting
        # for an initializing pod to become ready
        blocked_requests = max(0, self.in_flight - self.ready_pods)

        # ---- Step 5: init_reserved = initializing pods claimed by blocked requests ----
        # Each blocked request "claims" one initializing pod
        self.init_reserved = min(blocked_requests, self.not_ready_pods)

        # ---- Step 6: init_free = initializing pods with no request waiting ----
        # These were pre-warmed proactively
        self.init_free = self.not_ready_pods - self.init_reserved
 """

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
    rejected: bool = False           # ground-truth from gateway response
    saturated_at_arrival: bool = False  # derived: busy + init_reserved == N
    response_status: int = 0          # for debugging / post-hoc filtering

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
    global pod_watcher, http_client

    # Start pod watcher
    pod_watcher = PodWatcher(
        function_name=FUNCTION_NAME,
        namespace=FUNCTION_NAMESPACE,
    )
    pod_watcher.start()
    logger.info(f"Pod watcher started for {FUNCTION_NAME} in {FUNCTION_NAMESPACE}")
    expiration_controller = ExpirationController(
        function_name=FUNCTION_NAME,
        gateway_url=OPENFAAS_GATEWAY,
        pod_watcher=pod_watcher,
        in_flight_counter=in_flight_counter,
        max_replicas=MAX_REPLICAS,
        k_exp=1000.0,
    )

    expiration_controller.start()
    # HTTP client for proxying.
    # timeout=300s to match your function's 10s exec_timeout plus overhead.
    # In practice the gateway will timeout first.
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    logger.info(f"Proxying to: {OPENFAAS_GATEWAY}")

    yield

    expiration_controller.stop()
    pod_watcher.stop()
    await http_client.aclose()


app = FastAPI(title="State Tracker", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Proxy Endpoint
# ---------------------------------------------------------------------------

@app.api_route(
    "/function/{function_name:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_function_call(function_name: str, request: Request):
    """
    Proxy a function call to the real gateway.

    For every request that arrives:
      1. Increment in-flight counter
      2. Snapshot the state (this is what NSGD sees at "job arrival")
      3. Forward to the gateway
      4. Decrement in-flight counter on completion

    Your load generator calls:
        hey -n 1 -c 1 http://127.0.0.1:8000/function/matmul
    
    This proxy forwards to:
        http://127.0.0.1:8080/function/matmul
    """

    pod_counts = pod_watcher.get_counts()
    state = FunctionState(
        timestamp=time.time(),
        in_flight=in_flight_counter.count,
        ready_pods=pod_counts["ready"],
        not_ready_pods=pod_counts["not_ready"],
        total_pods=pod_counts["total"],
        max_replicas=MAX_REPLICAS,
    )

    # Record now, get a handle to update later.
    entry = _record_state(state)

    current_in_flight = in_flight_counter.increment()

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

        # Update the recorded entry with the outcome.
        _mark_outcome(
            entry,
            status=response.status_code,
            rejected=response.status_code in REJECTION_STATUS_CODES,
        )

        response_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
        }
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    except httpx.TimeoutException:
        # Timeout = job was admitted but slow. NOT a rejection.
        logger.warning(f"Timeout proxying to {OPENFAAS_GATEWAY}")
        _mark_outcome(entry, status=504, rejected=False)
        return Response(content="Gateway timeout", status_code=504)

    except httpx.ConnectError:
        # Can't reach gateway = infrastructure error, NOT a rejection.
        logger.error(f"Cannot connect to gateway at {OPENFAAS_GATEWAY}")
        _mark_outcome(entry, status=502, rejected=False)
        return Response(
            content=f"Cannot reach gateway at {OPENFAAS_GATEWAY}",
            status_code=502,
        )

    except Exception as e:
        logger.error(f"Proxy error: {e}")
        _mark_outcome(entry, status=502, rejected=False)
        return Response(content=str(e), status_code=502)

    finally:
        in_flight_counter.decrement()
"""
    # Step 1: count this request
    current_in_flight = in_flight_counter.increment()

    # Step 2: snapshot state at arrival time
    pod_counts = pod_watcher.get_counts()
    state = FunctionState(
        timestamp=time.time(),
        in_flight=current_in_flight,
        ready_pods=pod_counts["ready"],
        not_ready_pods=pod_counts["not_ready"],
        total_pods=pod_counts["total"],
        max_replicas=MAX_REPLICAS,
    )

    logger.debug(
        f"REQ IN  | inflight={current_in_flight} "
        f"cold={state.cold} init_res={state.init_reserved} "
        f"init_free={state.init_free} busy={state.busy} idle={state.idle_on}"
    )

    # Record state
    entry = _record_state(state)

    # Step 3: forward to gateway
    try:
        body = await request.body()

        # Build the target URL: gateway + same path
        target_url = f"{OPENFAAS_GATEWAY}/function/{function_name}"

        # Forward headers, but remove hop-by-hop ones that cause issues
        forward_headers = {}
        for key, value in request.headers.items():
            lower = key.lower()
            if lower not in ("host", "content-length", "transfer-encoding"):
                forward_headers[key] = value

        response = await http_client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=forward_headers,
        )

        # Return the gateway's response as-is
        # Filter out hop-by-hop response headers too
        response_headers = {}
        for key, value in response.headers.items():
            lower = key.lower()
            if lower not in ("transfer-encoding", "content-encoding", "content-length"):
                response_headers[key] = value

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    except httpx.TimeoutException:
        logger.warning(f"Timeout proxying to {OPENFAAS_GATEWAY}")
        return Response(content="Gateway timeout", status_code=504)

    except httpx.ConnectError:
        logger.error(f"Cannot connect to gateway at {OPENFAAS_GATEWAY}")
        return Response(
            content=f"Cannot reach gateway at {OPENFAAS_GATEWAY}",
            status_code=502,
        )

    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return Response(content=str(e), status_code=502)

    finally:
        # Step 4: always decrement, even on error
        in_flight_counter.decrement()
 """

def _record_state(state: FunctionState) -> dict:
    """Append state and return a reference to the dict for later update."""
    entry = asdict(state)
    with state_history_lock:
        state_history.append(entry)
        if len(state_history) > MAX_HISTORY:
            del state_history[: MAX_HISTORY // 10]
    return entry  # mutable reference — safe because dicts are by reference

def _mark_outcome(entry: dict, status: int, rejected: bool):
    """Update an already-recorded entry with the gateway response outcome."""
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
    """Current state snapshot. Poll this from your NSGD agent."""
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
    """
    Per-request state snapshots — training data for NSGD.
    Each entry is the state at the moment a request arrived.
    """
    with state_history_lock:
        if since > 0:
            filtered = [s for s in state_history if s["timestamp"] > since]
        else:
            filtered = list(state_history)
        return filtered[-limit:]


@app.get("/states/summary")
async def get_state_summary():
    """Human-readable summary for debugging."""
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
        "history_size": len(state_history),
        "watcher_alive": pod_watcher.is_alive(),
    }


@app.get("/states/export")
async def export_history():
    """
    Export full history as a JSON download.
    Use this to save training data to disk.
    """
    with state_history_lock:
        return JSONResponse(
            content=state_history,
            headers={"Content-Disposition": "attachment; filename=state_history.json"},
        )


@app.delete("/states/history")
async def clear_history():
    """Clear state history. Useful between experiment runs."""
    with state_history_lock:
        count = len(state_history)
        state_history.clear()
    return {"cleared": count}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "watcher_alive": pod_watcher.is_alive(),
        "gateway": OPENFAAS_GATEWAY,
    }





# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting state tracker on port {TRACKER_PORT}")
    print(f"  Proxying to: {OPENFAAS_GATEWAY}")
    print(f"  Tracking:    {FUNCTION_NAME} in {FUNCTION_NAMESPACE}")
    print(f"  Max replicas: {MAX_REPLICAS}")
    print()
    print("Send function calls to:")
    print(f"  http://127.0.0.1:{TRACKER_PORT}/function/{FUNCTION_NAME}")
    print()
    print("Monitor states at:")
    print(f"  http://127.0.0.1:{TRACKER_PORT}/states/current")
    print(f"  http://127.0.0.1:{TRACKER_PORT}/states/summary")
    print()

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=TRACKER_PORT,
        log_level=LOG_LEVEL.lower(),
        workers=1,  # MUST be 1 for accurate in-flight count
    )

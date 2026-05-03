import requests
import time
import threading
import math
import random
from dataclasses import dataclass, field

# --- CONFIGURATION ---
PROMETHEUS_URL = "http://127.0.0.1:9090/api/v1/query"
KSM_URL        = "http://10.152.183.20:8080/metrics"
OPENFAAS_GATEWAY = "http://127.0.0.1:8080"
OPENFAAS_FUNC_NAME    = "matmul.openfaas-fn"
K8S_DEPLOYMENT_NAME   = "matmul"
POLL_INTERVAL = 1
# ---------------------


@dataclass
class InitTracker:
    """
    Single source of truth for the init-free / init-reserved split.
    Updated by the autoscaler at decision time, reconciled against
    Kubernetes reality every poll cycle.
    """
    init_reserved: int = 0
    init_free:     int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def on_cold_start_spawned(self, n_stock: int):
        with self._lock:
            self.init_reserved += 1
            self.init_free     += n_stock

    def on_preemptive_spawn(self, n: int):
        with self._lock:
            self.init_free += n

    def on_pod_became_ready(self):
        with self._lock:
            if self.init_reserved > 0:
                self.init_reserved -= 1
            elif self.init_free > 0:
                self.init_free -= 1

    def reconcile(self, actual_total_initializing: int):
        with self._lock:
            tracked = self.init_reserved + self.init_free
            drift   = tracked - actual_total_initializing

            if drift > 0:
                free_reduction = min(self.init_free, drift)
                self.init_free     -= free_reduction
                self.init_reserved -= (drift - free_reduction)
                self.init_reserved  = max(0, self.init_reserved)

            elif drift < 0:
                self.init_free += abs(drift)

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.init_reserved, self.init_free


# Global tracker
tracker = InitTracker()
_prev_ready = 0

# ------------------------------------------------------------------ #
#  Calculation Logic (Policy 1)                                      #
# ------------------------------------------------------------------ #

def randomized_round(theta: float) -> int:
    """
    Converts continuous learning parameters (theta) into integers.
    Returns ceil(theta) with probability (theta - floor(theta)),
    otherwise returns floor(theta).
    """
    floor_val = math.floor(theta)
    prob_ceil = theta - floor_val
    
    if random.random() < prob_ceil:
        return math.ceil(theta)
    else:
        return floor_val

def evaluate_autoscaling_policy(state: dict, theta_stock: float, theta_idle: float) -> tuple[int, int]:
    """
    Evaluates the current state against the learned parameters to 
    calculate how many init-reserved and init-free replicas to spawn.
    Automatically updates the tracker with the decision.
    """
    pi_theta_stock = randomized_round(theta_stock)
    pi_theta_idle = randomized_round(theta_idle)

    idle = state["x1_idle"]
    init_free = state["init_free"]
    cold = state["cold"]

    to_spawn_reserved = 0
    to_spawn_free = 0

    # Policy 1: Cold Start Allocation
    if idle == 0:
        if cold > 0:
            to_spawn_reserved = 1
            to_spawn_free = min(pi_theta_stock, cold - 1)
            
            # Update internal tracker
            tracker.on_cold_start_spawned(to_spawn_free)
    
    # Policy 1: Warm Start & Preemptive Preparation
    elif idle > 0:
        # Simulate taking one idle pod for the incoming request
        effective_idle = idle - 1 
        
        if effective_idle < pi_theta_idle and cold > 0:
            target_free_spawn = pi_theta_stock - (effective_idle + init_free)
            
            if target_free_spawn > 0:
                to_spawn_free = min(target_free_spawn, cold)
                
                # Update internal tracker
                tracker.on_preemptive_spawn(to_spawn_free)

    return to_spawn_reserved, to_spawn_free


# ------------------------------------------------------------------ #
#  Metric fetchers & State Computation                               #
# ------------------------------------------------------------------ #

def get_inflight_requests() -> int:
    query = (
        f'sum(gateway_function_invocation_started{{function_name="{OPENFAAS_FUNC_NAME}"}}) '
        f'- sum(gateway_function_invocation_total{{function_name="{OPENFAAS_FUNC_NAME}"}})'
    )
    try:
        r = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=5)
        result = r.json()["data"]["result"]
        return max(0, int(float(result[0]["value"][1]))) if result else 0
    except Exception:
        return 0

def get_kubernetes_metrics() -> tuple[int, int, int]:
    try:
        lines = requests.get(KSM_URL, timeout=5).text.splitlines()
        p_max = p_total = p_ready = 0
        for line in lines:
            if K8S_DEPLOYMENT_NAME not in line or 'namespace="openfaas-fn"' not in line:
                continue
            if line.startswith("kube_deployment_spec_replicas{"):
                p_max   = int(float(line.split()[-1]))
            elif line.startswith("kube_deployment_status_replicas{"):
                p_total = int(float(line.split()[-1]))
            elif line.startswith("kube_deployment_status_replicas_ready{"):
                p_ready = int(float(line.split()[-1]))
        return p_max, p_total, p_ready
    except Exception as e:
        print(f"[ERROR] K8s metrics failed: {e}")
        return 0, 0, 0

def get_agent_states() -> dict:
    global _prev_ready

    inflight               = get_inflight_requests()
    n_limit, p_total, p_ready = get_kubernetes_metrics()

    newly_ready = max(0, p_ready - _prev_ready)
    for _ in range(newly_ready):
        tracker.on_pod_became_ready()
    _prev_ready = p_ready

    actual_initializing = max(0, p_total - p_ready)
    tracker.reconcile(actual_initializing)

    x4_init_reserved, init_free = tracker.snapshot()

    busy  = min(inflight, p_ready)
    idle  = max(0, p_ready - busy)
    cold  = max(0, n_limit - p_total)

    x3 = actual_initializing
    x4_init_reserved = min(x4_init_reserved, x3)
    init_free        = min(init_free,        x3 - x4_init_reserved)

    return {
        "n_limit":          n_limit,
        "cold":             cold,
        "x1_idle":          idle,
        "x2_busy":          busy,
        "x3_total_init":    x3,
        "x4_init_reserved": x4_init_reserved,
        "init_free":        init_free,
    }


# ------------------------------------------------------------------ #
#  Main loop                                                         #
# ------------------------------------------------------------------ #

def main():
    print("Starting NSGD Agent Observer with Calculation Logic...")
    print("-" * 110)
    print(f"{'N_MAX':<6} | {'COLD':<6} | {'IDLE(x1)':<9} | {'BUSY(x2)':<9} | "
          f"{'INIT(x3)':<9} | {'RES(x4)':<9} | {'FREE':<6} | {'DECISION (RES, FREE)'}")
    print("-" * 110)

    # Example learned parameters from the paper (you would update these via your ML logic)
    current_theta_stock = 3.2 
    current_theta_idle = 2.7  

    while True:
        s = get_agent_states()
        
        # Calculate autoscaling decision based on current state and theta values
        spawn_res, spawn_free = evaluate_autoscaling_policy(s, current_theta_stock, current_theta_idle)
        
        # If spawn_res or spawn_free > 0 here, you would trigger the Kubernetes API 
        # to scale up the deployment by (spawn_res + spawn_free).
        
        decision_str = f"+{spawn_res} Res, +{spawn_free} Free" if (spawn_res > 0 or spawn_free > 0) else "None"

        print(
            f"{s['n_limit']:<6} | {s['cold']:<6} | {s['x1_idle']:<9} | "
            f"{s['x2_busy']:<9} | {s['x3_total_init']:<9} | "
            f"{s['x4_init_reserved']:<9} | {s['init_free']:<6} | {decision_str}"
        )
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
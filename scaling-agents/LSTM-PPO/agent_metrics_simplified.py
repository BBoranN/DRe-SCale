import requests

# ── Where to fetch data from ──────────────────────────────────────────
PROMETHEUS_URL     = "http://127.0.0.1:9090/api/v1/query"
KUBE_METRICS_URL   = "http://10.152.183.20:8080/metrics"
FUNCTION_NAME      = "matmul.openfaas-fn"
DEPLOYMENT_NAME    = "matmul"


# ── Step 1: Ask Prometheus "how many requests are being processed RIGHT NOW?" ──
def get_inflight_requests() -> int:
    """
    Inflight = requests that STARTED but haven't FINISHED yet.
    Prometheus tracks both counters, we just subtract them.
    """
    query = (
        f'sum(gateway_function_invocation_started{{function_name="{FUNCTION_NAME}"}}) '
        f'- sum(gateway_function_invocation_total{{function_name="{FUNCTION_NAME}"}})'
    )

    try:
        response = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=5)
        result   = response.json()["data"]["result"]
        inflight = int(float(result[0]["value"][1])) if result else 0
        return max(0, inflight)   # never return a negative number
    except Exception:
        return 0                  # if Prometheus is down, assume 0


# ── Step 2: Ask Kubernetes "what is the state of my pods?" ────────────
def get_pod_counts() -> tuple[int, int, int]:
    """
    Reads raw metrics from kube-state-metrics and returns three numbers:

        max_pods   → the ceiling: how many pods are ALLOWED to exist
        total_pods → how many pods EXIST right now (including ones still starting)
        ready_pods → how many pods are FULLY STARTED and can serve traffic
    """
    max_pods = total_pods = ready_pods = 0

    try:
        raw_lines = requests.get(KUBE_METRICS_URL, timeout=5).text.splitlines()

        for line in raw_lines:
            # Skip lines unrelated to our deployment
            if DEPLOYMENT_NAME not in line or 'namespace="openfaas-fn"' not in line:
                continue

            value = int(float(line.split()[-1]))   # last token on each line is the number

            if   line.startswith("kube_deployment_spec_replicas{"):
                max_pods   = value   # the cap set in your deployment config

            elif line.startswith("kube_deployment_status_replicas{"):
                total_pods = value   # all pods: starting + ready

            elif line.startswith("kube_deployment_status_replicas_ready{"):
                ready_pods = value   # only fully started pods

    except Exception as e:
        print(f"[ERROR] Could not reach kube-state-metrics: {e}")

    return max_pods, total_pods, ready_pods


# ── Step 3: Combine everything into one clean state snapshot ──────────
def get_current_state() -> dict:
    """
    Using the raw numbers above, we derive the four states from the diagram:

        COLD      → empty slots  (no pod exists yet, but one COULD be created)
        INIT      → pods that exist but are still STARTING UP
        IDLE-ON   → pods that are fully ready but currently doing NOTHING
        BUSY      → pods that are fully ready and ACTIVELY handling a request
    """
    inflight_requests          = get_inflight_requests()
    max_pods, total_pods, ready_pods = get_pod_counts()

    # ── Derive the four states ────────────────────────────────────────
    cold    = max_pods   - total_pods   # slots with no pod in them yet
    init    = total_pods - ready_pods   # pods that exist but aren't ready yet
    busy    = min(inflight_requests, ready_pods)   # can't be busier than pods we have
    idle_on = ready_pods - busy                    # ready pods not doing anything

    return {
        "max_pods" : max_pods,
        "cold"     : cold,
        "init"     : init,
        "idle_on"  : idle_on,
        "busy"     : busy,
    }


# ── Step 4: Print a live table, refreshing every second ───────────────
def main():
    import time

    print("Live pod state — refreshing every second")
    print("─" * 55)
    print(f"{'MAX':>5}  {'COLD':>6}  {'INIT':>6}  {'IDLE-ON':>8}  {'BUSY':>6}")
    print("─" * 55)

    while True:
        state = get_current_state()

        print("─" * 55)
        print(f"{'MAX':>5}  {'COLD':>6}  {'INIT':>6}  {'IDLE-ON':>8}  {'BUSY':>6}")
        print("─" * 55)
        print(
            f"{state['max_pods']:>5}  "
            f"{state['cold']:>6}  "
            f"{state['init']:>6}  "
            f"{state['idle_on']:>8}  "
            f"{state['busy']:>6}"
        )

        time.sleep(1)


if __name__ == "__main__":
    main()
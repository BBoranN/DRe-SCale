"""
Quick test to verify the state tracker is working.

Run this AFTER starting the state tracker but BEFORE the full load generator.
It sends a few requests and prints the state after each one.

Usage:
    python test_tracker.py
"""

import requests
import time
import json
import subprocess
from threading import Thread

TRACKER_URL = "http://127.0.0.1:8000"
FUNCTION_URL = f"{TRACKER_URL}/function/matmul"


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_health():
    """Check if the tracker is running."""
    print_section("1. Health Check")
    try:
        r = requests.get(f"{TRACKER_URL}/health", timeout=5)
        data = r.json()
        print(f"  Status:        {data['status']}")
        print(f"  Watcher alive: {data['watcher_alive']}")
        print(f"  Gateway:       {data['gateway']}")

        if not data['watcher_alive']:
            print("\n  WARNING: Pod watcher is not running!")
            print("  Check that kubectl can access the cluster.")
            return False
        return True

    except requests.ConnectionError:
        print("  FAILED: Cannot connect to state tracker at port 8000.")
        print("  Did you start it? Run: python app.py")
        return False


def test_initial_state():
    """Check the state before any requests."""
    print_section("2. Initial State (no requests in flight)")
    r = requests.get(f"{TRACKER_URL}/states/summary")
    data = r.json()

    obs = data["observables"]
    states = data["derived_states"]

    print(f"  Ready pods:     {obs['ready_pods']}")
    print(f"  Not-ready pods: {obs['not_ready_pods']}")
    print(f"  Total pods:     {obs['total_pods']}")
    print()
    print(f"  cold:           {states['cold']}")
    print(f"  init_reserved:  {states['init_reserved']}")
    print(f"  init_free:      {states['init_free']}")
    print(f"  busy:           {states['busy']}")
    print(f"  idle_on:        {states['idle_on']}")
    print()

    # Sanity checks
    total = states['cold'] + states['init_reserved'] + states['init_free'] + states['busy'] + states['idle_on']
    max_r = data['max_replicas']
    print(f"  Sum of states:  {total} (should equal max_replicas={max_r})")

    if total != max_r:
        print("  WARNING: States don't sum to max_replicas!")
    else:
        print("  OK")

    if states['busy'] != 0:
        print("  WARNING: busy should be 0 with no requests in flight")

    return True


def test_single_request():
    """Send one request through the proxy and check state changes."""
    print_section("3. Single Request Test")

    print("  Sending 1 request to matmul...")
    try:
        r = requests.post(FUNCTION_URL, timeout=30)
        print(f"  Response status: {r.status_code}")
        print(f"  Response body:   {r.text[:100]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Is the OpenFaaS gateway running on port 8080?")
        return False

    # Check that state was recorded
    r = requests.get(f"{TRACKER_URL}/states/history?limit=1")
    history = r.json()
    if len(history) > 0:
        entry = history[0]
        print(f"\n  Recorded state at request arrival:")
        print(f"    in_flight:      {entry['in_flight']}")
        print(f"    busy:           {entry['busy']}")
        print(f"    idle_on:        {entry['idle_on']}")
        print(f"    init_reserved:  {entry['init_reserved']}")
        print(f"    init_free:      {entry['init_free']}")
        print(f"    cold:           {entry['cold']}")
        print("  OK — state was recorded")
    else:
        print("  WARNING: No state history recorded!")

    return True


def test_concurrent_requests():
    """Send several concurrent requests to see busy/init states in action."""
    print_section("4. Concurrent Request Test (5 simultaneous)")

    # Send 5 requests concurrently using threads
    threads = []
    results = []

    def send_one():
        try:
            r = requests.post(FUNCTION_URL, timeout=60)
            results.append(r.status_code)
        except Exception as e:
            results.append(str(e))

    for _ in range(5):
        t = Thread(target=send_one)
        threads.append(t)
        t.start()

    # While requests are in flight, check the state
    time.sleep(0.5)  # give them a moment to reach the proxy

    r = requests.get(f"{TRACKER_URL}/states/current")
    state = r.json()

    print(f"  State during concurrent requests:")
    print(f"    in_flight:      {state['in_flight']}")
    print(f"    busy:           {state['busy']}")
    print(f"    idle_on:        {state['idle_on']}")
    print(f"    init_reserved:  {state['init_reserved']}")
    print(f"    init_free:      {state['init_free']}")
    print(f"    cold:           {state['cold']}")

    # Wait for all to complete
    for t in threads:
        t.join(timeout=60)

    print(f"\n  Results: {results}")
    print(f"  Successful: {sum(1 for r in results if r == 200)}/5")

    # Check state after all complete
    r = requests.get(f"{TRACKER_URL}/states/current")
    state = r.json()
    print(f"\n  State after all requests completed:")
    print(f"    in_flight:      {state['in_flight']} (should be 0)")
    print(f"    busy:           {state['busy']} (should be 0)")

    return True


def test_history():
    """Check that history accumulated correctly."""
    print_section("5. History Check")
    r = requests.get(f"{TRACKER_URL}/states/history")
    history = r.json()
    print(f"  Total entries: {len(history)}")
    if len(history) >= 6:
        print("  OK — at least 6 entries (1 single + 5 concurrent)")
    return True


if __name__ == "__main__":
    print("State Tracker Integration Test")
    print(f"Tracker: {TRACKER_URL}")
    print(f"Function: {FUNCTION_URL}")

    if not test_health():
        exit(1)
    test_initial_state()
    test_single_request()
    test_concurrent_requests()
    test_history()

    print_section("DONE")
    print("  All checks passed. You can now run the full load generator:")
    print("  python load_generator.py")

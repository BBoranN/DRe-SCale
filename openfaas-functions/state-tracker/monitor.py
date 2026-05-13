"""
Compact State Monitor
=====================

Replaces:  watch -n 1 'curl -s .../states/summary | python -m json.tool'

Usage:     python monitor.py
           python monitor.py --interval 0.5
"""

import os
import sys
import time
import argparse
import requests
import csv
from datetime import datetime

TRACKER_URL = os.getenv("TRACKER_URL", "http://127.0.0.1:8000")
BASE_LOG_DIR = "/home/kuscu/DRe-SCale/openfaas-functions/state-tracker/logs"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def fmt_float(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def setup_log_directory():
    """Creates a new timestamped folder and returns the full path for the CSV."""
    # Create a unique folder name based on when the script started
    folder_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    full_dir_path = os.path.join(BASE_LOG_DIR, folder_name)
    
    # Make the directory (and any parent directories if they don't exist)
    os.makedirs(full_dir_path, exist_ok=True)
    
    # Return the exact file path where the CSV should be written
    return os.path.join(full_dir_path, "monitor.csv")

def log_to_csv(data, filename):
    """Writes the current state data as a row in a CSV file."""
    file_exists = os.path.isfile(filename)
    
    # Extract the nested dictionaries
    obs = data.get("observables", {})
    s = data.get("derived_states", {})
    agent = data.get("agent", {})
    
    # Flatten the data into a single dictionary for the CSV row
    row = {
        "timestamp": time.time(),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cold": s.get("cold"),
        "init_free": s.get("init_free"),
        "init_reserved": s.get("init_reserved"),
        "busy": s.get("busy"),
        "idle_on": s.get("idle_on"),
        "total_pods": obs.get("total_pods"),
        "ready_pods": obs.get("ready_pods"),
        "not_ready_pods": obs.get("not_ready_pods"),
        "inflight_requests": obs.get("in_flight_requests"),
        "rejected_requests": obs.get("total_rejected_requests", 0),
        "latest_cost": agent.get("latest_cost"),
        "cost_delta": agent.get("cost_delta"),
        "iteration": agent.get("iteration"),
        "phase": agent.get("phase")
    }
    
    fieldnames = list(row.keys())
    
    # Open the file in append mode ('a')
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader() # Write headers only the first time
        writer.writerow(row)

def monitor(interval: float, csv_path: str):
    while True:
        try:
            r = requests.get(f"{TRACKER_URL}/states/summary", timeout=3)
            data = r.json()

            log_to_csv(data, filename=csv_path)

            obs = data["observables"]
            s = data["derived_states"]
            inflight = obs["in_flight_requests"]
            hist = data["history_size"]
            alive = "OK" if data["watcher_alive"] else "DOWN"
            rejected = obs.get("total_rejected_requests", 0)
            agent = data.get("agent", {})
            latest_cost = fmt_float(agent.get("latest_cost"))
            cost_delta = fmt_float(agent.get("cost_delta"))
            iteration = agent.get("iteration")
            phase = agent.get("phase") or "n/a"
            theta_step = agent.get("theta_step") or []
            recent_costs = agent.get("recent_costs") or []
            recent_costs_text = " -> ".join(fmt_float(c) for c in recent_costs[-5:]) or "n/a"

            clear()

            print(f"  matmul state tracker   watcher: {alive}")
            print(f"  ─────────────────────────────────")
            print(f"  cold          {s['cold']:>4}  │  pods  {obs['total_pods']:>2}/{data['max_replicas']}")
            print(f"  init_free     {s['init_free']:>4}  │  ready {obs['ready_pods']:>2}")
            print(f"  init_reserved {s['init_reserved']:>4}  │  init  {obs['not_ready_pods']:>2}")
            print(f"  busy          {s['busy']:>4}  │  rejected {rejected}")
            print(f"  idle_on       {s['idle_on']:>4}  │  inflight {inflight}")
            print(f"  ─────────────────────────────────")
            print(f"  cost       {latest_cost:>7}  │  delta {cost_delta}")
            print(f"  iter       {str(iteration or 'n/a'):>7}  │  phase {phase}")
            print(f"  theta_step {str(theta_step):>18}")
            print(f"  recent     {recent_costs_text}")
            print(f"  ─────────────────────────────────")
            print(f"  history: {hist}  refresh: {interval}s")

        except requests.ConnectionError:
            clear()
            print("  Cannot connect to tracker")
            print(f"  {TRACKER_URL}")
        except Exception as e:
            clear()
            print(f"  Error: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    csv_filepath = setup_log_directory()

    print(f"Logging data to: {csv_filepath}")
    time.sleep(1.5)

    try:
        monitor(args.interval, csv_filepath)
    except KeyboardInterrupt:
        print()

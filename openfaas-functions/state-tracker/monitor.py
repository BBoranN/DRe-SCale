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

TRACKER_URL = os.getenv("TRACKER_URL", "http://127.0.0.1:8000")


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def monitor(interval: float):
    while True:
        try:
            r = requests.get(f"{TRACKER_URL}/states/summary", timeout=3)
            data = r.json()
            obs = data["observables"]
            s = data["derived_states"]
            inflight = obs["in_flight_requests"]
            hist = data["history_size"]
            alive = "OK" if data["watcher_alive"] else "DOWN"
            rejected = obs.get("total_rejected_requests", 0)

            clear()

            print(f"  matmul state tracker   watcher: {alive}")
            print(f"  ─────────────────────────────────")
            print(f"  cold          {s['cold']:>4}  │  pods  {obs['total_pods']:>2}/{data['max_replicas']}")
            print(f"  init_free     {s['init_free']:>4}  │  ready {obs['ready_pods']:>2}")
            print(f"  init_reserved {s['init_reserved']:>4}  │  init  {obs['not_ready_pods']:>2}")
            print(f"  busy          {s['busy']:>4}  │  rejected {rejected}")
            print(f"  idle_on       {s['idle_on']:>4}  │  inflight {inflight}")
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

    try:
        monitor(args.interval)
    except KeyboardInterrupt:
        print()

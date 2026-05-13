"""
Constant-Rate Poisson Workload Generator
========================================

Generates a constant-rate Poisson stream of requests for a fixed duration.
Use this to collect data points for response-time-vs-arrival-rate plots.

Usage:
    python poisson_generator.py --rate 3 --duration 600

To collect data for the full plot, run at several rates back-to-back:
    python poisson_generator.py --rate 1  --duration 600
    python poisson_generator.py --rate 3  --duration 600
    python poisson_generator.py --rate 5  --duration 600
    python poisson_generator.py --rate 10 --duration 600

Each run produces a homogeneous block of samples in your CSVs that you
can group by arrival rate in Jupyter.
"""

import argparse
import random
import subprocess
import time
from threading import Thread


FUNCTION_URL = "http://127.0.0.1:8000/function/matmul"


def fire_request():
    """Send one request via hey, suppressing output."""
    cmd = f"hey -n 1 -c 1 {FUNCTION_URL}"
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(
        description="Constant-rate Poisson workload generator"
    )
    parser.add_argument(
        "--rate", type=float, required=True,
        help="Arrival rate in requests per second (λ)",
    )
    parser.add_argument(
        "--duration", type=int, default=600,
        help="Duration in seconds (default: 600 = 10 minutes)",
    )
    parser.add_argument(
        "--seed", type=int, default=29,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--url", type=str, default=FUNCTION_URL,
        help="Function URL to target",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Constant-rate Poisson generator")
    print(f"  Target:    {args.url}")
    print(f"  Rate:      λ = {args.rate} req/sec")
    print(f"  Duration:  {args.duration}s ({args.duration / 60:.1f} min)")
    print(f"  Expected:  ~{int(args.rate * args.duration)} requests")
    print()

    start = time.time()
    end = start + args.duration
    sent = 0

    try:
        while time.time() < end:
            # Exponential inter-arrival times → Poisson process
            inter_arrival = random.expovariate(args.rate)
            time.sleep(inter_arrival)
            if time.time() >= end:
                break
            Thread(target=fire_request, daemon=True).start()
            sent += 1

            if sent % 50 == 0:
                elapsed = time.time() - start
                actual_rate = sent / elapsed
                print(
                    f"  [{elapsed:.0f}s] sent={sent} "
                    f"actual_rate={actual_rate:.2f} req/s"
                )

    except KeyboardInterrupt:
        print("\nInterrupted")

    elapsed = time.time() - start
    print()
    print(f"Done. Sent {sent} requests in {elapsed:.0f}s "
          f"(actual rate: {sent / elapsed:.2f} req/s)")


if __name__ == "__main__":
    main()

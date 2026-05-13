"""
Measure matmul function execution time in isolation.

Sends sequential requests one at a time with a cooldown between them
so no two requests overlap. Reports both:
  - Function's self-reported execution time (just the matmul work)
  - Total round-trip time (includes networking, gateway, pod scheduling)

Run this when no other workload is hitting the system.
"""

import argparse
import statistics
import time

import requests


def measure(url: str, n_requests: int, cooldown: float, warmup: int):
    function_times = []
    total_times = []

    # Warmup runs — discard these (might hit cold starts)
    print(f"Warmup ({warmup} requests, results discarded)...")
    for i in range(warmup):
        try:
            response = requests.post(url, timeout=60)
            print(f"  warmup {i + 1}: {response.text.strip()[:50]}")
        except Exception as e:
            print(f"  warmup {i + 1}: ERROR {e}")
        time.sleep(cooldown)

    print()
    print(f"Measuring {n_requests} sequential requests with {cooldown}s cooldown...")
    print()

    for i in range(n_requests):
        start = time.monotonic()
        try:
            response = requests.post(url, timeout=60)
            total = time.monotonic() - start

            if response.status_code != 200:
                print(f"Request {i + 1}: HTTP {response.status_code}")
                continue

            # The function returns the latency it measured internally
            function_time = float(response.text.strip())

            function_times.append(function_time)
            total_times.append(total)

            overhead_ms = (total - function_time) * 1000
            print(
                f"Request {i + 1:3d}: "
                f"function={function_time:.3f}s, "
                f"total={total:.3f}s, "
                f"overhead={overhead_ms:.0f}ms"
            )

        except Exception as e:
            print(f"Request {i + 1}: ERROR {e}")

        if i < n_requests - 1:
            time.sleep(cooldown)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    if function_times:
        print(f"Function execution (matmul only):")
        print(f"  mean:   {statistics.mean(function_times):.3f}s")
        print(f"  median: {statistics.median(function_times):.3f}s")
        print(f"  min:    {min(function_times):.3f}s")
        print(f"  max:    {max(function_times):.3f}s")
        if len(function_times) > 1:
            print(f"  stdev:  {statistics.stdev(function_times):.3f}s")

    if total_times:
        print(f"\nTotal round-trip (function + overhead):")
        print(f"  mean:   {statistics.mean(total_times):.3f}s")
        print(f"  median: {statistics.median(total_times):.3f}s")
        print(f"  min:    {min(total_times):.3f}s")
        print(f"  max:    {max(total_times):.3f}s")

        avg_overhead_ms = (
            (statistics.mean(total_times) - statistics.mean(function_times)) * 1000
        )
        print(f"\nMean overhead (proxy + gateway + network): {avg_overhead_ms:.0f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Measure matmul function execution time in isolation",
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/function/matmul",
        help="Function URL (default: through proxy at 8000)",
    )
    parser.add_argument(
        "--gateway-direct", action="store_true",
        help="Measure through OpenFaaS gateway directly (port 8080), "
             "bypassing the state tracker proxy",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of measurement requests (default: 20)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=2.0,
        help="Seconds between requests to prevent overlap (default: 2)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup requests to discard (default: 3)",
    )
    args = parser.parse_args()

    if args.gateway_direct:
        url = "http://127.0.0.1:8080/function/matmul"
    else:
        url = args.url

    print(f"Target URL: {url}")
    print()
    measure(url, args.n, args.cooldown, args.warmup)


if __name__ == "__main__":
    main()

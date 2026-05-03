#!/usr/bin/env python3
"""POST to OpenFaaS matmul on a fixed interval until Ctrl+C (stdlib only)."""

import argparse
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/function/matmul"
DEFAULT_INTERVAL = 0.1


def main() -> None:
    p = argparse.ArgumentParser(
        description="Send POST requests to matmul every N seconds until interrupted."
    )
    p.add_argument("--url", default=DEFAULT_URL, help="OpenFaaS function URL")
    p.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        metavar="SEC",
        help=f"Seconds between requests (default {DEFAULT_INTERVAL})",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default 120)",
    )
    args = p.parse_args()

    payload = b"{}"
    n_ok = 0
    n_err = 0

    print(f"POST every {args.interval}s -> {args.url}")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            req = urllib.request.Request(
                args.url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    n_ok += 1
                    if n_ok == 1 or n_ok % 20 == 0:
                        print(f"ok count={n_ok} (last HTTP {resp.status})")
            except urllib.error.HTTPError as e:
                n_err += 1
                print(f"HTTP {e.code}: {e.reason!r}")
            except urllib.error.URLError as e:
                n_err += 1
                print(f"URL error: {e.reason!r}")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nStopped. ok={n_ok} errors={n_err}")


if __name__ == "__main__":
    main()

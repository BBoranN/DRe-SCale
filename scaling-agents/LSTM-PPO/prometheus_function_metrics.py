#!/usr/bin/env python3
"""
Query Prometheus for OpenFaaS / Kubernetes function metrics used in this project.

  active  — in-flight invocations: sum(started) - sum(invocation_total)
  warm    — ready replicas: kube_deployment_status_replicas_ready

Examples:
  python prometheus_function_metrics.py --what active
  python prometheus_function_metrics.py --what warm --deployment matmul
  python prometheus_function_metrics.py --what both --json
  python prometheus_function_metrics.py --what all \\
      --prometheus-url http://127.0.0.1:9090 \\
      --function-name matmul.openfaas-fn --namespace openfaas-fn
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def build_query_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/api/v1/query"):
        return base
    return f"{base}/api/v1/query"


def prometheus_instant(query_url: str, query: str, timeout: float) -> float:
    params = urllib.parse.urlencode({"query": query})
    url = f"{query_url}?{params}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Prometheus HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Prometheus request failed: {e}") from e

    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus error response: {payload}")

    results = payload.get("data", {}).get("result") or []
    if not results:
        return 0.0
    return float(results[0]["value"][1])


def promql_active(function_name: str) -> str:
    return (
        f'sum(gateway_function_invocation_started{{function_name="{function_name}"}}) '
        f'- sum(gateway_function_invocation_total{{function_name="{function_name}"}})'
    )


def promql_warm(deployment: str, namespace: str | None) -> str:
    if namespace:
        return (
            "kube_deployment_status_replicas_ready{"
            f'deployment="{deployment}",namespace="{namespace}"}}'
        )
    return f'kube_deployment_status_replicas_ready{{deployment="{deployment}"}}'


def fetch_metrics(
    *,
    query_url: str,
    what: str,
    function_name: str,
    deployment: str,
    namespace: str | None,
    timeout: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if what in ("active", "both", "all"):
        q = promql_active(function_name)
        out["active_invocations"] = max(0.0, prometheus_instant(query_url, q, timeout))
    if what in ("warm", "both", "all"):
        q = promql_warm(deployment, namespace)
        out["warm_replicas"] = prometheus_instant(query_url, q, timeout)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch active invocation count and/or warm replica count from Prometheus.",
    )
    p.add_argument(
        "--what",
        choices=("active", "warm", "both", "all"),
        default="both",
        help="Metric to fetch: active=in-flight requests, warm=ready replicas, both|all=both (default: both).",
    )
    p.add_argument(
        "--prometheus-url",
        default="http://127.0.0.1:9090",
        help="Prometheus base URL (default: http://127.0.0.1:9090). /api/v1/query is appended if needed.",
    )
    p.add_argument(
        "--function-name",
        default="matmul.openfaas-fn",
        help='OpenFaaS function_name label value, e.g. matmul.openfaas-fn (default: matmul.openfaas-fn).',
    )
    p.add_argument(
        "--deployment",
        default="matmul",
        help="Kubernetes deployment name for warm replica query (default: matmul).",
    )
    p.add_argument(
        "--namespace",
        default=None,
        help='Optional Kubernetes namespace label for kube_deployment_status_replicas_ready, e.g. openfaas-fn.',
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds (default: 15).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print a single JSON object to stdout.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    query_url = build_query_url(args.prometheus_url)
    try:
        data = fetch_metrics(
            query_url=query_url,
            what=args.what,
            function_name=args.function_name,
            deployment=args.deployment,
            namespace=args.namespace,
            timeout=args.timeout,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(data))
        return 0

    for key, val in data.items():
        if key == "warm_replicas":
            print(f"{key}: {int(val)}")
        else:
            print(f"{key}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

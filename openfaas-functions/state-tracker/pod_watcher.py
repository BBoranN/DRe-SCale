"""
Kubernetes Pod Watcher
======================

Maintains a real-time cache of pod states using the K8s Watch API.

Two ways to connect to the K8s API:
  1. In-cluster:  when deployed as a pod (uses service account token)
  2. Local:       when running on your laptop (uses ~/.kube/config)

The watcher classifies every pod as either:
  - READY:     Running + all containers ready → can serve requests
  - NOT_READY: exists but not ready yet       → still initializing
"""

import logging
import threading
import time
from dataclasses import dataclass

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

logger = logging.getLogger("pod-watcher")


@dataclass
class PodInfo:
    """Cached state of a single pod."""
    name: str
    phase: str       # Pending, Running, Succeeded, Failed, Unknown
    ready: bool      # True if all containers pass readiness check
    created_at: float
    ready_at: float  # 0 if not yet ready


class PodWatcher:
    """
    Watches pods for one OpenFaaS function. Call get_counts() from
    any thread to get the current pod counts — it reads from cache,
    no API call.

    Usage:
        watcher = PodWatcher("matmul", "openfaas-fn")
        watcher.start()
        counts = watcher.get_counts()
        # {"ready": 3, "not_ready": 1, "total": 4}
    """

    def __init__(self, function_name: str, namespace: str = "openfaas-fn"):
        self.function_name = function_name
        self.namespace = namespace

        self._pods: dict[str, PodInfo] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread = None
        self._stop_event = threading.Event()

        # Try in-cluster config first, fall back to local kubeconfig
        try:
            config.load_incluster_config()
            logger.info("Using in-cluster K8s config")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("Using local kubeconfig (~/.kube/config)")
            except config.ConfigException as e:
                logger.error(
                    f"Cannot load K8s config: {e}\n"
                    "Make sure kubectl works on this machine."
                )
                raise

        self._v1 = client.CoreV1Api()

        # OpenFaaS labels function pods with faas_function=<name>
        self._label_selector = f"faas_function={function_name}"

    def start(self):
        """Populate cache with current pods, then start watching."""
        self._initial_list()

        self._thread = threading.Thread(
            target=self._watch_loop,
            name="pod-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_counts(self) -> dict:
        """
        Returns current counts. Called on every request — no API call,
        just reads from the in-memory cache.
        """
        with self._lock:
            ready = sum(1 for p in self._pods.values() if p.ready)
            not_ready = sum(1 for p in self._pods.values() if not p.ready)
            return {
                "ready": ready,
                "not_ready": not_ready,
                "total": ready + not_ready,
            }

    def get_pod_details(self) -> list[dict]:
        """Detailed pod info for debugging."""
        with self._lock:
            return [
                {
                    "name": p.name,
                    "phase": p.phase,
                    "ready": p.ready,
                    "age_seconds": round(time.time() - p.created_at, 1),
                    "time_to_ready": (
                        round(p.ready_at - p.created_at, 1)
                        if p.ready_at > 0
                        else None
                    ),
                }
                for p in self._pods.values()
            ]

    # ------------------------------------------------------------------
    # Initial list
    # ------------------------------------------------------------------

    def _initial_list(self):
        """
        LIST all existing pods and populate the cache.
        Runs once at startup and after watch reconnections.
        """
        try:
            pod_list = self._v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=self._label_selector,
            )

            with self._lock:
                self._pods.clear()
                for pod in pod_list.items:
                    info = self._parse_pod(pod)
                    if info:
                        self._pods[info.name] = info

            counts = self.get_counts()
            logger.info(
                f"Pod cache initialized: {counts['total']} pods "
                f"({counts['ready']} ready, {counts['not_ready']} not ready)"
            )

        except ApiException as e:
            if e.status == 403:
                logger.error(
                    f"Permission denied listing pods in {self.namespace}. "
                    "Make sure the RBAC rules are applied. See k8s-manifests.yaml."
                )
            else:
                logger.error(f"K8s API error listing pods: {e}")
        except Exception as e:
            logger.error(f"Failed to list pods: {e}")

    # ------------------------------------------------------------------
    # Watch loop
    # ------------------------------------------------------------------

    def _watch_loop(self):
        """
        Continuously watch pod events. Reconnects automatically
        when the watch stream closes (K8s does this periodically).
        """
        while not self._stop_event.is_set():
            try:
                w = watch.Watch()
                stream = w.stream(
                    self._v1.list_namespaced_pod,
                    namespace=self.namespace,
                    label_selector=self._label_selector,
                    timeout_seconds=300,
                )

                for event in stream:
                    if self._stop_event.is_set():
                        w.stop()
                        return

                    event_type = event["type"]
                    pod_obj = event["object"]
                    self._handle_event(event_type, pod_obj)

            except ApiException as e:
                if self._stop_event.is_set():
                    return
                logger.warning(f"Watch API error: {e.status}. Reconnecting in 2s...")
                time.sleep(2)
                self._initial_list()

            except Exception as e:
                if self._stop_event.is_set():
                    return
                logger.warning(f"Watch connection lost ({e}). Reconnecting in 2s...")
                time.sleep(2)
                self._initial_list()

    def _handle_event(self, event_type: str, pod_obj):
        """Process a single pod event."""
        pod_name = getattr(pod_obj.metadata, "name", None)
        if not pod_name:
            return

        info = self._parse_pod(pod_obj)

        with self._lock:
            if event_type == "DELETED" or info is None:
                # info is None when pod is Terminating or in a terminal phase.
                # Either way, remove it from our active cache.
                removed = self._pods.pop(pod_name, None)
                if removed:
                    reason = event_type if info is None else "DELETED"
                    logger.info(
                        f"Pod {pod_name} → REMOVED ({reason}, was ready={removed.ready})"
                    )

            elif event_type in ("ADDED", "MODIFIED"):
                old = self._pods.get(info.name)

                # Preserve original creation time
                if old:
                    info.created_at = old.created_at

                # Detect the moment a pod becomes ready
                if info.ready and (old is None or not old.ready):
                    info.ready_at = time.time()
                    init_time = info.ready_at - info.created_at
                    logger.info(
                        f"Pod {info.name} → READY (init took {init_time:.1f}s)"
                    )
                elif old and old.ready_at > 0:
                    info.ready_at = old.ready_at

                self._pods[info.name] = info

    # ------------------------------------------------------------------
    # Pod parsing
    # ------------------------------------------------------------------

    def _parse_pod(self, pod_obj) -> PodInfo | None:
        """
        Extract state from a V1Pod object.

        Ready means:
          - phase is "Running"
          - ALL containers have ready=True

        Not ready means:
          - phase is "Pending" (waiting for schedule/image pull)
          - phase is "Running" but containers not ready (still starting)

        We skip terminal phases (Succeeded/Failed) since those pods
        are about to be garbage collected.
        """
        try:
            name = pod_obj.metadata.name
            phase = pod_obj.status.phase or "Unknown"

            # Skip terminal pods
            if phase in ("Succeeded", "Failed"):
                return None

            # Skip TERMINATING pods. When K8s scales down, a pod gets
            # a deletionTimestamp but its phase stays "Running" until
            # the process actually exits. These pods are shutting down
            # and should NOT be counted as available capacity.
            # This is why you saw pods=37/24 — the 13 extra were
            # Terminating pods that still had phase=Running.
            if pod_obj.metadata.deletion_timestamp is not None:
                return None

            ready = False
            if phase == "Running" and pod_obj.status.container_statuses:
                ready = all(
                    cs.ready for cs in pod_obj.status.container_statuses
                )

            created_at = time.time()
            if pod_obj.metadata.creation_timestamp:
                created_at = pod_obj.metadata.creation_timestamp.timestamp()

            return PodInfo(
                name=name,
                phase=phase,
                ready=ready,
                created_at=created_at,
                ready_at=0.0,
            )

        except Exception as e:
            logger.warning(f"Failed to parse pod {getattr(pod_obj.metadata, 'name', '?')}: {e}")
            return None
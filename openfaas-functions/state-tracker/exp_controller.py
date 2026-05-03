"""
Expiration Controller
=====================

Implements θ_exp from the paper. Periodically checks for idle pods
that have exceeded the expiration timeout and scales down.

Instead of tracking per-pod idle times (which would require knowing
which specific pod is serving which request), we track the aggregate:
  - How many pods are idle right now?
  - How long has each idle "slot" existed?

We use a simple FIFO model: when idle_on increases, we push the
current timestamp. When idle_on decreases (a request consumed an
idle pod), we pop the oldest entry. When the oldest entry exceeds
the timeout, we scale down by 1.
"""

import time
import math
import logging
import threading
import httpx

logger = logging.getLogger("expiration-controller")


class ExpirationController:
    def __init__(
        self,
        function_name: str,
        gateway_url: str,
        pod_watcher,            # your existing PodWatcher
        in_flight_counter,      # your existing InFlightCounter
        max_replicas: int,
        k_exp: float = 1000.0,
        check_interval: float = 1.0,
    ):
        self.function_name = function_name
        self.gateway_url = gateway_url
        self.pod_watcher = pod_watcher
        self.in_flight_counter = in_flight_counter
        self.max_replicas = max_replicas
        self.k_exp = k_exp
        self.check_interval = check_interval

        # θ_exp as learned by the agent. Updated via set_theta_exp().
        self._theta_exp: float = 1.0
        self._theta_lock = threading.Lock()

        # FIFO queue of timestamps: each entry represents one pod
        # becoming idle. Oldest entries expire first.
        self._idle_since: list[float] = []
        self._idle_lock = threading.Lock()
        self._last_idle_count: int = 0

        self._thread: threading.Thread = None
        self._stop_event = threading.Event()
        self._http = httpx.Client(timeout=10.0)

    @property
    def timeout_seconds(self) -> float:
        """Current expiration timeout derived from θ_exp."""
        with self._theta_lock:
            if self._theta_exp <= 0:
                return float("inf")
            return self.k_exp / self._theta_exp

    def set_theta_exp(self, theta_exp: float):
        """Called by the agent after each NSGD update."""
        with self._theta_lock:
            self._theta_exp = theta_exp
        logger.info(
            f"θ_exp updated to {theta_exp:.4f} "
            f"(timeout = {self.timeout_seconds:.1f}s)"
        )

    def start(self):
        self._thread = threading.Thread(
            target=self._reaper_loop,
            name="expiration-controller",
            daemon=True,
        )
        self._thread.start()
        logger.info("Expiration controller started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._http.close()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reaper_loop(self):
        while not self._stop_event.is_set():
            try:
                self._update_idle_tracking()
                self._expire_if_needed()
            except Exception as e:
                logger.error(f"Reaper error: {e}")

            self._stop_event.wait(self.check_interval)

    def _update_idle_tracking(self):
        """
        Maintain the FIFO of idle-since timestamps.

        We observe the current idle_on count and reconcile:
          - If idle_on increased: new pods became idle, push timestamps
          - If idle_on decreased: pods were consumed by requests, pop oldest
        """
        counts = self.pod_watcher.get_counts()
        in_flight = self.in_flight_counter.count
        ready = counts["ready"]

        current_idle = max(0, ready - min(in_flight, ready))

        with self._idle_lock:
            prev = self._last_idle_count
            now = time.time()

            if current_idle > prev:
                # More idle pods than before — push new entries
                for _ in range(current_idle - prev):
                    self._idle_since.append(now)

            elif current_idle < prev:
                # Fewer idle pods — requests consumed them.
                # Pop the OLDEST entries (they were used first).
                to_remove = min(prev - current_idle, len(self._idle_since))
                self._idle_since = self._idle_since[to_remove:]

            self._last_idle_count = current_idle

    def _expire_if_needed(self):
        """
        Check if the oldest idle pod has exceeded the timeout.
        If so, scale down by the number of expired pods.
        """
        timeout = self.timeout_seconds
        now = time.time()

        with self._idle_lock:
            # Count how many idle entries have expired
            expired = 0
            for ts in self._idle_since:
                if now - ts >= timeout:
                    expired += 1
                else:
                    break  # sorted by time, so no more can be expired

            if expired == 0:
                return

            # Remove expired entries from tracking
            self._idle_since = self._idle_since[expired:]
            self._last_idle_count -= expired

        # Compute desired replica count
        counts = self.pod_watcher.get_counts()
        current_total = counts["total"]
        desired = max(0, current_total - expired)

        logger.info(
            f"Expiring {expired} idle pods "
            f"(timeout={timeout:.1f}s, "
            f"current={current_total}, desired={desired})"
        )

        self._scale_to(desired)

    def _scale_to(self, replicas: int):
        """Call OpenFaaS scale endpoint."""
        replicas = max(0, min(replicas, self.max_replicas))
        try:
            resp = self._http.post(
                f"{self.gateway_url}/system/scale-function/{self.function_name}",
                json={
                    "service": self.function_name,
                    "replicas": replicas,
                },
            )
            if resp.status_code == 200:
                logger.info(f"Scaled {self.function_name} to {replicas} replicas")
            else:
                logger.warning(
                    f"Scale request returned {resp.status_code}: {resp.text}"
                )
        except Exception as e:
            logger.error(f"Failed to scale: {e}")
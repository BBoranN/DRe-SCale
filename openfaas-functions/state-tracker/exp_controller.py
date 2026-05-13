"""
Expiration Controller
=====================

Implements θ_exp from the paper. Periodically checks for idle pods
that have exceeded the expiration timeout and scales down.

We track aggregate idle counts using a FIFO of timestamps:
  - When idle_on increases: push a new timestamp
  - When idle_on decreases: pop the oldest entry
  - When the oldest entry exceeds the timeout: scale down by 1

K8s Termination Lag Handling
----------------------------
After we send a scale-down request, K8s takes some time to process it
and mark pods as terminating (which makes them disappear from
pod_watcher counts). During that lag window, the FIFO must NOT be
updated — otherwise we'd see total > expected, interpret it as new
idle pods becoming ready, and push fake timestamps that overwrite
real old ones.

The fix: after sending a scale-down, record the expected pod count
and timestamp. Skip FIFO updates until either:
  - pod_watcher's total drops to <= expected (K8s caught up), or
  - a safety timeout elapses (scale-down failed; resume tracking)
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
        pod_watcher,
        in_flight_counter,
        max_replicas: int,
        k_exp: float = 100.0,
        check_interval: float = 1.0,
        initial_theta_exp: float = 5.0,
        scale_lag_timeout: float = 30.0,
        scale_endpoint_url: str = "http://127.0.0.1:3000/scale",
    ):
        self.function_name = function_name
        self.gateway_url = gateway_url
        self.pod_watcher = pod_watcher
        self.in_flight_counter = in_flight_counter
        self.max_replicas = max_replicas
        self.k_exp = k_exp
        self.check_interval = check_interval
        self.scale_lag_timeout = scale_lag_timeout
        self.scale_endpoint_url = scale_endpoint_url

        # θ_exp as learned by the agent. Updated via set_theta_exp().
        self._theta_exp: float = initial_theta_exp
        self._theta_lock = threading.Lock()

        # FIFO queue of timestamps: each entry represents one pod
        # becoming idle. Oldest entries expire first.
        self._idle_since: list[float] = []
        self._idle_lock = threading.Lock()
        self._last_idle_count: int = 0

        # K8s termination lag tracking. While _expected_pod_count is set,
        # we skip FIFO updates because pod_watcher hasn't yet caught up
        # with the scale-down we just sent.
        self._expected_pod_count: int | None = None
        self._expected_set_at: float = 0.0
        self._expected_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Short timeout so a slow scale endpoint doesn't freeze the loop
        self._http = httpx.Client(timeout=3.0)

        logger.info(
            f"ExpirationController initialized: "
            f"k_exp={k_exp}, initial_theta_exp={initial_theta_exp}, "
            f"initial_timeout={k_exp / initial_theta_exp:.1f}s, "
            f"scale_lag_timeout={scale_lag_timeout}s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def timeout_seconds(self) -> float:
        """Current expiration timeout derived from θ_exp."""
        with self._theta_lock:
            if self._theta_exp <= 0:
                return float("inf")
            return self.k_exp * self._theta_exp

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

    def get_status(self) -> dict:
        """Snapshot of internal state for diagnostics."""
        with self._idle_lock:
            timestamps = list(self._idle_since)
            last_idle = self._last_idle_count
        with self._theta_lock:
            theta_exp = self._theta_exp
        with self._expected_lock:
            expected = self._expected_pod_count
            expected_age = (
                time.time() - self._expected_set_at
                if expected is not None else None
            )

        now = time.time()
        ages = [round(now - t, 1) for t in timestamps]
        timeout = self.timeout_seconds

        return {
            "thread_alive": self._thread.is_alive() if self._thread else False,
            "fifo_size": len(timestamps),
            "oldest_age_seconds": ages[0] if ages else None,
            "newest_age_seconds": ages[-1] if ages else None,
            "expired_in_fifo": sum(1 for a in ages if a >= timeout),
            "timeout_seconds": round(timeout, 1),
            "theta_exp": round(theta_exp, 4),
            "last_idle_count": last_idle,
            "waiting_for_k8s": expected is not None,
            "expected_pod_count": expected,
            "wait_age_seconds": (
                round(expected_age, 1) if expected_age is not None else None
            ),
            "ages_sample": ages[:20],
        }

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reaper_loop(self):
        loop_count = 0
        while not self._stop_event.is_set():
            try:
                self._update_idle_tracking()
                self._expire_if_needed()

                # Periodic heartbeat — every ~30 seconds
                loop_count += 1
                if loop_count % 30 == 0:
                    status = self.get_status()
                    logger.info(
                        f"Reaper alive: fifo={status['fifo_size']}, "
                        f"oldest_age={status['oldest_age_seconds']}s, "
                        f"timeout={status['timeout_seconds']}s, "
                        f"θ_exp={status['theta_exp']}, "
                        f"waiting_k8s={status['waiting_for_k8s']}"
                    )
            except Exception as e:
                logger.error(f"Reaper error: {e}", exc_info=True)

            self._stop_event.wait(self.check_interval)

    def _update_idle_tracking(self):
        """
        Maintain the FIFO of idle-since timestamps.

        Skip updates while waiting for K8s to process a recent scale-down.
        """
        counts = self.pod_watcher.get_counts()
        in_flight = self.in_flight_counter.count
        ready = counts["ready"]
        total = counts["total"]

        # ---- K8s lag check ----
        with self._expected_lock:
            if self._expected_pod_count is not None:
                elapsed = time.time() - self._expected_set_at

                if total <= self._expected_pod_count:
                    # K8s caught up — resume normal operation
                    logger.debug(
                        f"K8s caught up: total={total} <= "
                        f"expected={self._expected_pod_count} "
                        f"(took {elapsed:.1f}s)"
                    )
                    self._expected_pod_count = None
                elif elapsed > self.scale_lag_timeout:
                    # Scale-down apparently failed — stop waiting
                    logger.warning(
                        f"Scale-down lag exceeded {self.scale_lag_timeout}s "
                        f"(total={total}, expected={self._expected_pod_count}). "
                        f"Resuming tracking. Scale endpoint may be broken."
                    )
                    self._expected_pod_count = None
                else:
                    # Still waiting — skip FIFO update this cycle
                    return

        # ---- Normal FIFO maintenance ----
        current_idle = max(0, ready - min(in_flight, ready))

        with self._idle_lock:
            prev = self._last_idle_count
            now = time.time()

            if current_idle > prev:
                for _ in range(current_idle - prev):
                    self._idle_since.append(now)
            elif current_idle < prev:
                to_remove = min(prev - current_idle, len(self._idle_since))
                if to_remove > 0:
                    self._idle_since = self._idle_since[:-to_remove]

            self._last_idle_count = current_idle

    def _expire_if_needed(self):
        """
        Check if the oldest idle pod has exceeded the timeout.
        If so, scale down by the number of expired pods.
        """
        # Don't fire new expirations while still waiting for the previous one.
        with self._expected_lock:
            if self._expected_pod_count is not None:
                return

        timeout = self.timeout_seconds
        now = time.time()

        with self._idle_lock:
            expired = 0
            for ts in self._idle_since:
                if now - ts >= timeout:
                    expired += 1
                else:
                    break

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

        # Mark that we're now waiting for K8s to catch up.
        # Set BEFORE the HTTP call so even if it succeeds quickly,
        # the next _update_idle_tracking will see we're waiting.
        with self._expected_lock:
            self._expected_pod_count = desired
            self._expected_set_at = time.time()

        # Send scale-down request
        success = self._scale_to(desired)

        if not success:
            # Scale call failed outright — clear the wait flag immediately
            # so we don't stall for scale_lag_timeout seconds.
            with self._expected_lock:
                self._expected_pod_count = None
            logger.warning(
                "Scale call failed. Cleared wait flag — "
                "FIFO will resume on next iteration."
            )

    def _scale_to(self, replicas: int) -> bool:
        """
        Call the scale endpoint. Returns True if request succeeded
        with HTTP 200, False otherwise.
        """
        replicas = max(0, min(replicas, self.max_replicas))
        try:
            resp = self._http.get(
                self.scale_endpoint_url,
                params={"replicas": replicas},
            )
            if resp.status_code == 200:
                logger.info(
                    f"Scale request accepted: {self.function_name} → "
                    f"{replicas} replicas"
                )
                return True
            else:
                logger.warning(
                    f"Scale request returned {resp.status_code}: {resp.text}"
                )
                return False
        except httpx.TimeoutException:
            logger.error("Scale request timed out (3s)")
            return False
        except Exception as e:
            logger.error(f"Failed to scale: {e}")
            return False
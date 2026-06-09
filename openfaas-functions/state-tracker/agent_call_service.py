"""
Agent Call Service
==================

Thin async client for sending state observations to the NSGD agent.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger("agent-call-service")


class AgentCallService:
    def __init__(self, agent_url: str, timeout_seconds: float = 2.0, enabled: bool = True):
        self.agent_url = agent_url.rstrip("/")
        self.enabled = enabled
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2.0)),
        )

    async def close(self):
        await self._client.aclose()

    async def get_theta(self) -> tuple[dict[str, Any] | None, str | None]:
        """
        Read the current policy without advancing the NSGD algorithm.

        Returns (response_json, error). The middleware should fail open when
        error is not None so function traffic still reaches OpenFaaS.
        """
        if not self.enabled:
            return None, "agent disabled"

        try:
            response = await self._client.get(f"{self.agent_url}/theta")
            response.raise_for_status()
            return response.json(), None
        except Exception as exc:
            logger.warning("Agent theta fetch failed: %s", exc)
            return None, str(exc)

    async def process_event(
        self,
        state,
        has_rejected_job: bool,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """
        Send one completed request observation to the agent.
        """
        if not self.enabled:
            return None, "agent disabled"

        payload = {
            "state": [
                state.cold,
                state.idle_on,
                state.busy,
                state.init_free + state.init_reserved,
                state.init_reserved,
            ],
            "has_rejected_job": has_rejected_job,
            "timestamp": state.timestamp,
        }

        try:
            response = await self._client.post(f"{self.agent_url}/event", json=payload)
            response.raise_for_status()
            return response.json(), None
        except Exception as exc:
            logger.warning("Agent call failed: %s", exc)
            return None, str(exc)

    async def call_agent(self, state) -> tuple[dict[str, Any] | None, str | None]:
        """
        Backward-compatible wrapper for older callers.

        Prefer get_theta() before forwarding and process_event() after the
        request outcome is known.
        """
        return await self.process_event(
            state,
            has_rejected_job=state.saturated_at_arrival,
        )

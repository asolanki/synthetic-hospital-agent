"""HTTP client for the simulator's /env (reset and step) and /score endpoints.

Only the corrected, server-side-filtered /env route is used for agent episodes.
The original eval/agents harness leaks the label through the problem list and is
never called from this project.
"""

from __future__ import annotations

from typing import Any

import httpx

from common import config


class SimulatorError(RuntimeError):
    pass


class Simulator:
    """Harness-side client. Holds the scorer token; the policy never does."""

    def __init__(self, base_url: str = config.BASE_URL, token: str = config.SCORER_TOKEN,
                 timeout: float = 120.0, max_connections: int = 64):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout,
            headers={"X-Scorer-Token": token},
            limits=httpx.Limits(max_connections=max_connections),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "Simulator":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            r = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise SimulatorError(f"{method} {path} -> {type(exc).__name__}: {exc}") from exc
        if r.status_code >= 400:
            raise SimulatorError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    # -- environment ---------------------------------------------------------
    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def reset(self, gt_id: int, budget: int = config.DEFAULT_BUDGET) -> dict:
        return await self._request("POST", "/env/reset", json={"gt_id": gt_id, "budget": budget})

    async def step(self, episode_id: str, name: str, arguments: dict | None) -> dict:
        return await self._request("POST", "/env/step", json={
            "episode_id": episode_id, "name": name, "arguments": arguments or {}})

    async def close(self, episode_id: str) -> dict:
        return await self._request("POST", "/env/close", json={"episode_id": episode_id})

    async def oracle(self, episode_id: str) -> dict:
        return await self._request("GET", f"/env/oracle/{episode_id}")

    # -- scoring -------------------------------------------------------------
    async def score(self, gt_id: int, prediction: Any, task: str | None = config.TASK) -> dict:
        return await self._request("POST", "/score", json={
            "gt_id": gt_id, "prediction": prediction, "task": task})

    async def score_batch(self, items: list[tuple[int, dict]], task: str | None = config.TASK) -> list[dict]:
        """Score (gt_id, prediction) pairs. Returns one metrics result per item, in order."""
        out: list[dict] = []
        for i in range(0, len(items), 256):
            chunk = items[i:i + 256]
            data = await self._request("POST", "/score/batch", json={"items": [
                {"gt_id": g, "prediction": p, "task": task} for g, p in chunk]})
            for item in data["items"]:
                if not item["ok"]:
                    raise SimulatorError(f"scoring gt_id={item['gt_id']} failed: {item['error']}")
                out.append(item["result"])
        return out

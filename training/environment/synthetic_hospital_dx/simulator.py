"""Minimal client for the simulator's /env endpoints, shared by the task and the toolset.

The scorer token stays on the harness side: the task (setup, scoring) and the tool
server run as host processes, never inside the model's context.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_BASE_URL = "http://localhost:58000"
DEFAULT_TOKEN = "dev-scorer-token-change-in-production"


def base_url() -> str:
    return os.environ.get("SH_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def headers() -> dict[str, str]:
    return {"X-Scorer-Token": os.environ.get("EPIC_SIM_SCORER_TOKEN", DEFAULT_TOKEN)}


async def post(path: str, body: dict, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(base_url=base_url(), headers=headers(), timeout=timeout) as client:
        response = await client.post(path, json=body)
        response.raise_for_status()
        return response.json()


def post_sync(path: str, body: dict, timeout: float = 120.0) -> dict:
    with httpx.Client(base_url=base_url(), headers=headers(), timeout=timeout) as client:
        response = client.post(path, json=body)
        response.raise_for_status()
        return response.json()

"""Stop (never destroy) this Vast instance at an explicit UTC deadline.

Run under Supervisor so the deadline survives SSH disconnects. Restarting the
process does not extend the absolute deadline. Uses only the instance-scoped key
injected by Vast; no account-wide API key is stored in this project.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True, help="ISO-8601 timestamp with timezone")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline must include a timezone")
    instance = os.environ["CONTAINER_ID"]
    key = os.environ["CONTAINER_API_KEY"]
    print(f"Instance {instance}: stop deadline {deadline.astimezone(timezone.utc).isoformat()}", flush=True)
    while (remaining := deadline.timestamp() - time.time()) > 0:
        time.sleep(min(remaining, 60))
    print("Rental deadline reached; requesting stop (disk retained).", flush=True)
    request = Request(
        f"https://console.vast.ai/api/v0/instances/{instance}/",
        data=b'{"state":"stopped"}',
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except (URLError, TimeoutError, ValueError):
        raise RuntimeError("Vast stop request failed; no credentials logged") from None
    if not isinstance(result, dict) or result.get("success") is not True:
        raise RuntimeError("Vast rejected the stop request; no credentials logged")
    print("Vast stop request accepted.", flush=True)


if __name__ == "__main__":
    main()

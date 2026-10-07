"""OpenAI-compatible proxy in front of DeepInfra that turns Gemma 3's native pythonic tool calls
into structured `tool_calls` (LOG.md 2026-10-07, Gemma plan A).

Gemma 3 has no tool-call tokens. Under a realistic agent prompt it writes a sentence and then its
own call syntax, e.g. ```tool_code [open_chart(patient_id=1690)] ```, and DeepInfra returns that as
plain text, so an agent harness never sees a tool call. This proxy does what vLLM's "pythonic"
tool-call parser does for self-hosted Gemma:

- only assistant messages without structured `tool_calls` are touched;
- calls must name a tool offered in that request and take literal arguments (JSON true/false/null
  are accepted); positional arguments follow the tool schema's parameter order;
- anything that does not parse cleanly is left as text.

Prompts, tools, history and sampling pass through unchanged. Upstream 429/5xx are retried with
backoff. Every message the proxy inspects is appended to a JSONL audit log.

    set -a; . ./.env; set +a
    uv run python training/gemma_toolcall_proxy.py --port 58100 --log runs/gemma-proxy.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

UPSTREAM = "https://api.deepinfra.com/v1/openai/chat/completions"
CODE_BLOCK = re.compile(r"```([A-Za-z_]*)[ \t]*\n?(.*?)```", re.S)
CODE_LANGS = {"tool_code", "python", ""}
JSON_NAMES = {"true": True, "false": False, "null": None}


class _JsonNames(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in JSON_NAMES:
            return ast.copy_location(ast.Constant(JSON_NAMES[node.id]), node)
        return node


def _tool_params(tools: list[dict]) -> dict[str, list[str]]:
    out = {}
    for t in tools or []:
        fn = t.get("function", t)
        out[fn["name"]] = list(((fn.get("parameters") or {}).get("properties") or {}))
    return out


def _calls_from_expr(expr: str, params: dict[str, list[str]]) -> list[dict] | None:
    """Parse `[f(a=1), g(2)]` or `f(a=1)`; None unless every call is a known tool with literal args."""
    try:
        tree = _JsonNames().visit(ast.parse(expr.strip(), mode="eval")).body
    except SyntaxError:
        return None
    nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    calls = []
    for node in nodes:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in params):
            return None
        order = params[node.func.id]
        if len(node.args) > len(order):
            return None
        try:
            args = {order[i]: ast.literal_eval(a) for i, a in enumerate(node.args)}
            for kw in node.keywords:
                if kw.arg is None:
                    return None
                args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError, SyntaxError):
            return None
        calls.append({"name": node.func.id, "arguments": args})
    return calls or None


def _bracket_lists(text: str, names: list[str]):
    """Yield (start, end) of balanced `[name(...)...]` spans for known tool names."""
    if not names:
        return
    start_re = re.compile(r"\[\s*(?:" + "|".join(map(re.escape, names)) + r")\s*\(")
    for m in start_re.finditer(text):
        depth, quote, esc = 0, None, False
        for i in range(m.start(), len(text)):
            c = text[i]
            if quote:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    quote = None
                continue
            if c in "\"'":
                quote = c
            elif c in "[(":
                depth += 1
            elif c in "])":
                depth -= 1
                if depth == 0:
                    yield m.start(), i + 1
                    break


def extract_calls(content: str, tools: list[dict]) -> tuple[list[dict], str] | None:
    """Return (calls, prose before the first call) if the message contains tool calls, else None.

    Calls come from ```tool_code / ```python / untagged blocks whose code parses as calls to the
    offered tools; other blocks are ignored. With no such block, bare `[tool(...)]` lists count.
    """
    params = _tool_params(tools)
    found = []  # (start offset, calls)
    for b in CODE_BLOCK.finditer(content):
        if b.group(1).lower() in CODE_LANGS and (parsed := _calls_from_expr(b.group(2), params)):
            found.append((b.start(), parsed))
    if not found:
        for s, e in _bracket_lists(content, list(params)):
            if parsed := _calls_from_expr(content[s:e], params):
                found.append((s, parsed))
    if not found:
        return None
    calls = [c for _, parsed in found for c in parsed]
    return calls, content[: found[0][0]].strip()


class Proxy(BaseHTTPRequestHandler):
    client: httpx.Client
    key: str
    log_path: Path
    lock = threading.Lock()

    def log_message(self, *a):  # quiet stdlib access log
        pass

    def _audit(self, record: dict) -> None:
        with self.lock, self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _upstream(self, body: dict) -> httpx.Response:
        delay = 2.0
        for _ in range(12):
            r = self.client.post(UPSTREAM, json=body, headers={"Authorization": f"Bearer {self.key}"})
            if r.status_code != 429 and r.status_code < 500:
                return r
            time.sleep(min(delay, 60.0))
            delay *= 1.6
        return r

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        want_stream = bool(body.pop("stream", False))
        stream_usage = bool((body.pop("stream_options", None) or {}).get("include_usage"))
        r = self._upstream(body)
        if r.status_code != 200:
            self.send_response(r.status_code)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(r.content)
            return
        data = r.json()
        for choice in data.get("choices", []):
            msg = choice.get("message") or {}
            record = {"t": time.time(), "stream": want_stream, "finish": choice.get("finish_reason"),
                      "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
                      "content_chars": len(msg.get("content") or ""),
                      "upstream_tool_calls": msg.get("tool_calls") or [], "converted": False, "names": []}
            if not msg.get("tool_calls") and msg.get("content"):
                got = extract_calls(msg["content"], body.get("tools") or [])
                if got:
                    calls, prose = got
                    msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                         for c in calls]
                    msg["content"] = prose or None
                    choice["finish_reason"] = "tool_calls"
                    record.update(converted=True, names=[c["name"] for c in calls])
                else:
                    record["snippet"] = msg["content"][-300:]
            self._audit(record)
        if want_stream:
            self._send_sse(data, stream_usage)
            return
        out = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _send_sse(self, data: dict, include_usage: bool) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        base = {k: data.get(k) for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
        for choice in data.get("choices", []):
            msg = choice.get("message") or {}
            delta = {"role": "assistant", "content": msg.get("content")}
            if msg.get("tool_calls"):
                delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(msg["tool_calls"])]
            chunk = base | {"choices": [{"index": choice.get("index", 0), "delta": delta,
                                         "finish_reason": choice.get("finish_reason")}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if include_usage and data.get("usage"):
            self.wfile.write(f"data: {json.dumps(base | {'choices': [], 'usage': data['usage']})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=58100)
    ap.add_argument("--log", type=Path, default=Path("runs/gemma-proxy.jsonl"))
    args = ap.parse_args()
    Proxy.key = os.environ["DEEPINFRA_API_KEY"]
    Proxy.client = httpx.Client(timeout=300)
    Proxy.log_path = args.log
    args.log.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Proxy)
    print(f"gemma tool-call proxy on 127.0.0.1:{args.port} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

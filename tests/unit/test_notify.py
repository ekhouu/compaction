"""Tests for evaluation/notify.py Discord webhook notifications.

Verifies message formatting and send logic WITHOUT actually hitting Discord.
Uses a mock HTTP handler to capture payloads.

Run standalone:
    python tests/unit/test_notify.py
"""

from __future__ import annotations

import http.server
import json
import os
import pathlib
import sys
import threading
import traceback
from typing import Any

_HERE = pathlib.Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib.util

# Import notify.py directly to avoid pulling in torch via evaluation/__init__.py
_notify_path = _REPO_ROOT / "evaluation" / "notify.py"
_spec = importlib.util.spec_from_file_location("evaluation.notify", _notify_path)
_notify_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_notify_mod)

send_message = _notify_mod.send_message
notify_eval_start = _notify_mod.notify_eval_start
notify_method_done = _notify_mod.notify_method_done
notify_eval_done = _notify_mod.notify_eval_done


# ── Tiny HTTP server that captures POST bodies ──────────────────────

_captured_payloads: list[dict[str, Any]] = []


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _captured_payloads.append(json.loads(body))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args):
        pass  # silence request logs


def _start_mock_server() -> tuple[http.server.HTTPServer, str]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}/webhook"


# ── Tests ────────────────────────────────────────────────────────────

def test_noop_when_no_env_var():
    """send_message should return False when DISCORD_WEBHOOK_URL is unset."""
    old = os.environ.pop("DISCORD_WEBHOOK_URL", None)
    try:
        assert send_message("hello") is False
    finally:
        if old is not None:
            os.environ["DISCORD_WEBHOOK_URL"] = old


def test_send_message_basic():
    """send_message should POST correct JSON to the webhook."""
    server, url = _start_mock_server()
    _captured_payloads.clear()
    os.environ["DISCORD_WEBHOOK_URL"] = url
    try:
        result = send_message("test content")
        assert result is True
        assert len(_captured_payloads) == 1
        assert _captured_payloads[0]["content"] == "test content"
    finally:
        del os.environ["DISCORD_WEBHOOK_URL"]
        server.shutdown()


def test_send_message_with_embeds():
    """send_message should include embeds in the payload."""
    server, url = _start_mock_server()
    _captured_payloads.clear()
    os.environ["DISCORD_WEBHOOK_URL"] = url
    try:
        embeds = [{"title": "Test", "color": 123}]
        result = send_message("", embeds=embeds)
        assert result is True
        assert _captured_payloads[0]["embeds"] == embeds
    finally:
        del os.environ["DISCORD_WEBHOOK_URL"]
        server.shutdown()


def test_notify_eval_start_format():
    """notify_eval_start should produce a well-formed embed with all fields."""
    server, url = _start_mock_server()
    _captured_payloads.clear()
    os.environ["DISCORD_WEBHOOK_URL"] = url
    try:
        result = notify_eval_start(
            methods=["original", "AM", "TQ_int8"],
            dataset="quality",
            model="Qwen/Qwen3-4B",
            target_size=0.1,
            n_articles=10,
            experiment_name="cp_test",
        )
        assert result is True
        embed = _captured_payloads[0]["embeds"][0]
        assert embed["title"] == "Evaluation Started"
        field_names = [f["name"] for f in embed["fields"]]
        assert "Model" in field_names
        assert "Dataset" in field_names
        assert "Methods" in field_names
        assert "Experiment" in field_names
    finally:
        del os.environ["DISCORD_WEBHOOK_URL"]
        server.shutdown()


def test_notify_method_done_format():
    """notify_method_done should include method name and progress."""
    server, url = _start_mock_server()
    _captured_payloads.clear()
    os.environ["DISCORD_WEBHOOK_URL"] = url
    try:
        result = notify_method_done(
            method="AM",
            article_idx=2,
            n_articles=10,
            accuracy=0.85,
            perplexity=4.2,
            elapsed_sec=42.0,
        )
        assert result is True
        embed = _captured_payloads[0]["embeds"][0]
        field_map = {f["name"]: f["value"] for f in embed["fields"]}
        assert field_map["Method"] == "`AM`"
        assert field_map["Progress"] == "3/10"
        assert "85" in field_map["Accuracy"]
        assert "4.2" in field_map["Perplexity"]
    finally:
        del os.environ["DISCORD_WEBHOOK_URL"]
        server.shutdown()


def test_notify_eval_done_format():
    """notify_eval_done should summarise per-method stats."""
    server, url = _start_mock_server()
    _captured_payloads.clear()
    os.environ["DISCORD_WEBHOOK_URL"] = url
    try:
        stats = {
            "original": {"accuracy": 0.92, "mean_perplexity": 3.1},
            "AM": {"accuracy": 0.88, "mean_perplexity": 3.5},
        }
        result = notify_eval_done(stats, results_path="logs/test.json", total_elapsed_sec=300)
        assert result is True
        embed = _captured_payloads[0]["embeds"][0]
        assert embed["title"] == "Evaluation Complete"
        results_field = next(f for f in embed["fields"] if f["name"] == "Results")
        assert "original" in results_field["value"]
        assert "AM" in results_field["value"]
    finally:
        del os.environ["DISCORD_WEBHOOK_URL"]
        server.shutdown()


# ── Runner ───────────────────────────────────────────────────────────

def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures: list[str] = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            traceback.print_exc()
            failures.append(name)
    if failures:
        print(f"\n{len(failures)}/{len(tests)} FAILED: {failures}")
        return 1
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())

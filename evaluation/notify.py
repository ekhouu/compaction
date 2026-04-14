# evaluation/notify.py
"""
Discord webhook notifications for evaluation progress.

Reads the webhook URL from the ``DISCORD_WEBHOOK_URL`` environment variable.
All functions silently no-op when the env var is unset, so callers never need
to guard on availability.

Usage in the evaluator::

    from evaluation.notify import notify_eval_start, notify_method_done, notify_eval_done

    notify_eval_start(methods=["original","AM","TQ_int8","TQ_int8_AM"],
                      dataset="quality", model="Qwen/Qwen3-4B",
                      target_size=0.1, n_articles=10)

    notify_method_done(method="AM", article_idx=3, accuracy=0.85,
                       perplexity=4.2, elapsed_sec=42.1)

    notify_eval_done(overall_stats={...}, results_path="logs/foo.json")
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _webhook_url() -> Optional[str]:
    return os.environ.get("DISCORD_WEBHOOK_URL")


def _post(payload: dict) -> bool:
    """POST a JSON payload to the webhook.  Returns True on success."""
    url = _webhook_url()
    if not url:
        return False
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        # Never let a notification failure crash the eval
        return False


def send_message(content: str, embeds: Optional[List[dict]] = None) -> bool:
    """Send a message to the configured Discord webhook.

    Parameters
    ----------
    content : str
        Plain text content (appears above embeds).
    embeds : list of dict, optional
        Discord embed objects for richer formatting.

    Returns
    -------
    bool
        True if the message was sent successfully, False otherwise
        (including when no webhook URL is configured).
    """
    payload: dict[str, Any] = {"content": content}
    if embeds:
        payload["embeds"] = embeds
    return _post(payload)


# ── Convenience helpers for eval lifecycle ─────────────────────────


def notify_eval_start(
    methods: List[str],
    dataset: str,
    model: str,
    target_size: float,
    n_articles: int,
    experiment_name: Optional[str] = None,
) -> bool:
    """Notify that an evaluation run has started."""
    embed = {
        "title": "Evaluation Started",
        "color": 3447003,  # blue
        "fields": [
            {"name": "Model", "value": f"`{model}`", "inline": True},
            {"name": "Dataset", "value": f"`{dataset}`", "inline": True},
            {"name": "Target Size", "value": f"`{target_size}`", "inline": True},
            {"name": "Articles", "value": str(n_articles), "inline": True},
            {"name": "Methods", "value": ", ".join(f"`{m}`" for m in methods)},
        ],
    }
    if experiment_name:
        embed["fields"].insert(0, {"name": "Experiment", "value": f"`{experiment_name}`"})
    return send_message("", embeds=[embed])


def notify_method_done(
    method: str,
    article_idx: int,
    n_articles: int,
    accuracy: Optional[float] = None,
    perplexity: Optional[float] = None,
    elapsed_sec: Optional[float] = None,
) -> bool:
    """Notify that a method finished processing one article."""
    fields = [
        {"name": "Method", "value": f"`{method}`", "inline": True},
        {"name": "Progress", "value": f"{article_idx + 1}/{n_articles}", "inline": True},
    ]
    if accuracy is not None:
        fields.append({"name": "Accuracy", "value": f"{accuracy:.1%}", "inline": True})
    if perplexity is not None:
        fields.append({"name": "Perplexity", "value": f"{perplexity:.2f}", "inline": True})
    if elapsed_sec is not None:
        fields.append({"name": "Time", "value": f"{elapsed_sec:.1f}s", "inline": True})

    embed = {
        "title": "Article Complete",
        "color": 15844367,  # gold
        "fields": fields,
    }
    return send_message("", embeds=[embed])


def notify_eval_done(
    overall_stats: Dict[str, Any],
    results_path: Optional[str] = None,
    total_elapsed_sec: Optional[float] = None,
) -> bool:
    """Notify that the full evaluation run has finished."""
    # Build a compact summary from overall_stats
    lines: list[str] = []
    for method, stats in overall_stats.items():
        if not isinstance(stats, dict):
            continue
        acc = stats.get("overall_accuracy")
        ppl = stats.get("overall_avg_perplexity") or stats.get("mean_perplexity")
        parts = [f"`{method}`"]
        if acc is not None:
            parts.append(f"acc={acc:.1%}")
        if ppl is not None:
            parts.append(f"ppl={ppl:.2f}")
        lines.append(" | ".join(parts))

    summary = "\n".join(lines) if lines else "_No per-method stats available_"

    fields = [{"name": "Results", "value": summary}]
    if results_path:
        fields.append({"name": "Log", "value": f"`{results_path}`"})
    if total_elapsed_sec is not None:
        mins = total_elapsed_sec / 60
        fields.append({"name": "Total Time", "value": f"{mins:.1f} min", "inline": True})

    embed = {
        "title": "Evaluation Complete",
        "color": 3066993,  # green
        "fields": fields,
    }
    return send_message("", embeds=[embed])

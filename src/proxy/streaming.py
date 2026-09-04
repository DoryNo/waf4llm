from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator


async def reassemble_stream_text(chunks: list[bytes]) -> str:
    """Extract concatenated text from SSE stream chunks (for output guard)."""
    texts: list[str] = []
    # Network chunks do not necessarily align with SSE lines/events. Reassemble
    # bytes first, then parse complete events so a split JSON payload is handled.
    stream = b"".join(chunks).decode("utf-8", errors="replace")
    for event in re.split(r"\r?\n\r?\n", stream):
        data_lines = [line[5:] for line in event.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        payload = "\n".join(data_lines).strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue

        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content is None:
                content = choice.get("text")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("text")
                )
    return "".join(texts)


async def passthrough_stream_with_capture(
    upstream_gen: AsyncGenerator[bytes, None],
) -> tuple[AsyncGenerator[bytes, None], list[bytes]]:
    """Wraps upstream generator to capture chunks while yielding them.

    Returns a new generator and a list that will be populated as chunks flow.
    Note: list is mutated in place; caller should read after generator exhausted.
    """
    captured: list[bytes] = []

    async def gen():
        async for chunk in upstream_gen:
            captured.append(chunk)
            yield chunk

    return gen(), captured

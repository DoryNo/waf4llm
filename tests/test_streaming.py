from __future__ import annotations

import pytest

from src.proxy.streaming import reassemble_stream_text


@pytest.mark.asyncio
async def test_reassemble_stream_handles_network_chunk_boundaries():
    stream = (
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo "}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    chunks = [stream[:17], stream[17:51], stream[51:97], stream[97:]]

    assert await reassemble_stream_text(chunks) == "Hello world"


@pytest.mark.asyncio
async def test_reassemble_stream_supports_legacy_completion_text():
    chunks = [
        b'data: {"choices":[{"text":"Hel"}]}\n\n',
        b'data: {"choices":[{"text":"lo"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    assert await reassemble_stream_text(chunks) == "Hello"


@pytest.mark.asyncio
async def test_reassemble_stream_preserves_utf8_split_between_bytes():
    event = 'data: {"choices":[{"delta":{"content":"Привет"}}]}\n\n'.encode()
    chunks = [event[:25], event[25:26], event[26:]]

    assert await reassemble_stream_text(chunks) == "Привет"

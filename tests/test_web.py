from __future__ import annotations

from contextlib import contextmanager
from http.client import HTTPConnection
import json
import threading
from typing import Iterator

from src.web import SESSION_COOKIE_NAME, WebTransport, create_web_server, render_web_markdown, split_renderable_markdown

from tests.test_cli_smoke import build_registry


@contextmanager
def run_test_server(transport: WebTransport | None = None) -> Iterator[tuple[str, int]]:
    server = create_web_server(
        transport=transport or WebTransport(build_registry()),
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        server_address = server.server_address
        host = server_address[0]
        port = server_address[1]
        resolved_host = host.decode("utf-8") if isinstance(host, bytes) else str(host)
        yield resolved_host, int(port)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _read_ndjson_response(response) -> list[dict[str, object]]:
    raw = response.read().decode("utf-8").strip().splitlines()
    return [json.loads(line) for line in raw if line.strip()]


def test_web_transport_handles_start_command_as_html() -> None:
    transport = WebTransport(build_registry())

    response = transport.handle_text("/start", session_id="browser-1")

    assert response.is_command is True
    assert response.render_mode == "html"
    assert "<b>Привет! Я бот поддержки SmartTherm.</b>" in response.text
    assert response.dialog_key == "web:browser-1"


def test_web_transport_routes_text_as_markdown_and_isolates_sessions() -> None:
    registry = build_registry()
    transport = WebTransport(registry)

    first = transport.handle_text("Привет из первой вкладки", session_id="browser-1")
    second = transport.handle_text("Привет из второй вкладки", session_id="browser-2")

    assert first.is_command is False
    assert first.render_mode == "markdown"
    assert "<p>single-response</p>" == first.text
    assert second.text == "<p>single-response</p>"

    first_context = registry.acquire("web:browser-1")
    second_context = registry.acquire("web:browser-2")
    try:
        assert [message.content for message in first_context.session.history] == [
            "Привет из первой вкладки",
            "single-response",
        ]
        assert [message.content for message in second_context.session.history] == [
            "Привет из второй вкладки",
            "single-response",
        ]
    finally:
        registry.release(first_context)
        registry.release(second_context)


def test_web_transport_streams_regular_text_and_final_markdown() -> None:
    transport = WebTransport(build_registry(), preview_interval_ms=10_000, preview_min_chars=4)

    events = [event.to_payload() for event in transport.stream_text("Привет", session_id="browser-1")]

    assert [event["event"] for event in events] == ["start", "token", "token", "preview", "final"]
    assert events[0]["render_mode"] == "markdown"
    assert events[1]["text"] == "от"
    assert events[2]["text"] == "вет"
    assert events[3]["render_mode"] == "markdown"
    assert events[3]["html"] == "<p>ответ</p>"
    assert events[4]["render_mode"] == "markdown"
    assert events[4]["text"] == "<p>ответ</p>"


def test_web_transport_streams_commands_as_single_command_event() -> None:
    registry = build_registry()
    transport = WebTransport(registry)

    transport.handle_text("/remember name=Андрей", session_id="browser-1")
    events = [event.to_payload() for event in transport.stream_text("/clear", session_id="browser-1")]

    assert events == [
        {
            "event": "command",
            "text": "🗑️  История чата очищена",
            "render_mode": "text",
            "is_command": True,
            "reset_transcript": True,
            "rag_enabled": False,
        }
    ]

    context = registry.acquire("web:browser-1")
    try:
        assert context.session.history == []
        assert [(fact.key, fact.value) for fact in context.session.service.list_memory_facts()] == [("name", "Андрей")]
    finally:
        registry.release(context)


def test_render_web_markdown_supports_basic_blocks() -> None:
    rendered = render_web_markdown("# Заголовок\n\n- пункт\n- второй\n\n`code`")

    assert "<h1>Заголовок</h1>" in rendered
    assert "<li>пункт</li>" in rendered
    assert "<li>второй</li>" in rendered
    assert "<code>code</code>" in rendered


def test_render_web_markdown_strips_raw_html_and_unsafe_links() -> None:
    rendered = render_web_markdown('<script>alert(1)</script><b>ok</b> [x](javascript:alert(1)) [safe](https://example.com)')

    assert "<script>" not in rendered
    assert "javascript:" not in rendered
    assert "<b>ok</b>" not in rendered
    assert "<span>x</span>" in rendered
    assert 'href="https://example.com"' in rendered


def test_split_renderable_markdown_keeps_trailing_line_as_plain_text_tail() -> None:
    prefix, tail = split_renderable_markdown("Готово\n- пункт")

    assert prefix == "Готово\n"
    assert tail == "- пункт"


def test_split_renderable_markdown_keeps_unclosed_fence_in_tail() -> None:
    prefix, tail = split_renderable_markdown("Готово\n```python\nprint('x')")

    assert prefix == "Готово\n"
    assert tail == "```python\nprint('x')"


def test_http_server_serves_page_and_streams_command_event() -> None:
    with run_test_server() as (host, port):
        connection = HTTPConnection(host, port, timeout=5)
        connection.request("GET", "/")
        page_response = connection.getresponse()
        page_html = page_response.read().decode("utf-8")
        cookie = page_response.getheader("Set-Cookie")
        connection.close()

        assert page_response.status == 200
        assert "SmartTherm Support" in page_html
        assert SESSION_COOKIE_NAME in (cookie or "")

        headers = {
            "Content-Type": "application/json",
            "Cookie": cookie or "",
        }
        body = json.dumps({"text": "/start"}, ensure_ascii=False)
        connection = HTTPConnection(host, port, timeout=5)
        connection.request("POST", "/api/message", body=body.encode("utf-8"), headers=headers)
        api_response = connection.getresponse()
        payload = _read_ndjson_response(api_response)
        content_type = api_response.getheader("Content-Type")
        connection.close()

        assert api_response.status == 200
        assert content_type == "application/x-ndjson; charset=utf-8"
        assert payload == [
            {
                "event": "command",
                "text": payload[0]["text"],
                "render_mode": "html",
                "is_command": True,
                "reset_transcript": False,
                "rag_enabled": False,
            }
        ]
        assert "<b>Привет! Я бот поддержки SmartTherm.</b>" in str(payload[0]["text"])


def test_http_server_streams_regular_message_as_ndjson_sequence() -> None:
    transport = WebTransport(build_registry(), preview_interval_ms=10_000, preview_min_chars=4)
    with run_test_server(transport) as (host, port):
        connection = HTTPConnection(host, port, timeout=5)
        body = json.dumps({"text": "Привет"}, ensure_ascii=False)
        connection.request(
            "POST",
            "/api/message",
            body=body.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = _read_ndjson_response(response)
        connection.close()

        assert response.status == 200
        assert [event["event"] for event in payload] == ["start", "token", "token", "preview", "final"]
        assert payload[0]["render_mode"] == "markdown"
        assert payload[1]["text"] == "от"
        assert payload[2]["text"] == "вет"
        assert payload[3]["render_mode"] == "markdown"
        assert payload[3]["html"] == "<p>ответ</p>"
        assert payload[4]["render_mode"] == "markdown"
        assert payload[4]["text"] == "<p>ответ</p>"


def test_http_server_stream_persists_history_only_after_final_event() -> None:
    registry = build_registry()
    transport = WebTransport(registry)

    payload = [event.to_payload() for event in transport.stream_text("Привет", session_id="browser-1")]

    assert payload[-1]["event"] == "final"

    context = registry.acquire("web:browser-1")
    try:
        assert [message.content for message in context.session.history] == ["Привет", "ответ"]
    finally:
        registry.release(context)

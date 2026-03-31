from __future__ import annotations

from contextlib import contextmanager
from http.client import HTTPConnection
import json
import threading
from typing import Iterator

from src.web import SESSION_COOKIE_NAME, WebTransport, create_web_server

from tests.test_cli_smoke import build_registry


@contextmanager
def run_test_server() -> Iterator[tuple[str, int]]:
    server = create_web_server(
        transport=WebTransport(build_registry()),
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


def test_web_transport_handles_start_command_as_html() -> None:
    transport = WebTransport(build_registry())

    response = transport.handle_text("/start", session_id="browser-1")

    assert response.is_command is True
    assert response.render_mode == "html"
    assert "<b>Привет! Я бот поддержки SmartTherm.</b>" in response.text
    assert response.dialog_key == "web:browser-1"


def test_web_transport_routes_text_and_isolates_sessions() -> None:
    registry = build_registry()
    transport = WebTransport(registry)

    first = transport.handle_text("Привет из первой вкладки", session_id="browser-1")
    second = transport.handle_text("Привет из второй вкладки", session_id="browser-2")

    assert first.is_command is False
    assert first.render_mode == "text"
    assert first.text == "single-response"
    assert second.text == "single-response"

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


def test_web_transport_clear_sets_reset_flag_and_preserves_memory() -> None:
    registry = build_registry()
    transport = WebTransport(registry)

    transport.handle_text("/remember name=Андрей", session_id="browser-1")
    transport.handle_text("Привет", session_id="browser-1")
    response = transport.handle_text("/clear", session_id="browser-1")

    assert response.is_command is True
    assert response.reset_transcript is True
    assert "История чата очищена" in response.text

    context = registry.acquire("web:browser-1")
    try:
        assert context.session.history == []
        assert [(fact.key, fact.value) for fact in context.session.service.list_memory_facts()] == [("name", "Андрей")]
    finally:
        registry.release(context)


def test_http_server_serves_page_and_chat_api() -> None:
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
        payload = json.loads(api_response.read().decode("utf-8"))
        connection.close()

        assert api_response.status == 200
        assert payload == {
            "text": payload["text"],
            "render_mode": "html",
            "is_command": True,
            "reset_transcript": False,
            "rag_enabled": False,
        }
        assert "<b>Привет! Я бот поддержки SmartTherm.</b>" in payload["text"]


def test_http_server_returns_text_payload_for_regular_message() -> None:
    with run_test_server() as (host, port):
        connection = HTTPConnection(host, port, timeout=5)
        body = json.dumps({"text": "Привет"}, ensure_ascii=False)
        connection.request(
            "POST",
            "/api/message",
            body=body.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()

        assert response.status == 200
        assert payload == {
            "text": "single-response",
            "render_mode": "text",
            "is_command": False,
            "reset_transcript": False,
            "rag_enabled": False,
        }

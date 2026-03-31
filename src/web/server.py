"""Minimal stdlib HTTP server for SmartTherm web chat."""

from __future__ import annotations

from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
from typing import Any
from urllib.parse import urlparse

from src.chat.application.command_service import CommandService

from .transport import WebStreamEvent, WebTransport

SESSION_COOKIE_NAME = "smart_therm_web_session"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 365


class WebChatHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server with non-blocking shutdown for request threads."""

    daemon_threads = True
    block_on_close = False


def _build_commands_hint() -> str:
    return " ".join(name for name, _description in CommandService.command_items())


def _ndjson_line(event: WebStreamEvent) -> bytes:
    return (json.dumps(event.to_payload(), ensure_ascii=False) + "\n").encode("utf-8")


def build_web_app_html() -> str:
    """Render the single-page minimal web UI."""
    commands_hint = _build_commands_hint()
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmartTherm Support</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: rgba(255, 251, 245, 0.9);
      --panel-strong: #fffdf8;
      --line: #d9cdbd;
      --text: #1e1b16;
      --muted: #6c6257;
      --accent: #1f6f5f;
      --accent-weak: #dcefe9;
      --user: #efe6d8;
      --shadow: 0 18px 40px rgba(84, 68, 43, 0.12);
      --code-bg: #efe6d8;
      --quote-bg: #f4eee4;
      --link: #0b5a8f;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(235, 223, 200, 0.9), transparent 32%),
        linear-gradient(180deg, #f7f2e8 0%, #f0ebe2 100%);
      color: var(--text);
      font: 14px/1.5 "SF Mono", "JetBrains Mono", "IBM Plex Mono", monospace;
    }}
    .shell {{
      width: min(920px, calc(100vw - 32px));
      margin: 24px auto;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(8px);
    }}
    .header {{
      padding: 18px 20px 14px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(31, 111, 95, 0.08), rgba(255, 255, 255, 0.5));
    }}
    .title {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
    }}
    .commands {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .transcript {{
      height: min(68vh, 720px);
      padding: 18px 20px;
      overflow-y: auto;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.55), rgba(247, 240, 229, 0.8)),
        repeating-linear-gradient(
          180deg,
          transparent 0,
          transparent 23px,
          rgba(217, 205, 189, 0.25) 24px
        );
    }}
    .message {{
      max-width: 88%;
      margin-bottom: 14px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-strong);
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .message-user {{
      margin-left: auto;
      background: var(--user);
    }}
    .message-system {{
      border-style: dashed;
      color: var(--muted);
      background: rgba(255, 253, 248, 0.7);
    }}
    .message-html pre,
    .message-markdown pre {{
      margin: 0;
      padding: 12px;
      overflow-x: auto;
      border-radius: 10px;
      background: var(--code-bg);
      white-space: pre-wrap;
      font: inherit;
    }}
    .message-markdown code {{
      padding: 0 4px;
      border-radius: 6px;
      background: var(--code-bg);
    }}
    .message-markdown blockquote {{
      margin: 0;
      padding-left: 12px;
      border-left: 3px solid var(--line);
      background: var(--quote-bg);
    }}
    .message-markdown ul,
    .message-markdown ol {{
      margin: 8px 0;
      padding-left: 22px;
    }}
    .message-markdown p,
    .message-markdown h1,
    .message-markdown h2,
    .message-markdown h3 {{
      margin: 0 0 10px;
    }}
    .message-markdown p:last-child,
    .message-markdown h1:last-child,
    .message-markdown h2:last-child,
    .message-markdown h3:last-child,
    .message-markdown ul:last-child,
    .message-markdown ol:last-child,
    .message-markdown blockquote:last-child,
    .message-markdown pre:last-child {{
      margin-bottom: 0;
    }}
    .message-markdown a,
    .message-html a {{
      color: var(--link);
    }}
    .message-tail {{
      margin-top: 8px;
      color: var(--muted);
      white-space: pre-wrap;
    }}
    .composer {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 16px 20px 20px;
      border-top: 1px solid var(--line);
      background: rgba(255, 251, 245, 0.95);
    }}
    .input {{
      width: 100%;
      min-height: 52px;
      resize: vertical;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffdf8;
      color: var(--text);
      font: inherit;
    }}
    .input:focus {{
      outline: 2px solid rgba(31, 111, 95, 0.18);
      border-color: var(--accent);
    }}
    .send {{
      min-width: 108px;
      padding: 0 18px;
      border: 0;
      border-radius: 14px;
      background: var(--accent);
      color: #f7f5f1;
      font: inherit;
      cursor: pointer;
    }}
    .send:disabled {{
      opacity: 0.6;
      cursor: progress;
    }}
    @media (max-width: 720px) {{
      .shell {{
        width: calc(100vw - 16px);
        margin: 8px auto;
        border-radius: 16px;
      }}
      .transcript {{
        height: 62vh;
        padding: 14px;
      }}
      .composer {{
        grid-template-columns: 1fr;
        padding: 14px;
      }}
      .message {{
        max-width: 100%;
      }}
      .send {{
        min-height: 48px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="header">
      <h1 class="title">SmartTherm Support</h1>
      <p class="subtitle">Минимальный веб-интерфейс к существующему чат-ядру.</p>
      <div class="commands">Команды: {commands_hint}</div>
    </section>
    <section id="transcript" class="transcript" aria-live="polite"></section>
    <form id="composer" class="composer">
      <textarea
        id="input"
        class="input"
        name="text"
        placeholder="Введите вопрос или команду. Например: /start"
        rows="3"
      ></textarea>
      <button id="send" class="send" type="submit">Отправить</button>
    </form>
  </main>
  <script>
    const transcript = document.getElementById("transcript");
    const composer = document.getElementById("composer");
    const input = document.getElementById("input");
    const send = document.getElementById("send");
    const decoder = new TextDecoder();
    let activeAssistantNode = null;
    let activeAssistantPreviewHtml = "";
    let activeAssistantTailText = "";

    function scrollTranscript() {{
      transcript.scrollTop = transcript.scrollHeight;
    }}

    function createMessageNode(kind, renderMode) {{
      const node = document.createElement("article");
      node.className = "message message-" + kind + (renderMode ? " message-" + renderMode : "");
      transcript.appendChild(node);
      scrollTranscript();
      return node;
    }}

    function appendMessage(text, kind, renderMode) {{
      const node = createMessageNode(kind, renderMode);
      updateMessageNode(node, text, renderMode);
      return node;
    }}

    function updateMessageNode(node, text, renderMode) {{
      node.className = "message message-" + (node.dataset.kind || "assistant") + (renderMode ? " message-" + renderMode : "");
      if (renderMode === "html" || renderMode === "markdown") {{
        node.innerHTML = text;
      }} else {{
        node.textContent = text;
      }}
      scrollTranscript();
    }}

    function setBusy(isBusy) {{
      send.disabled = isBusy;
      input.disabled = isBusy;
    }}

    function escapeHtml(text) {{
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function resetActiveAssistantNode() {{
      activeAssistantNode = null;
      activeAssistantPreviewHtml = "";
      activeAssistantTailText = "";
    }}

    function ensureAssistantPlaceholder() {{
      if (activeAssistantNode) {{
        return activeAssistantNode;
      }}
      activeAssistantNode = createMessageNode("assistant", "text");
      activeAssistantNode.dataset.kind = "assistant";
      activeAssistantNode.textContent = "";
      return activeAssistantNode;
    }}

    function renderActiveAssistantNode() {{
      const node = ensureAssistantPlaceholder();
      node.dataset.kind = "assistant";

      if (activeAssistantPreviewHtml) {{
        const tailHtml = activeAssistantTailText
          ? '<div class="message-tail">' + escapeHtml(activeAssistantTailText) + "</div>"
          : "";
        updateMessageNode(node, activeAssistantPreviewHtml + tailHtml, "markdown");
        return;
      }}

      updateMessageNode(node, activeAssistantTailText, "text");
    }}

    appendMessage("Введите /start, чтобы увидеть список команд.", "system", "text").dataset.kind = "system";

    async function readNdjsonStream(response) {{
      if (!response.body) {{
        throw new Error("Streaming not supported by this browser");
      }}

      const reader = response.body.getReader();
      let buffer = "";

      while (true) {{
        const {{ value, done }} = await reader.read();
        if (done) {{
          break;
        }}
        buffer += decoder.decode(value, {{ stream: true }});

        let lineBreakIndex = buffer.indexOf("\\n");
        while (lineBreakIndex >= 0) {{
          const line = buffer.slice(0, lineBreakIndex).trim();
          buffer = buffer.slice(lineBreakIndex + 1);
          if (line) {{
            handleStreamEvent(JSON.parse(line));
          }}
          lineBreakIndex = buffer.indexOf("\\n");
        }}
      }}

      const rest = buffer.trim();
      if (rest) {{
        handleStreamEvent(JSON.parse(rest));
      }}
    }}

    function handleStreamEvent(payload) {{
      if (payload.event === "start") {{
        const node = ensureAssistantPlaceholder();
        node.dataset.messageId = payload.message_id || "";
        node.dataset.kind = "assistant";
        activeAssistantPreviewHtml = "";
        activeAssistantTailText = "";
        node.textContent = "";
        return;
      }}

      if (payload.event === "token") {{
        activeAssistantTailText += payload.text || "";
        renderActiveAssistantNode();
        return;
      }}

      if (payload.event === "preview") {{
        activeAssistantPreviewHtml = payload.html || "";
        activeAssistantTailText = payload.tail_text || "";
        renderActiveAssistantNode();
        return;
      }}

      if (payload.event === "final") {{
        const node = ensureAssistantPlaceholder();
        node.dataset.kind = "assistant";
        updateMessageNode(node, payload.text || "", payload.render_mode || "markdown");
        resetActiveAssistantNode();
        return;
      }}

      if (payload.event === "command") {{
        if (payload.reset_transcript) {{
          transcript.innerHTML = "";
        }}
        const node = appendMessage(payload.text || "", "assistant", payload.render_mode || "text");
        node.dataset.kind = "assistant";
        resetActiveAssistantNode();
        return;
      }}

      if (payload.event === "error") {{
        const errorNode = appendMessage("Ошибка: " + (payload.error || "Unknown error"), "system", "text");
        errorNode.dataset.kind = "system";
        resetActiveAssistantNode();
      }}
    }}

    async function sendMessage(text) {{
      const response = await fetch("/api/message", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ text }}),
        credentials: "same-origin",
      }});

      if (!response.ok) {{
        let errorMessage = "Request failed";
        try {{
          const errorPayload = await response.json();
          errorMessage = errorPayload.error || errorMessage;
        }} catch (_error) {{
        }}
        throw new Error(errorMessage);
      }}

      await readNdjsonStream(response);
    }}

    composer.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const text = input.value.trim();
      if (!text) {{
        return;
      }}

      const userNode = appendMessage(text, "user", "text");
      userNode.dataset.kind = "user";
      input.value = "";
      setBusy(true);
      resetActiveAssistantNode();

      try {{
        await sendMessage(text);
      }} catch (error) {{
        const message = error instanceof Error ? error.message : "Unknown error";
        const errorNode = appendMessage("Ошибка: " + message, "system", "text");
        errorNode.dataset.kind = "system";
      }} finally {{
        setBusy(false);
        resetActiveAssistantNode();
        input.focus();
      }}
    }});

    input.addEventListener("keydown", (event) => {{
      if (event.key === "Enter" && !event.shiftKey) {{
        event.preventDefault();
        composer.requestSubmit();
      }}
    }});
  </script>
</body>
</html>
"""


def _new_session_id() -> str:
    return secrets.token_urlsafe(24)


def _extract_session_id(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    cookie = SimpleCookie()
    cookie.load(cookie_header)
    morsel = cookie.get(SESSION_COOKIE_NAME)
    if morsel is None:
        return None
    session_id = morsel.value.strip()
    return session_id or None


def _session_from_headers(headers: Any) -> tuple[str, bool]:
    existing = _extract_session_id(headers.get("Cookie"))
    if existing is not None:
        return existing, False
    return _new_session_id(), True


def _cookie_header(session_id: str) -> str:
    cookie = SimpleCookie()
    cookie[SESSION_COOKIE_NAME] = session_id
    cookie[SESSION_COOKIE_NAME]["path"] = "/"
    cookie[SESSION_COOKIE_NAME]["httponly"] = True
    cookie[SESSION_COOKIE_NAME]["samesite"] = "Lax"
    cookie[SESSION_COOKIE_NAME]["max-age"] = str(SESSION_COOKIE_MAX_AGE)
    return cookie.output(header="").strip()


def create_web_server(*, transport: WebTransport, host: str, port: int) -> WebChatHTTPServer:
    """Create a stdlib HTTP server bound to the web transport."""

    page_html = build_web_app_html().encode("utf-8")

    class WebChatHandler(BaseHTTPRequestHandler):
        server_version = "SmartThermWeb/1.0"

        def _send_bytes(
            self,
            *,
            status: HTTPStatus,
            content_type: str,
            body: bytes,
            session_id: str | None = None,
            set_cookie: bool = False,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if set_cookie and session_id is not None:
                self.send_header("Set-Cookie", _cookie_header(session_id))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self,
            status: HTTPStatus,
            payload: dict[str, Any],
            *,
            session_id: str | None = None,
            set_cookie: bool = False,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(
                status=status,
                content_type="application/json; charset=utf-8",
                body=body,
                session_id=session_id,
                set_cookie=set_cookie,
            )

        def _begin_stream(
            self,
            *,
            session_id: str,
            set_cookie: bool,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if set_cookie:
                self.send_header("Set-Cookie", _cookie_header(session_id))
            self.end_headers()

        def _write_stream_event(self, event: WebStreamEvent) -> None:
            self.wfile.write(_ndjson_line(event))
            self.wfile.flush()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/":
                self._send_bytes(
                    status=HTTPStatus.NOT_FOUND,
                    content_type="text/plain; charset=utf-8",
                    body="Not found".encode("utf-8"),
                )
                return

            session_id, set_cookie = _session_from_headers(self.headers)
            self._send_bytes(
                status=HTTPStatus.OK,
                content_type="text/html; charset=utf-8",
                body=page_html,
                session_id=session_id,
                set_cookie=set_cookie,
            )

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/message":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            session_id, set_cookie = _session_from_headers(self.headers)
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw_body = self.rfile.read(length)

            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Invalid JSON body"},
                    session_id=session_id,
                    set_cookie=set_cookie,
                )
                return

            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Field 'text' must be a non-empty string"},
                    session_id=session_id,
                    set_cookie=set_cookie,
                )
                return

            try:
                events = transport.stream_text(text.strip(), session_id=session_id)
                self._begin_stream(session_id=session_id, set_cookie=set_cookie)
                for event in events:
                    self._write_stream_event(event)
            except Exception as error:
                try:
                    self._write_stream_event(WebStreamEvent(event="error", error=str(error)))
                except Exception:
                    pass

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return WebChatHTTPServer((host, port), WebChatHandler)


def run_web_server(*, transport: WebTransport, host: str, port: int) -> None:
    """Start serving the minimal web UI."""
    server = create_web_server(transport=transport, host=host, port=port)
    try:
        server.serve_forever()
    finally:
        server.server_close()

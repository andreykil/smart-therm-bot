"""Конвертация markdown-ответов в Telegram-compatible HTML."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

import markdown


class _TelegramHTMLRenderer(HTMLParser):
    """Свести HTML из markdown к подмножеству тегов, поддерживаемому Telegram."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._list_stack: list[dict[str, int | str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: value for name, value in attrs}

        if tag in {"strong", "b"}:
            self._parts.append("<b>")
            return
        if tag in {"em", "i"}:
            self._parts.append("<i>")
            return
        if tag in {"s", "strike", "del"}:
            self._parts.append("<s>")
            return
        if tag in {"u", "ins"}:
            self._parts.append("<u>")
            return
        if tag == "blockquote":
            self._parts.append("<blockquote>")
            return
        if tag == "pre":
            self._parts.append("<pre>")
            return
        if tag == "code":
            language = attr_map.get("class")
            if language and language.startswith("language-"):
                self._parts.append(f'<code class="{html.escape(language, quote=True)}">')
            else:
                self._parts.append("<code>")
            return
        if tag == "a":
            href = attr_map.get("href")
            if href:
                self._parts.append(f'<a href="{html.escape(href, quote=True)}">')
            return
        if tag == "br":
            self._parts.append("\n")
            return
        if tag == "p":
            return
        if tag in {"ul", "ol"}:
            if tag == "ol":
                self._list_stack.append({"type": "ol", "index": 0})
            else:
                self._list_stack.append({"type": "ul", "index": 0})
            return
        if tag == "li":
            if self._parts and not self._parts[-1].endswith("\n"):
                self._parts.append("\n")
            if self._list_stack:
                current = self._list_stack[-1]
                if current["type"] == "ol":
                    current["index"] = int(current["index"]) + 1
                    self._parts.append(f'{current["index"]}. ')
                else:
                    self._parts.append("• ")
            else:
                self._parts.append("• ")
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._parts.append("<b>")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"strong", "b"}:
            self._parts.append("</b>")
            return
        if tag in {"em", "i"}:
            self._parts.append("</i>")
            return
        if tag in {"s", "strike", "del"}:
            self._parts.append("</s>")
            return
        if tag in {"u", "ins"}:
            self._parts.append("</u>")
            return
        if tag == "blockquote":
            self._parts.append("</blockquote>\n")
            return
        if tag == "pre":
            self._parts.append("</pre>\n")
            return
        if tag == "code":
            self._parts.append("</code>")
            return
        if tag == "a":
            self._parts.append("</a>")
            return
        if tag == "p":
            self._parts.append("\n\n")
            return
        if tag == "li":
            self._parts.append("\n")
            return
        if tag in {"ul", "ol"}:
            if self._list_stack:
                self._list_stack.pop()
            self._parts.append("\n")
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._parts.append("</b>\n\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(html.escape(data, quote=False))

    def render(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text.strip()


def render_telegram_html_from_markdown(text: str) -> str:
    """Преобразовать markdown-текст в HTML, который принимает Telegram."""
    if not text.strip():
        return ""

    markdown_html = markdown.markdown(text, extensions=["extra", "sane_lists"])
    renderer = _TelegramHTMLRenderer()
    renderer.feed(markdown_html)
    renderer.close()

    rendered = renderer.render()
    if rendered:
        return rendered

    return html.escape(text, quote=False)

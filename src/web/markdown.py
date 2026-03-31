"""Safe markdown rendering for the web chat UI."""

from __future__ import annotations

import html
from html.parser import HTMLParser
from urllib.parse import urlparse

import markdown

_ALLOWED_TAGS = {
    "p",
    "br",
    "strong",
    "b",
    "em",
    "i",
    "code",
    "pre",
    "blockquote",
    "ul",
    "ol",
    "li",
    "a",
    "h1",
    "h2",
    "h3",
}
_ALLOWED_LINK_SCHEMES = {"http", "https", "mailto"}


def _is_safe_href(href: str) -> bool:
    parsed = urlparse(href)
    if not parsed.scheme:
        return False
    return parsed.scheme.lower() in _ALLOWED_LINK_SCHEMES


class _SafeMarkdownHTMLRenderer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._open_tags: list[str] = []
        self._ignored_tag_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized not in _ALLOWED_TAGS:
            self._ignored_tag_depth += 1
            return

        if self._ignored_tag_depth > 0:
            self._ignored_tag_depth += 1
            return

        if normalized == "a":
            href = ""
            for attr_name, attr_value in attrs:
                if attr_name.lower() == "href" and attr_value:
                    href = attr_value.strip()
                    break
            if not href or not _is_safe_href(href):
                self._parts.append("<span>")
                self._open_tags.append("span")
                return
            escaped_href = html.escape(href, quote=True)
            self._parts.append(f'<a href="{escaped_href}" rel="noopener noreferrer" target="_blank">')
            self._open_tags.append("a")
            return

        self._parts.append(f"<{normalized}>")
        self._open_tags.append(normalized)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if self._ignored_tag_depth > 0:
            self._ignored_tag_depth -= 1
            return

        if not self._open_tags:
            return

        open_tag = self._open_tags.pop()
        if open_tag == "span":
            self._parts.append("</span>")
            return
        self._parts.append(f"</{open_tag}>")

    def handle_data(self, data: str) -> None:
        self._parts.append(html.escape(data, quote=False))

    def handle_entityref(self, name: str) -> None:
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._parts.append(f"&#{name};")

    def render(self) -> str:
        while self._open_tags:
            open_tag = self._open_tags.pop()
            if open_tag == "span":
                self._parts.append("</span>")
            else:
                self._parts.append(f"</{open_tag}>")
        return "".join(self._parts).strip()


def render_web_markdown(markdown_text: str) -> str:
    """Convert model markdown to a small safe HTML subset for the web UI."""
    if not markdown_text.strip():
        return ""

    escaped_source = html.escape(markdown_text, quote=False)
    rendered_html = markdown.markdown(escaped_source, extensions=["extra", "sane_lists"])
    parser = _SafeMarkdownHTMLRenderer()
    parser.feed(rendered_html)
    parser.close()
    sanitized = parser.render()
    if sanitized:
        return sanitized
    return f"<p>{html.escape(markdown_text, quote=False)}</p>"


def split_renderable_markdown(markdown_text: str) -> tuple[str, str]:
    """Split text into a renderable prefix and a plain-text tail.

    The goal is to render only the stable part of a streaming answer and leave
    the possibly incomplete tail as plain text.
    """
    if not markdown_text:
        return "", ""

    fence_count = markdown_text.count("```")
    candidate_prefix = markdown_text
    fenced_tail = ""
    if fence_count % 2 == 1:
        last_fence_index = markdown_text.rfind("```")
        if last_fence_index >= 0:
            candidate_prefix = markdown_text[:last_fence_index]
            fenced_tail = markdown_text[last_fence_index:]

    last_newline_index = candidate_prefix.rfind("\n")
    if last_newline_index >= 0:
        prefix = candidate_prefix[: last_newline_index + 1]
        tail = candidate_prefix[last_newline_index + 1 :] + fenced_tail
        if prefix.strip():
            return prefix, tail

    if fenced_tail:
        if candidate_prefix.strip():
            return candidate_prefix, fenced_tail
        return "", markdown_text

    return markdown_text, ""

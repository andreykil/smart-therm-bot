from src.bot.telegram_markdown import render_telegram_html_from_markdown


def test_render_telegram_html_from_markdown_supports_basic_blocks() -> None:
    rendered = render_telegram_html_from_markdown("# Заголовок\n\n- пункт\n- второй")

    assert "<b>Заголовок</b>" in rendered
    assert "• пункт" in rendered
    assert "• второй" in rendered


def test_render_telegram_html_from_markdown_preserves_inline_formatting() -> None:
    rendered = render_telegram_html_from_markdown("**жирный** и `code` и [ссылка](https://example.com)")

    assert "<b>жирный</b>" in rendered
    assert "<code>code</code>" in rendered
    assert '<a href="https://example.com">ссылка</a>' in rendered

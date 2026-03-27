import pytest

from src.utils.prompt_manager import PromptManager


@pytest.fixture()
def prompt_manager(monkeypatch: pytest.MonkeyPatch) -> PromptManager:
    prompts = {
        "simple": "Привет",
        "templated": "Вопрос: {question}",
        "double": "{{json}} {person}",
    }

    def fake_load_prompts(self: PromptManager) -> None:
        self._prompts = prompts

    monkeypatch.setattr(PromptManager, "_load_prompts", fake_load_prompts)
    PromptManager.reset()
    return PromptManager()


def test_get_prompt_returns_plain_template(prompt_manager: PromptManager) -> None:
    assert prompt_manager.get_prompt("simple") == "Привет"


def test_get_prompt_validates_missing_and_unexpected_variables(prompt_manager: PromptManager) -> None:
    with pytest.raises(ValueError):
        prompt_manager.get_prompt("templated")

    with pytest.raises(ValueError):
        prompt_manager.get_prompt("simple", extra="value")


def test_extract_placeholders_ignores_escaped_braces(prompt_manager: PromptManager) -> None:
    assert prompt_manager.get_prompt("double", person="Анна") == "{json} Анна"

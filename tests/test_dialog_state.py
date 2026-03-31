from pathlib import Path
import sqlite3
import tempfile

import pytest

from src.chat.domain.models import DialogMessage
from src.memory.sqlite_dialog_state import SQLiteDialogState
from src.memory.sqlite_repository import SQLiteMemoryRepository

from tests.helpers.in_memory_dialog_state import InMemoryDialogState


def test_in_memory_dialog_state_stores_single_canonical_history() -> None:
    state = InMemoryDialogState()

    state.append_turn(
        user_message="Привет",
        assistant_message="Здравствуйте",
        rag_enabled=True,
        rag_query="Привет",
        rag_total_found=2,
    )

    messages = state.recent_messages()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[0].rag_total_found == 2
    assert [message.to_llm_message() for message in messages] == [
        {"role": "user", "content": "Привет"},
        {"role": "assistant", "content": "Здравствуйте"},
    ]


def test_in_memory_dialog_state_clears_history_and_counts_turns() -> None:
    state = InMemoryDialogState()
    state.append_turn(
        user_message="Как подключить?",
        assistant_message="Подключите по схеме.",
        rag_enabled=False,
        rag_query="Как подключить?",
        rag_total_found=0,
    )

    assert state.stats() == {
        "messages_persisted": 2,
        "cached_messages": 2,
        "facts": 0,
        "history_window": 0,
    }

    state.clear()
    assert state.recent_messages() == []


def test_in_memory_dialog_state_clear_history_preserves_facts() -> None:
    state = InMemoryDialogState()
    state.remember_fact("name", "Андрей")
    state.append_turn(
        user_message="Как подключить?",
        assistant_message="Подключите по схеме.",
        rag_enabled=False,
        rag_query="Как подключить?",
        rag_total_found=0,
    )

    state.clear_history()

    assert state.recent_messages() == []
    assert [(fact.key, fact.value) for fact in state.list_facts()] == [("name", "Андрей")]


def test_in_memory_dialog_state_normalizes_fact_keys() -> None:
    state = InMemoryDialogState()

    fact = state.remember_fact(" Name ", " Андрей ")

    assert fact.key == "name"
    assert fact.value == "Андрей"
    assert [(saved.key, saved.value) for saved in state.list_facts()] == [("name", "Андрей")]


def test_in_memory_dialog_state_forget_uses_canonical_key() -> None:
    state = InMemoryDialogState()
    state.remember_fact("name", "Андрей")

    deleted = state.forget_fact(" Name ")

    assert deleted is True
    assert state.list_facts() == []


def test_sqlite_dialog_state_persists_turn_atomically() -> None:
    db_dir = Path(tempfile.mkdtemp())
    repository = SQLiteMemoryRepository(db_dir / "memory.sqlite3")
    state = SQLiteDialogState(repository, dialog_key="chat:42", history_window=12)

    state.append_turn(
        user_message="Привет",
        assistant_message="Здравствуйте",
        rag_enabled=False,
        rag_query="Привет",
        rag_total_found=0,
    )

    assert repository.count_messages("chat:42") == 2
    assert [message.role for message in state.recent_messages()] == ["user", "assistant"]
    assert len(state.recent_messages()) == 2


def test_sqlite_memory_repository_rolls_back_partial_turn_on_failure() -> None:
    db_dir = Path(tempfile.mkdtemp())
    repository = SQLiteMemoryRepository(db_dir / "memory.sqlite3")

    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_on_assistant_insert
            BEFORE INSERT ON dialog_messages
            WHEN NEW.content = 'FORCE_FAIL'
            BEGIN
                SELECT RAISE(FAIL, 'forced failure');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_turn(
            "chat:42",
            [
                DialogMessage(role="user", content="Привет"),
                DialogMessage(role="assistant", content="FORCE_FAIL"),
            ],
        )

    assert repository.count_messages("chat:42") == 0


def test_sqlite_dialog_state_clear_history_preserves_facts() -> None:
    db_dir = Path(tempfile.mkdtemp())
    repository = SQLiteMemoryRepository(db_dir / "memory.sqlite3")
    state = SQLiteDialogState(repository, dialog_key="chat:42", history_window=12)

    state.remember_fact("name", "Андрей")
    state.append_turn(
        user_message="Привет",
        assistant_message="Здравствуйте",
        rag_enabled=False,
        rag_query="Привет",
        rag_total_found=0,
    )

    state.clear_history()

    assert repository.count_messages("chat:42") == 0
    assert repository.count_facts("chat:42") == 1
    assert state.recent_messages() == []
    assert [(fact.key, fact.value) for fact in state.list_facts()] == [("name", "Андрей")]

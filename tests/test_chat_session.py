from src.chat import ChatSession


def test_chat_session_stores_single_canonical_history() -> None:
    session = ChatSession()

    session.append_turn(
        user_message="Привет",
        assistant_message="Здравствуйте",
        rag_enabled=True,
        rag_query="Привет",
        rag_total_found=2,
    )

    messages = session.messages
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[0].rag_total_found == 2
    assert session.to_llm_history() == [
        {"role": "user", "content": "Привет"},
        {"role": "assistant", "content": "Здравствуйте"},
    ]


def test_chat_session_clears_history_and_counts_turns() -> None:
    session = ChatSession()
    session.append_turn(
        user_message="Как подключить?",
        assistant_message="Подключите по схеме.",
        rag_enabled=False,
        rag_query="Как подключить?",
        rag_total_found=0,
    )

    assert session.stats() == {"messages": 2, "turns": 1}

    session.clear()
    assert session.messages == []

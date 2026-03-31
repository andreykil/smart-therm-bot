from __future__ import annotations

import scripts.reindex_rag as reindex_rag


class FakeIndexManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def index_from_file(self, chunks_file: str, *, save: bool = True):
        del save
        self.calls.append(chunks_file)
        return type(
            "FakeStats",
            (),
            {
                "total_chunks": 1,
                "faiss_vectors": 1,
                "bm25_documents": 1,
                "embedding_dim": 1024,
            },
        )()


class FakeRuntime:
    def __init__(self, index_manager: FakeIndexManager) -> None:
        self.index_manager = index_manager


def test_reindex_uses_chunks_file_from_config_by_default(monkeypatch, tmp_path) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text('{"id":"1","text":"hello","source":"test"}\n', encoding="utf-8")

    config = reindex_rag.Config.model_validate(
        {
            "project_root": tmp_path,
            "rag": {"chunks_file": "chunks.jsonl"},
            "llm": {"base_url": "http://localhost:11434"},
        }
    )
    index_manager = FakeIndexManager()

    monkeypatch.setattr(reindex_rag.Config, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(reindex_rag, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(reindex_rag, "build_rag_runtime", lambda **_: FakeRuntime(index_manager))

    reindex_rag.main([])

    assert index_manager.calls == [str(chunks_file)]


def test_reindex_cli_chunks_file_overrides_config(monkeypatch, tmp_path) -> None:
    default_chunks = tmp_path / "default.jsonl"
    override_chunks = tmp_path / "override.jsonl"
    default_chunks.write_text('{"id":"1","text":"default","source":"test"}\n', encoding="utf-8")
    override_chunks.write_text('{"id":"1","text":"override","source":"test"}\n', encoding="utf-8")

    config = reindex_rag.Config.model_validate(
        {
            "project_root": tmp_path,
            "rag": {"chunks_file": "default.jsonl"},
            "llm": {"base_url": "http://localhost:11434"},
        }
    )
    index_manager = FakeIndexManager()

    monkeypatch.setattr(reindex_rag.Config, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(reindex_rag, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr(reindex_rag, "build_rag_runtime", lambda **_: FakeRuntime(index_manager))

    reindex_rag.main(["--chunks-file", "override.jsonl"])

    assert index_manager.calls == [str(override_chunks)]

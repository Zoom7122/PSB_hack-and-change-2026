"""Веб-чат юридического ассистента ПСБ: HTTP API, история диалогов и интерфейс.

Запуск из корня репозитория:
    uvicorn API.app:app --host 0.0.0.0 --port 8000

Ответы формирует движок `API/engine.py` (тот же код, что в платформенном сабмите).
История хранится в SQLite отдельно для каждого тестового пользователя.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from API.engine import LegalAssistant, load_env_file

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("PSB_HISTORY_DB", str(BASE_DIR / "history.sqlite3")))
DEMO_USERS = ("user1", "user2", "user3")
LOG = logging.getLogger("legal_assistant")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)

    @field_validator("content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Сообщение не может состоять только из пробелов")
        return value


class ChatRequest(BaseModel):
    """Внешние поля модуля 1 из CONTRACTS-ALL.md."""
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=10000)
    history: list[Message] = Field(default_factory=list, max_length=40)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Вопрос не может состоять только из пробелов")
        return value


class SourceReference(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    revision_id: str | None
    section: str | None
    page: int | None
    source_path: str
    kind: Literal["law", "internal"] = "internal"
    outdated: bool = False
    excerpt: str = ""


class ChatResponse(BaseModel):
    request_id: str
    status: Literal["answered", "clarification_required", "out_of_scope", "insufficient"]
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    mode: str | None = None
    seconds: float | None = None


class HistoryMessage(Message):
    id: int
    created_at: str
    status: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    user_id: str
    messages: list[HistoryMessage]


def normalize(text: str) -> str:
    """Только безопасное форматирование: слова, пунктуация и переносы сохраняются."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    return text.strip()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """Соединение с историей; закрывается после использования."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=10000")
    db.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)")}
    if "meta" not in columns:                       # база старой версии модуля 1
        db.execute("ALTER TABLE messages ADD COLUMN meta TEXT")
    db.commit()
    try:
        yield db
    finally:
        db.close()


def validate_user(x_demo_user: str | None) -> str:
    """Только для локального демо: заголовок не является аутентификацией."""
    user = x_demo_user or DEMO_USERS[0]
    if user not in DEMO_USERS:
        raise HTTPException(status_code=400, detail="Неизвестный тестовый пользователь")
    return user


def record(db: sqlite3.Connection, user: str, role: str, content: str, meta: dict | None = None) -> None:
    db.execute(
        "INSERT INTO messages(user_id, role, content, created_at, meta) VALUES (?,?,?,?,?)",
        (user, role, content, datetime.now(timezone.utc).isoformat(),
         json.dumps(meta, ensure_ascii=False) if meta else None),
    )


_assistant: LegalAssistant | None = None
_assistant_lock = threading.Lock()


def get_assistant() -> LegalAssistant:
    global _assistant
    with _assistant_lock:
        if _assistant is None:
            _assistant = LegalAssistant()
        return _assistant


def set_assistant(assistant: LegalAssistant | None) -> None:
    """Подмена движка (тесты)."""
    global _assistant
    with _assistant_lock:
        _assistant = assistant


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env_file()
    get_assistant()                                  # база знаний загружается до первого вопроса
    yield
    if _assistant is not None:
        _assistant.close()


app = FastAPI(title="ПСБ — юридический ассистент", version="1.0.0", lifespan=lifespan)


@app.get("/")
def homepage() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", **get_assistant().health()}


@app.get("/api/users")
def users() -> dict[str, list[dict[str, str]]]:
    return {"users": [{"id": u, "label": f"Пользователь {i}"} for i, u in enumerate(DEMO_USERS, 1)]}


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": get_assistant().documents()}


@app.get("/api/documents/{document_id}")
def document(document_id: int) -> dict:
    found = get_assistant().document(document_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return found


@app.get("/api/history", response_model=HistoryResponse)
def get_history(x_demo_user: str | None = Header(default=None)) -> HistoryResponse:
    user = validate_user(x_demo_user)
    with connection() as db:
        rows = db.execute(
            "SELECT id, role, content, created_at, meta FROM messages WHERE user_id = ? ORDER BY id",
            (user,),
        ).fetchall()
    messages = []
    for row in rows:
        meta = json.loads(row["meta"]) if row["meta"] else {}
        messages.append(HistoryMessage(id=row["id"], role=row["role"], content=row["content"],
                                       created_at=row["created_at"], status=meta.get("status"),
                                       sources=meta.get("sources", [])))
    return HistoryResponse(user_id=user, messages=messages)


@app.delete("/api/history")
def clear_history(x_demo_user: str | None = Header(default=None)) -> dict[str, str]:
    user = validate_user(x_demo_user)
    with connection() as db:
        db.execute("DELETE FROM messages WHERE user_id = ?", (user,))
        db.commit()
    return {"status": "cleared"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, x_demo_user: str | None = Header(default=None)) -> ChatResponse:
    user = validate_user(x_demo_user)
    question = normalize(payload.question)
    if not question:
        raise HTTPException(status_code=422, detail="Пустой вопрос")
    history = [{"role": m.role, "content": normalize(m.content)} for m in payload.history]
    result = get_assistant().ask(question, history)
    response = ChatResponse(request_id=f"req-{uuid.uuid4().hex[:12]}", **result)
    with connection() as db:
        record(db, user, "user", question)
        record(db, user, "assistant", response.answer,
               {"status": response.status, "sources": [s.model_dump() for s in response.sources]})
        db.commit()
    return response

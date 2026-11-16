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
    conversation_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Вопрос не может состоять только из пробелов")
        return value


class ConversationRename(BaseModel):
    """Проверяет новое название сохранённого диалога."""
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=80)

    @field_validator("title")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Убирает лишние пробелы и запрещает пустые названия."""
        value = " ".join(value.split())
        if not value:
            raise ValueError("Название не может быть пустым")
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
    db.execute("BEGIN IMMEDIATE")
    db.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""")
    conversation_columns = {row["name"] for row in db.execute("PRAGMA table_info(conversations)")}
    if "title_is_custom" not in conversation_columns:
        db.execute("ALTER TABLE conversations ADD COLUMN title_is_custom INTEGER NOT NULL DEFAULT 0")
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
    if "conversation_id" not in columns:
        db.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
    migrate_history(db)
    db.execute("CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages(user_id, conversation_id, id)")
    db.execute("CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations(user_id, updated_at)")
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


def migrate_history(db: sqlite3.Connection) -> None:
    """Переносит старую общую историю каждого пользователя в отдельный диалог."""
    users = db.execute("SELECT DISTINCT user_id FROM messages WHERE conversation_id IS NULL").fetchall()
    for row in users:
        user = row["user_id"]
        first = db.execute("SELECT content, created_at FROM messages WHERE user_id = ? AND conversation_id IS NULL ORDER BY id LIMIT 1", (user,)).fetchone()
        last = db.execute("SELECT created_at FROM messages WHERE user_id = ? AND conversation_id IS NULL ORDER BY id DESC LIMIT 1", (user,)).fetchone()
        conversation_id = f"legacy-{user}"
        db.execute("INSERT OR IGNORE INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                   (conversation_id, user, first["content"][:80], first["created_at"], last["created_at"]))
        db.execute("UPDATE messages SET conversation_id = ? WHERE user_id = ? AND conversation_id IS NULL", (conversation_id, user))


def require_conversation(db: sqlite3.Connection, user: str, conversation_id: str) -> sqlite3.Row:
    """Находит диалог в рамках выбранного профиля."""
    row = db.execute("SELECT * FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return row


def record(db: sqlite3.Connection, user: str, role: str, content: str,
           meta: dict | None = None, conversation_id: str | None = None) -> None:
    """Добавляет сообщение с метаданными в указанный диалог."""
    db.execute(
        "INSERT INTO messages(user_id, role, content, created_at, meta, conversation_id) VALUES (?,?,?,?,?,?)",
        (user, role, content, datetime.now(timezone.utc).isoformat(),
         json.dumps(meta, ensure_ascii=False) if meta else None, conversation_id),
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
            "SELECT id, role, content, created_at, meta FROM messages WHERE user_id = ? AND conversation_id = ? ORDER BY id",
            (user, f"legacy-{user}"),
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
        db.execute("DELETE FROM messages WHERE user_id = ? AND conversation_id = ?", (user, f"legacy-{user}"))
        db.execute("DELETE FROM conversations WHERE user_id = ? AND id = ?", (user, f"legacy-{user}"))
        db.commit()
    return {"status": "cleared"}


@app.get("/api/conversations")
def conversations(x_demo_user: str | None = Header(default=None)) -> dict:
    """Возвращает сохранённые диалоги профиля от новых к старым."""
    user = validate_user(x_demo_user)
    with connection() as db:
        rows = db.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC, rowid DESC", (user,)).fetchall()
    return {"conversations": [dict(row) for row in rows]}


@app.post("/api/conversations", status_code=201)
def create_conversation(x_demo_user: str | None = Header(default=None)) -> dict:
    """Создаёт пустой диалог для выбранного профиля."""
    user = validate_user(x_demo_user)
    now = datetime.now(timezone.utc).isoformat()
    conversation_id = uuid.uuid4().hex
    with connection() as db:
        db.execute("INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", (conversation_id, user, "Новый диалог", now, now))
        db.commit()
    return {"id": conversation_id, "title": "Новый диалог", "created_at": now, "updated_at": now}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, payload: ConversationRename,
                        x_demo_user: str | None = Header(default=None)) -> dict:
    """Переименовывает диалог выбранного профиля и сохраняет название."""
    user = validate_user(x_demo_user)
    with connection() as db:
        require_conversation(db, user, conversation_id)
        db.execute("UPDATE conversations SET title = ?, title_is_custom = 1 WHERE id = ? AND user_id = ?",
                   (payload.title, conversation_id, user))
        db.commit()
    return {"id": conversation_id, "title": payload.title}


@app.get("/api/conversations/{conversation_id}/messages", response_model=HistoryResponse)
def conversation_messages(conversation_id: str, x_demo_user: str | None = Header(default=None)) -> HistoryResponse:
    """Загружает сообщения только принадлежащего профилю диалога."""
    user = validate_user(x_demo_user)
    with connection() as db:
        require_conversation(db, user, conversation_id)
        rows = db.execute("SELECT id, role, content, created_at, meta FROM messages WHERE user_id = ? AND conversation_id = ? ORDER BY id", (user, conversation_id)).fetchall()
    messages = []
    for row in rows:
        meta = json.loads(row["meta"]) if row["meta"] else {}
        messages.append(HistoryMessage(id=row["id"], role=row["role"], content=row["content"],
                                       created_at=row["created_at"], status=meta.get("status"), sources=meta.get("sources", [])))
    return HistoryResponse(user_id=user, messages=messages)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, x_demo_user: str | None = Header(default=None)) -> ChatResponse:
    user = validate_user(x_demo_user)
    question = normalize(payload.question)
    if not question:
        raise HTTPException(status_code=422, detail="Пустой вопрос")
    conversation_id = payload.conversation_id or f"legacy-{user}"
    if payload.conversation_id:
        with connection() as db:
            require_conversation(db, user, conversation_id)
            rows = db.execute("SELECT role, content FROM messages WHERE user_id = ? AND conversation_id = ? ORDER BY id DESC LIMIT 40", (user, conversation_id)).fetchall()
            history = [dict(row) for row in reversed(rows)]
    else:
        history = [{"role": m.role, "content": normalize(m.content)} for m in payload.history]
    result = get_assistant().ask(question, history)
    response = ChatResponse(request_id=f"req-{uuid.uuid4().hex[:12]}", **result)
    with connection() as db:
        now = datetime.now(timezone.utc).isoformat()
        db.execute("INSERT OR IGNORE INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", (conversation_id, user, "Новый диалог", now, now))
        require_conversation(db, user, conversation_id)
        db.execute("UPDATE conversations SET title = CASE WHEN title = 'Новый диалог' AND title_is_custom = 0 THEN ? ELSE title END, updated_at = ? WHERE id = ? AND user_id = ?", (question[:80], now, conversation_id, user))
        record(db, user, "user", question, conversation_id=conversation_id)
        record(db, user, "assistant", response.answer,
               {"status": response.status, "sources": [s.model_dump() for s in response.sources]}, conversation_id)
        db.commit()
    return response

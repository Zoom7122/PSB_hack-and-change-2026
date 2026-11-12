"""PSB module 1: HTTP gateway, validation, basic normalization and demo history.

No Query Processing, retrieval, or language model is connected here.
This independent module deliberately does not modify teammates' Front/main.py.
Run from the project root: uvicorn module1.app:app --reload
"""
from __future__ import annotations

import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("PSB_HISTORY_DB", str(BASE_DIR / "history.sqlite3")))
DEMO_USERS = ("user1", "user2", "user3")


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
    """Exact external fields of CONTRACTS-ALL.md module 1."""
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


class ChatResponse(BaseModel):
    request_id: str
    status: Literal["answered", "clarification_required", "out_of_scope", "insufficient"]
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)


class HistoryMessage(Message):
    id: int
    created_at: str


class HistoryResponse(BaseModel):
    user_id: str
    messages: list[HistoryMessage]


def normalize(text: str) -> str:
    """Safe formatting only: keep original words, punctuation, and newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    return text.strip()


def connection() -> sqlite3.Connection:
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
    db.commit()
    return db


def validate_user(x_demo_user: str | None) -> str:
    """Only for a local demo. Do not use header identity as production auth."""
    user = x_demo_user or DEMO_USERS[0]
    if user not in DEMO_USERS:
        raise HTTPException(status_code=400, detail="Неизвестный тестовый пользователь")
    return user


def record(db: sqlite3.Connection, user: str, role: str, content: str) -> None:
    db.execute(
        "INSERT INTO messages(user_id, role, content, created_at) VALUES (?,?,?,?)",
        (user, role, content, datetime.now(timezone.utc).isoformat()),
    )


app = FastAPI(title="PSB Legal Assistant — Module 1", version="0.1.0")


@app.get("/")
def homepage() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "module": "api_gateway", "processing": "not_connected"}


@app.get("/api/users")
def users() -> dict[str, list[dict[str, str]]]:
    return {"users": [{"id": u, "label": f"Пользователь {i}"} for i, u in enumerate(DEMO_USERS, 1)]}


@app.get("/api/history", response_model=HistoryResponse)
def get_history(x_demo_user: str | None = Header(default=None)) -> HistoryResponse:
    user = validate_user(x_demo_user)
    with connection() as db:
        rows = db.execute(
            "SELECT id, role, content, created_at FROM messages WHERE user_id = ? ORDER BY id",
            (user,),
        ).fetchall()
    return HistoryResponse(user_id=user, messages=[HistoryMessage(**dict(row)) for row in rows])


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, x_demo_user: str | None = Header(default=None)) -> ChatResponse:
    user = validate_user(x_demo_user)
    question = normalize(payload.question)
    if not question:
        raise HTTPException(status_code=422, detail="Пустой вопрос")
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    # Important: no invented legal claims or fake sources in mock mode.
    # When module 2 / RAG is integrated, replace this result with real pipeline output.
    result = ChatResponse(
        request_id=request_id,
        status="insufficient",
        answer="Демонстрационный режим: вопрос сохранён. Юридический поиск и генерация ответа ещё не подключены.",
        sources=[],
    )
    with connection() as db:
        record(db, user, "user", question)
        record(db, user, "assistant", result.answer)
        db.commit()
    return result

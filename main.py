import os
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="PSB Legal AI Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- МОДЕЛИ ДАННЫХ (Pydantic) ---
class DocumentChunk(BaseModel):
    id: int
    doc_name: str
    section: str
    text: str
    bm25_score: float
    vector_score: float
    rrf_score: float
    date_version: str
    is_active: bool = True


class ChatMessage(BaseModel):
    role: str  # "user" или "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []
    use_reranker: bool = True
    strict_mode: bool = True  # Контроль достоверности


class ChatResponse(BaseModel):
    answer: str
    status: str  # "success", "partial", "refusal", "clarification"
    status_message: str
    sources: List[DocumentChunk]
    pipeline_trace: dict
    latency_sec: float


# --- ЭНДПОИНТ ДЛЯ ПОДКЛЮЧЕНИЯ БЭКЕНДА ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Запрос не может быть пустым")

    # =========================================================================
    # ВНИМАНИЕ БЭКЕНДЕРАМ:
    # Здесь подключается ваш реальный RAG пайплайн:
    # 1. Поиск BM25 + BGE-M3
    # 2. RRF слияние
    # 3. Qwen3 Reranker
    # 4. DeepSeek V4.1 Flash генерация + Self-Check
    # =========================================================================

    # ДЕМО-ЛОГИКА (Для мгновенной работы из коробки):
    if "устав" in query.lower() or "правоспособност" in query.lower():
        status = "success"
        status_message = "Ответ подтвержден базой знаний"
        answer = (
            "Для проверки правоспособности контрагента согласно **Регламенту ПСБ № 114-П** и **ст. 49–51 ГК РФ** "
            "необходимо запросить актуальную выписку из ЕГРЮЛ (сроком не более 30 дней) и действующую редакцию Устава со всеми изменениями [1].\n\n"
            "**Порядок проверки:**\n"
            "1. Сверить полномочия генерального директора по выписке и протоколу назначения [1].\n"
            "2. Проверить наличие ограничений на совершение крупных сделок в Уставе (раздел 7) [2].\n"
            "3. Убедиться в отсутствии сведений о недостоверности адреса или банкротстве [1]."
        )
        sources = [
            DocumentChunk(
                id=1,
                doc_name="Регламент проверки контрагентов ПСБ (Ред. от 15.01.2024)",
                section="п. 3.2.1 Полномочия органов управления",
                text="Юрист обязан удостовериться в правоспособности юридического лица по выписке из ЕГРЮЛ и действующей редакции устава с отметками ФНС.",
                bm25_score=0.88,
                vector_score=0.91,
                rrf_score=0.032,
                date_version="15.01.2024 (Действующая)",
                is_active=True,
            ),
            DocumentChunk(
                id=2,
                doc_name="ГК РФ Статья 49. Правоспособность юридического лица",
                section="Часть 1, п. 2",
                text="Юридическое лицо может быть ограничено в правах лишь в случаях и в порядке, предусмотренных законом.",
                bm25_score=0.74,
                vector_score=0.85,
                rrf_score=0.027,
                date_version="01.01.2024",
                is_active=True,
            ),
        ]
    elif "взятк" in query.lower() or "обойти" in query.lower() or "незакон" in query.lower():
        status = "refusal"
        status_message = "Отказ: Запрос противоречит законодательству РФ и политике комплаенс"
        answer = (
            "Ассистент не предоставляет консультаций и инструкций по совершению противоправных действий. "
            "В соответствии со **ст. 290–291 УК РФ** и **Антикоррупционной политикой Банка**, любые действия, направленные на обход официальных процедур, строго запрещены."
        )
        sources = []
    else:
        status = "success"
        status_message = "Ответ сформирован на основе регламентов"
        answer = (
            f"По вашему запросу: «*{query}*» найдена информация в нормативной документации Банка [1].\n\n"
            "Применение условий договора должно осуществляться с учетом положений действующего регламента кредитования юридических лиц."
        )
        sources = [
            DocumentChunk(
                id=1,
                doc_name="Типовой кредитный регламент ПСБ № 45-Р",
                section="Раздел 4. Порядок согласования",
                text="Срок рассмотрения нестандартных условий кредитного соглашения юридическим департаментом составляет не более 3 рабочих дней.",
                bm25_score=0.82,
                vector_score=0.87,
                rrf_score=0.029,
                date_version="01.02.2024",
                is_active=True,
            )
        ]

    latency = round(time.time() - start_time, 2)

    pipeline_trace = {
        "analysis": "Успешно",
        "retrieval": {
            "bm25_count": 15,
            "vector_count": 15,
            "rrf_merged": len(sources),
            "reranked_top_k": len(sources),
        },
        "model": "DeepSeek V4.1 Flash",
        "hallucination_check": "Пройден (Факт-чек 100%)",
        "strict_date_match": True,
    }

    return ChatResponse(
        answer=answer,
        status=status,
        status_message=status_message,
        sources=sources,
        pipeline_trace=pipeline_trace,
        latency_sec=latency,
    )


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
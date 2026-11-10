from __future__ import annotations
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import answering
import chunking
import config
import ingest
import question_analysis as qa
import search
from hybrid import HybridSearcher
from llm_client import make_client
from vector_index import build_or_load
MAX_WORDS = min(getattr(config, 'ANSWER_MAX_WORDS', 150), 140)
MAX_CHARS = 9000
FALLBACK_ANSWER = 'Не удалось подготовить ответ по этому вопросу в отведённое время. Требуется ручная проверка по базе документов.'

# Вывод в консоль: если поток перенаправлен (pipe/файл), Windows использует
# кодировку локали (cp1251/cp866), и печать символов «→», «×» роняет скрипт
# с UnicodeEncodeError. Переключаем такой поток на UTF-8; интерактивную
# консоль не трогаем, чтобы в ней осталась родная кодировка.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


@dataclass
class QuestionRow:
    qid: str
    question: str

@dataclass
class Solved:
    qid: str
    answer: str
    source: str = 'model'
    seconds: float = 0.0
    debug: dict | None = None

def find_questions_file(explicit: str | None=None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f'Файл вопросов не найден: {path}')
        return path
    for name in ('questions.csv', 'demo_questions.csv'):
        candidate = Path(config.DATA_DIR) / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Не найден ни questions.csv, ни demo_questions.csv в {config.DATA_DIR}')

def load_questions(path: Path) -> list[QuestionRow]:
    rows: list[QuestionRow] = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            qid = (row.get('id') or '').strip()
            question = (row.get('question') or '').strip()
            if qid and question:
                rows.append(QuestionRow(qid=qid, question=question))
    if not rows:
        raise ValueError(f'В файле {path} нет ни одного вопроса с полями id, question')
    return rows

def normalize_answer(text: str) -> str:
    clean = answering.sanitize_answer(text, max_words=MAX_WORDS)
    if len(clean) > MAX_CHARS:
        cut = clean[:MAX_CHARS]
        stop = max(cut.rfind('. '), cut.rfind('! '), cut.rfind('? '))
        clean = cut[:stop + 1] if stop > 0 else cut
        clean = clean.strip()
    return clean

def write_submission(rows: list[Solved], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'answer'], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({'id': row.qid, 'answer': normalize_answer(row.answer)})

def validate_submission(expected_ids: list[str], path: Path) -> list[str]:
    problems: list[str] = []
    if not path.exists():
        return [f'файл не создан: {path}']
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if columns != ['id', 'answer']:
            problems.append(f"колонки должны быть ['id', 'answer'], а есть {columns}")
        got: list[str] = []
        for line_no, row in enumerate(reader, start=2):
            qid = (row.get('id') or '').strip()
            answer = (row.get('answer') or '').strip()
            got.append(qid)
            if not answer:
                problems.append(f'строка {line_no}: пустой ответ (id={qid})')
                continue
            words = len(answer.split())
            if words > 150:
                problems.append(f'строка {line_no}: {words} слов, регламент — до 150 (id={qid})')
            if len(answer) > 10000:
                problems.append(f'строка {line_no}: {len(answer)} символов, лимит 10 000 (id={qid})')
    if len(got) != len(expected_ids):
        problems.append(f'строк {len(got)}, а вопросов {len(expected_ids)}')
    missing = set(expected_ids) - set(got)
    extra = set(got) - set(expected_ids)
    if missing:
        problems.append(f'нет ответов на вопросы: {sorted(missing)[:5]}')
    if extra:
        problems.append(f'лишние id: {sorted(extra)[:5]}')
    if len(set(got)) != len(got):
        problems.append('есть повторяющиеся id')
    return problems

def solve_one(row: QuestionRow, searcher: HybridSearcher, client, meta: dict) -> Solved:
    started = time.time()
    analysis = qa.analyze(row.question, meta)
    hits = searcher.search(analysis.search_text, final_k=config.RERANK_TOP_N)
    best_bm25 = max((getattr(hit, 'bm25_score', 0.0) for hit in hits), default=0.0)
    best_vector = max((getattr(hit, 'vector_score', 0.0) for hit in hits), default=0.0)
    result = answering.answer_question(analysis, hits, client, meta=meta, best_bm25=best_bm25, best_vector=best_vector)
    source = 'refused' if result.refused else 'model'
    debug = {'id': row.qid, 'kind': analysis.kind, 'analysis': analysis.describe(), 'answer': result.text, 'source': source, 'attempts': result.attempts, 'citations': [{'index': c.index, 'label': c.label} for c in result.citations], 'claims': [{'claim': str(c.claim), 'verdict': c.verdict, 'found': c.found} for c in result.claim_checks], 'problems': result.problems, 'reason': result.reason}
    return Solved(qid=row.qid, answer=result.text, source=source, seconds=time.time() - started, debug=debug)

def fallback(row: QuestionRow, reason: str) -> Solved:
    return Solved(qid=row.qid, answer=FALLBACK_ANSWER, source='fallback', seconds=0.0, debug={'id': row.qid, 'source': 'fallback', 'reason': reason})

def solve_all(rows: list[QuestionRow], searcher: HybridSearcher, client, meta: dict, *, budget_sec: int, workers: int=1, verbose: bool=True) -> list[Solved]:
    deadline = time.time() + budget_sec
    soft_deadline = deadline - max(1.0, budget_sec * 0.15)
    results: dict[str, Solved] = {}
    errors: list[str] = []
    lock = threading.Lock()
    # Предохранитель: если модель стабильно недоступна, нет смысла ждать
    # ретраев по каждому вопросу — это съедает бюджет.
    breaker = {"failures": 0, "open": False}
    breaker_limit = 3

    def handle(row: QuestionRow) -> None:
        if time.time() > soft_deadline:
            with lock:
                results[row.qid] = fallback(row, 'бюджет времени исчерпан')
            return
        try:
            solved = solve_one(row, searcher, client, meta)
            with lock:
                breaker["failures"] = 0
                results[row.qid] = solved
                if verbose:
                    print(f"  {row.qid:>3} | {solved.source:<7} | {solved.seconds:4.1f} с | {solved.answer[:70].replace(chr(10), ' ')}…")
        except Exception as exc:
            message = f'{row.qid}: {type(exc).__name__}: {exc}'
            with lock:
                errors.append(message)
                breaker["failures"] += 1
                if breaker["failures"] >= breaker_limit and not breaker["open"]:
                    breaker["open"] = True
                    print(f"[solution] {breaker_limit} ошибки подряд — "
                          f"остальные вопросы пойдут резервными без обращения к модели")
                results[row.qid] = fallback(row, message)
                if verbose:
                    print(f'  {row.qid:>3} | ОШИБКА  | {message}')
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(handle, rows))
    else:
        for row in rows:
            handle(row)
    if errors:
        print(f'\nОшибок при обработке: {len(errors)}')
        for message in errors[:10]:
            print(f'  {message}')
    return [results.get(row.qid) or fallback(row, 'нет результата') for row in rows]

def build_searcher(client):
    """Собрать индексы. Если эмбеддинги недоступны — работать на одном BM25.

    Почему так: без этого при любой ошибке провайдера эмбеддингов прогон падал
    целиком, и submission.csv не создавался вовсе — то есть сабмит проваливался
    по причине «Отсутствует submission.csv». Лучше ответить по лексике (BM25 сам
    даёт hit@5 0.82), чем не ответить никак.
    """
    archive = ingest.find_knowledge_base(config.DATA_DIR)
    documents = ingest.load_documents_from_zip(archive)
    chunks, metas = chunking.build_chunks(documents, verbose=False)
    lexical = search.build_index(chunks)
    try:
        vectors, rebuilt = build_or_load(chunks, client, model=config.EMB_MODEL)
        state = 'собран' if rebuilt else 'из кэша'
        print(f"Фрагментов: {len(chunks)} | документов: {len(metas)} | индекс: {state} (dim={vectors.dim})")
    except Exception as exc:
        print(f"[solution] ЭМБЕДДИНГИ НЕДОСТУПНЫ ({type(exc).__name__}: {exc})")
        print("[solution] работаю на одном BM25: ответы менее точные, но файл создастся")
        vectors = empty_vector_index(chunks)
    meta = {doc_id: row.__dict__ for doc_id, row in metas.items()}
    cached = qa.load_doc_meta()
    if cached:
        meta = cached
    return (HybridSearcher(lexical, vectors, client=client), meta)

class DryRunClient:

    def chat(self, messages, temperature=None) -> str:
        return 'Пробный ответ: проверка конвейера без обращения к модели [1].'

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    def embed_documents(self, texts, batch_size=None):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed(self, texts, batch_size=None):
        return self.embed_documents(texts, batch_size)

    def rerank(self, query, documents, top_n):
        return [(index, 1.0) for index in range(min(top_n, len(documents)))]

    def rerank_by_llm(self, query, documents, top_n):
        return [(index, 1.0) for index in range(min(top_n, len(documents)))]

    def close(self) -> None:
        return None

def empty_vector_index(chunks):
    """Пустой векторный индекс той же длины: включает режим «только BM25».

    HybridSearcher требует совпадения длин, а поиск по векторам при пустой
    матрице просто не даст результатов — слияние возьмёт только BM25.
    """
    import numpy as np
    from vector_index import VectorIndex
    matrix = np.zeros((len(chunks), 1), dtype=np.float32)
    return VectorIndex(chunks, matrix)


def build_dry_run_searcher(chunks_note: str=''):
    archive = ingest.find_knowledge_base(config.DATA_DIR)
    documents = ingest.load_documents_from_zip(archive)
    chunks, metas = chunking.build_chunks(documents, verbose=False)
    lexical = search.build_index(chunks)
    from vector_index import build_vector_index
    client = DryRunClient()
    vectors = build_vector_index(chunks, client, verbose=False)
    meta = {doc_id: row.__dict__ for doc_id, row in metas.items()}
    print(f'[dry-run] фрагментов: {len(chunks)}, векторы: заглушка dim={vectors.dim}{chunks_note}')
    return (HybridSearcher(lexical, vectors, client=client), meta, client)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', help='путь к CSV с вопросами')
    parser.add_argument('--out', default='submission.csv', help='куда писать результат')
    parser.add_argument('--limit', type=int, default=0, help='обработать первые N вопросов')
    parser.add_argument('--workers', type=int, default=int(getattr(config, 'SOLVER_WORKERS', 4)), help='сколько вопросов обрабатывать параллельно')
    parser.add_argument('--budget', type=int, default=int(getattr(config, 'TIME_BUDGET_SEC', 1200)), help='бюджет времени в секундах')
    parser.add_argument('--dry-run', action='store_true', help='проверить конвейер без обращения к API')
    args = parser.parse_args()
    started = time.time()
    questions_path = find_questions_file(args.questions)
    rows = load_questions(questions_path)
    if args.limit:
        rows = rows[:args.limit]
    print(f'Вопросов: {len(rows)} | файл: {questions_path}')
    print(f'Бюджет: {args.budget} с | потоков: {args.workers}')
    client = None
    try:
        if args.dry_run:
            searcher, meta, client = build_dry_run_searcher()
        else:
            if not config.LLM_API_KEY:
                print('Не задан LLM_API_KEY. Для проверки конвейера: python solution.py --dry-run')
                return 1
            print(config.describe())
            client = make_client()
            searcher, meta = build_searcher(client)
        print('\nОБРАБОТКА')
        solved = solve_all(rows, searcher, client, meta, budget_sec=args.budget, workers=args.workers)
        out_path = Path(args.out)
        write_submission(solved, out_path)
        debug_path = Path(config.ARTIFACTS_DIR) / 'submission_debug.jsonl'
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open('w', encoding='utf-8') as handle:
            for item in solved:
                handle.write(json.dumps(item.debug or {'id': item.qid, 'source': item.source}, ensure_ascii=False) + '\n')
        counts: dict[str, int] = {}
        for item in solved:
            counts[item.source] = counts.get(item.source, 0) + 1
        elapsed = time.time() - started
        words = [len(normalize_answer(item.answer).split()) for item in solved]
        print('\n' + '=' * 70)
        print('СВОДКА')
        print(f"  ответов моделью: {counts.get('model', 0)}")
        print(f"  отказов (нет ответа в базе): {counts.get('refused', 0)}")
        print(f"  резервных (время/ошибка): {counts.get('fallback', 0)}")
        print(f'  время: {elapsed:.0f} с из {args.budget} с')
        print(f'  слов в ответе: среднее {sum(words) / len(words):.0f}, максимум {max(words)} (лимит 150)')
        print(f'  файл: {out_path.resolve()}')
        print(f'  журнал: {debug_path.resolve()}')
        problems = validate_submission([row.qid for row in rows], out_path)
        if problems:
            print('\nПРОВЕРКА ФАЙЛА: есть проблемы')
            for problem in problems[:15]:
                print(f'  - {problem}')
            return 1
        print('\nПРОВЕРКА ФАЙЛА: всё в порядке (колонки, id, длина, кодировка)')
        print('Файл готов к загрузке.')
        return 0
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
if __name__ == '__main__':
    sys.exit(main())

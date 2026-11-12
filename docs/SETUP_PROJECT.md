# Как перенести проект в PyCharm и запушить в git

Пошагово, для Windows. Всё, что ниже, выполняется в терминале PyCharm
(`Alt+F12`). Никаких действий в браузере, кроме подтверждения входа в GitHub.

## 1. Создать проект

PyCharm → **New Project** → **Pure Python**

* Location: `C:\Users\<вы>\PycharmProjects\psb-rag-modules`
* Interpreter type: **Virtualenv**, base interpreter — Python 3.11 или 3.12
* Снимите галочку **Create a main.py** (файл не нужен, только мешает)

Название проекта с дефисом — нормально для git; если PyCharm ругается,
используйте `psb_rag_modules`.

## 2. Скопировать файлы

Порядок важен только в одном смысле: сначала папки, потом файлы — так PyCharm
сразу видит структуру. Копируйте по одному, 22 файла:

**Корень (12 модулей + 3 служебных файла):**

1. `requirements.txt`
2. `.gitignore`
3. `README.md`
4. `config.py`
5. `refs.py`
6. `textproc.py`
7. `ingest.py`
8. `chunking.py`
9. `search.py`
10. `vector_index.py`
11. `llm_client.py`
12. `hybrid.py`
13. `contract_adapter.py`
14. `refs.py` (если ещё не скопирован выше)

**папка `scripts/`:**

15. `prepare_chunks.py`

**папка `tests/`:**

16. `test_contract.py`
17. в `tests/fixtures/` — 4 файла JSON

**папка `data/`:**

18. `knowledge-base.zip`

**папка `docs/`:**

19. `CONTRACTS-ALL.md`
20. `MODULE_3_4_REVIEW.md`
21. `SETUP_PROJECT.md` (этот файл)

Чего здесь **нет** и быть не должно: `answering.py`, `solution.py`,
`check_step*.py`, `make_zip.py`, `build_single.py`, `submission*.zip`,
`artifacts/`, `.env`, `backups/`, `ui/`. Это зона сабмита и чужие модули.

## 3. Установить зависимости

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Проверить, что всё живо

```powershell
python scripts/prepare_chunks.py
python tests/test_contract.py
```

Ожидается:

```
документов: 21 | фрагментов: 360
ИТОГ: пройдено 107, провалено 0
Фикстуры записаны в tests/fixtures/ (4 файла)
```

Если `ModuleNotFoundError: rank_bm25` — активирован не тот интерпретатор
(сравните `.venv\Scripts\python.exe` и `where python`).

## 5. Инициализировать git

```powershell
git init
git add .
git commit -m "Модули 3 и 4: гибридный поиск, RRF, реранк, контракт v1.1"
git branch -M feature/modules-3-4
```

Проверить, что лишнего не попало:

```powershell
git status
```

В списке должно быть 23 файла и ни одного `.env`, `.jsonl`, `__pycache__`.

## 6. Подключить удалённый репозиторий

Вариант **A — есть общий репозиторий команды** (нужны права на запись):

```powershell
git remote add origin https://github.com/<организация>/<репозиторий>.git
git fetch origin
git branch --set-upstream-to=origin/main feature/modules-3-4
git pull --rebase origin main
git push -u origin feature/modules-3-4
```

`--rebase` перед пушем — чтобы не тащить merge-коммит в чужую ветку.

Вариант **B — свой репозиторий** (создать пустой на GitHub, без README):

```powershell
git remote add origin https://github.com/<ваш-логин>/psb-rag-modules.git
git push -u origin feature/modules-3-4
```

Если GitHub просит пароль — вводите **Personal Access Token**, обычный пароль
не подойдёт. Токен: Settings → Developer settings → Personal access tokens →
Fine-grained → права `Contents: Read and write` на нужный репозиторий.

## 7. Открыть Pull Request

```powershell
git push -u origin feature/modules-3-4
```

GitHub выведет ссылку вида
`https://github.com/<…>/compare/feature/modules-3-4?expand=1` — открыть,
заголовок **«Модули 3 и 4: гибридный поиск + реранк, контракт v1.1»**, в
описании — три вопроса из `docs/MODULE_3_4_REVIEW.md`, §6.

## Если что-то пошло не так

| Симптом | Причина |
|---|---|
| `ModuleNotFoundError: rank_bm25` | не активирован `.venv` или установка шла в другой интерпретатор |
| `python` открывает Microsoft Store | используйте `.venv\Scripts\python.exe`, а не системный `python` |
| `fatal: not a git repository` | `git init` выполнен не в корне проекта |
| `Updates were rejected` | `git pull --rebase origin main` и повторить `push` |
| `.env` попал в `git status` | проверьте, что `.gitignore` в корне и называется именно так, затем `git check-ignore -v .env` |
| `.gitignore` выглядит пустым | это скрытый файл; Explorer не показывает файлы с точкой. Скопируйте содержимое из `docs/STRUCTURE.txt` |
| PyCharm не видит `tests/fixtures` | папка пустая в git не хранится; здесь файлы есть, но после `git clone` проверьте, что они на месте |

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
COPY query_processing_module/requirements.txt query_processing_module/
COPY data_processing/requirements.txt data_processing/
COPY platform_solution/requirements.txt platform_solution/
RUN pip install -r requirements.txt -r platform_solution/requirements.txt

COPY . .

ENV DATA_DIR=/app/data \
    PSB_HISTORY_DB=/app/var/history.sqlite3

RUN useradd --create-home --uid 1000 app && mkdir -p /app/var && chown -R app /app/var
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]

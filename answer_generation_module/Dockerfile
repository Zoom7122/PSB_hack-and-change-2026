FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY answer_generation ./answer_generation
COPY api ./api
EXPOSE 8005
CMD ["sh", "-c", "exec uvicorn api.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8005}"]

FROM python:3.13-slim

# MODEL_DIR and LOG_DB are the env vars app.py reads (see its docstring)
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models LOG_DB=/data/predictions.db

# non-root user, plus a folder it may write the prediction log to
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ ./src/
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
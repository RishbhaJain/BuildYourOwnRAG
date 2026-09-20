FROM python:3.11-slim

ARG INSTALL_ML_DEPS=1

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_AUTO_DOWNLOAD_MODEL=1 \
    RAG_MODE=production

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

COPY requirements.txt requirements-service.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-service.txt \
    && if [ "$INSTALL_ML_DEPS" = "1" ]; then python -m pip install -r requirements.txt; fi

COPY --chown=app:app . .
RUN chown app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

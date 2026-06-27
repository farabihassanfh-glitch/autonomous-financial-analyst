# Public deployment image (Railway). Slim — core agent only, no RAG/torch.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install core deps first for better layer caching (RAG extras omitted on purpose).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY financial_analyst_agent/ ./financial_analyst_agent/
COPY app_streamlit.py .

EXPOSE 8501

# Railway injects $PORT; default to 8501 for local `docker run`.
CMD streamlit run app_streamlit.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

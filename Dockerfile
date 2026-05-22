# Container image for the FastAPI app (REST API + streaming UI).
#
# Build:  docker build -t rag-system .
# Run:    docker run --rm -p 8000:8000 --env-file app/.env rag-system
#
# On Render/Railway the platform injects $PORT; the start command binds to it.

FROM python:3.11-slim

# System deps for Docling's PDF rendering (pdfium / poppler-compatible libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app

# Install Python deps first so the wheel cache layer is reused on code changes
COPY app/requirements.txt /workspace/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ /workspace/app/

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000

EXPOSE 8000

# Bind to the platform-provided $PORT (defaults to 8000 locally).
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

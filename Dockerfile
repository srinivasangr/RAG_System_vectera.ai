# Single-stage Dockerfile — convenience for reviewers who prefer a container.
# Not required to run the app (the venv path in README is faster); kept as an
# alternative so `docker compose up` is one command.
#
# Build:  docker build -t rag-system .
# Run:    docker run --rm -p 8501:8501 --env-file app/.env rag-system

FROM python:3.11-slim

# System deps for Docling's PDF rendering (pdfium / poppler-compatible libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first so the wheel cache layer is reused on code changes
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . /workspace

# Streamlit defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

# Run the app from the repo root; the streamlit script handles sys.path itself.
CMD ["streamlit", "run", "app/rag_system/ui/streamlit_app.py"]

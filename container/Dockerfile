# The image Cloudflare runs. wrangler builds it locally with Docker on
# `wrangler deploy` and pushes it to Cloudflare's registry.

# slim, not full: a smaller image boots faster, and boot time is the cold start.
# No --platform here: wrangler already builds for linux/amd64.
FROM python:3.12-slim

WORKDIR /app

# The container runs with no internet access (enableInternet = false), so anything
# LlamaIndex would download on first use — tiktoken vocabularies, NLTK data — is
# fetched here, at build time, into fixed paths used again at run time.
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    LLAMA_INDEX_CACHE_DIR=/app/.cache/llama_index \
    PYTHONUNBUFFERED=1

# Dependencies first, code second, so editing the app doesn't reinstall everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o-mini'); \
from llama_index.core import Document; \
from app.llama.engine import _splitters; \
from llama_index.core.ingestion import IngestionPipeline; \
IngestionPipeline(transformations=_splitters()).run(documents=[Document(text='# warm up\n\nHello.')])"

# Must match defaultPort in the Container class (src/index.js).
EXPOSE 8080

# 0.0.0.0, not 127.0.0.1: requests arrive from outside the container's loopback.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

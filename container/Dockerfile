# The image Cloudflare runs. wrangler builds it locally with Docker on
# `wrangler deploy` and pushes it to Cloudflare's registry.

# slim, not full: a smaller image boots faster, and boot time is the cold start.
# No --platform here: wrangler already builds for linux/amd64.
FROM python:3.12-slim

# uv installs the exact versions in uv.lock. Pinned to the version used locally.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv

WORKDIR /app

# BRIDGE_URL: the host the Container class intercepts (src/index.js) — the default
# in app/settings.py is the local `wrangler dev` bridge.
#
# The container runs with no internet access (enableInternet = false), so anything
# LlamaIndex would download on first use — tiktoken vocabularies, NLTK data — is
# fetched at build time (warm_up below) into these fixed paths.
ENV BRIDGE_URL=http://bridge.internal \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    LLAMA_INDEX_CACHE_DIR=/app/.cache/llama_index \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

# Dependencies first, code second, so editing the app doesn't reinstall everything.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

COPY app ./app
RUN python -c "from app.llama.engine import warm_up; warm_up()"

# Must match defaultPort in the Container class (src/index.js).
EXPOSE 8080

# Straight from the venv, not `uv run`: starts faster and never touches the network.
# 0.0.0.0, not 127.0.0.1: requests arrive from outside the container's loopback.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

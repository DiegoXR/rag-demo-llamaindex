"""rag-demo-llamaindex — the Python half. For now, a FastAPI hello world.

This file knows nothing about Cloudflare: it is a plain web server on port 8080.
The Worker in ../../src/index.js is the only way to reach it.
"""

import platform
import time

from fastapi import FastAPI

app = FastAPI()

# Set on every cold start: uptime near 0 means this request woke the container.
STARTED_AT = time.time()


@app.get("/")
def hello():
    return {
        "message": "Hello world from rag-demo-llamaindex",
        "python": platform.python_version(),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }


@app.get("/health")
def health():
    return {"ok": True}

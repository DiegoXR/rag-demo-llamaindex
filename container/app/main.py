"""rag-demo-llamaindex — the Python half: FastAPI serving the demo contract.

Endpoints (the shared contract every demo implements, plus document management):
  POST   /chat                  SSE: token… sources done | error
  POST   /documents?title=…     markdown body -> indexed into Vectorize, kept in R2
  GET    /documents             list of indexed documents
  DELETE /documents/{doc_id}
  GET    /health                cheap, never calls the LLM
  GET    /meta

This app knows nothing about Cloudflare beyond the bridge URL. The Worker in
../../src/index.js is the only way to reach it, and it has already checked the
hub's token by the time a request arrives here.
"""

import asyncio
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app import bridge_client
from app.settings import (
    MAX_DOC_BYTES,
    MAX_DOCS,
    MAX_INPUT_CHARS,
    MAX_MESSAGES,
    MAX_TITLE_CHARS,
    TIMEOUT_S,
)

app = FastAPI()

# LlamaIndex (and its OpenAI client) import in a couple of seconds; loading it on
# the first real request keeps /health fast right after a cold start.
_engine = None


def engine():
    global _engine
    if _engine is None:
        from app.llama import engine as loaded

        _engine = loaded
    return _engine


_executor = ThreadPoolExecutor(max_workers=4)


# --- Health and metadata --------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/meta")
def meta():
    return {
        "id": "rag-demo-llamaindex",
        "title": "LlamaIndex over your Markdown",
        "description": "Upload .md documents and chat with them. A Router Query Engine picks "
        "vector search for specific questions or a summary over every document.",
        "stack": ["Python", "FastAPI", "LlamaIndex", "Workers AI", "Vectorize", "R2", "Cloudflare Containers"],
        "repo": "https://github.com/DiegoXR/rag-demo-llamaindex",
    }


# --- Documents ------------------------------------------------------------------


def slugify(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:50].strip("-")


@app.post("/documents", status_code=201)
async def upload_document(request: Request, title: str):
    title = title.strip()
    if not 1 <= len(title) <= MAX_TITLE_CHARS:
        raise HTTPException(400, f"title must be 1–{MAX_TITLE_CHARS} characters")
    doc_id = slugify(title)
    if not doc_id:
        raise HTTPException(400, "title must contain letters or digits")

    body = await request.body()
    if not body or len(body) > MAX_DOC_BYTES:
        raise HTTPException(413, f"document must be 1–{MAX_DOC_BYTES} bytes")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "document must be UTF-8 markdown")

    existing = {d["doc_id"] for d in bridge_client.list_docs()}
    if doc_id not in existing and len(existing) >= MAX_DOCS:
        raise HTTPException(409, f"at most {MAX_DOCS} documents; delete one first")

    chunks = await _run(engine().ingest, doc_id, title, text)
    return {"doc_id": doc_id, "title": title, "chunks": chunks, "replaced": doc_id in existing}


@app.get("/documents")
def list_documents():
    return {"documents": bridge_client.list_docs()}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not re.fullmatch(r"[a-z0-9-]{1,50}", doc_id):
        raise HTTPException(404, "not found")
    if not await _run(engine().delete, doc_id):
        raise HTTPException(404, "not found")
    return {"deleted": doc_id}


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_executor, fn, *args)


# --- Chat -----------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    sessionId: str | None = None


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/chat")
def chat(body: ChatRequest):
    if not body.messages or body.messages[-1].role != "user":
        raise HTTPException(400, "the last message must come from the user")
    if len(body.messages) > MAX_MESSAGES:
        return JSONResponse({"error": f"at most {MAX_MESSAGES} messages"}, status_code=413)
    if sum(len(m.content) for m in body.messages) > MAX_INPUT_CHARS:
        return JSONResponse({"error": f"at most {MAX_INPUT_CHARS} characters"}, status_code=413)

    # The router answers one question; earlier turns are not sent to the LLM yet.
    question = body.messages[-1].content
    return StreamingResponse(_answer(question), media_type="text/event-stream")


def _answer(question: str):
    deadline = time.monotonic() + TIMEOUT_S
    try:
        router, tokens = engine().build_query_engine()
        # Routing + retrieval happen before the first token; bound them too.
        response = _executor.submit(router.query, question).result(timeout=TIMEOUT_S)

        if hasattr(response, "response_gen"):
            for text in response.response_gen:
                if time.monotonic() > deadline:
                    yield sse({"type": "error", "message": "The answer took too long"})
                    return
                yield sse({"type": "token", "text": text})
        else:
            yield sse({"type": "token", "text": str(response)})

        sources = [
            {
                "title": node.metadata.get("title", ""),
                "snippet": node.get_content()[:300],
                "score": round(node.score, 3) if node.score is not None else None,
            }
            for node in response.source_nodes
        ]
        yield sse({"type": "sources", "sources": sources})
        yield sse(
            {
                "type": "done",
                "usage": {
                    "input_tokens": tokens.prompt_llm_token_count,
                    "output_tokens": tokens.completion_llm_token_count,
                },
            }
        )
    except FutureTimeout:
        yield sse({"type": "error", "message": "The answer took too long"})
    except Exception as e:  # never leak details (keys never reach here, but stay terse)
        print(json.dumps({"chat_error": type(e).__name__, "detail": str(e)[:300]}))
        yield sse({"type": "error", "message": "The demo could not answer right now"})

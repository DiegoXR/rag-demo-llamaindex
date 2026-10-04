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
from typing import Literal

import httpx
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


def log(event: str, **fields) -> None:
    """One JSON line per event: the container's stdout lands in Workers Logs."""
    print(json.dumps({"event": event, **fields}, ensure_ascii=False))


def _user(request: Request) -> str:
    # Set by the hub (the Access email); for attribution only — the hub enforces quotas.
    return request.headers.get("x-hub-user", "unknown")


# The bridge (Workers AI, Vectorize, R2) failing is an upstream failure: answer 502
# with a readable message instead of FastAPI's bare 500.
@app.exception_handler(httpx.HTTPError)
async def bridge_failed(request: Request, exc: httpx.HTTPError):
    log("bridge_error", path=request.url.path, error=type(exc).__name__, detail=str(exc)[:300])
    return JSONResponse({"detail": "A storage or model service failed; try again."}, status_code=502)


# --- Health and metadata --------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/meta")
def meta():
    return {
        "id": "rag-demo-llamaindex",
        "title": "LlamaIndex over your Markdown",
        "description": "Upload .md documents and chat with them. A LlamaIndex agent remembers "
        "the conversation and picks vector search for specific questions or a summary over "
        "every document.",
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

    existing = {d["doc_id"] for d in await asyncio.to_thread(bridge_client.list_docs)}
    if doc_id not in existing and len(existing) >= MAX_DOCS:
        raise HTTPException(409, f"at most {MAX_DOCS} documents; delete one first")

    started = time.monotonic()
    chunks = await asyncio.to_thread(engine().ingest, doc_id, title, text)
    replaced = doc_id in existing
    log("ingest", user=_user(request), doc_id=doc_id, bytes=len(body), chunks=chunks,
        replaced=replaced, ms=int((time.monotonic() - started) * 1000))
    return {"doc_id": doc_id, "title": title, "chunks": chunks, "replaced": replaced}


@app.get("/documents")
def list_documents():
    return {"documents": bridge_client.list_docs()}


@app.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    if not re.fullmatch(r"[a-z0-9-]{1,50}", doc_id):
        raise HTTPException(404, "not found")
    if not await asyncio.to_thread(engine().delete, doc_id):
        raise HTTPException(404, "not found")
    log("delete", user=_user(request), doc_id=doc_id)
    return {"deleted": doc_id}


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
async def chat(request: Request, body: ChatRequest):
    if not body.messages or body.messages[-1].role != "user":
        raise HTTPException(400, "the last message must come from the user")
    if len(body.messages) > MAX_MESSAGES:
        return JSONResponse({"error": f"at most {MAX_MESSAGES} messages"}, status_code=413)
    if sum(len(m.content) for m in body.messages) > MAX_INPUT_CHARS:
        return JSONResponse({"error": f"at most {MAX_INPUT_CHARS} characters"}, status_code=413)
    return StreamingResponse(_answer(body.messages, _user(request)), media_type="text/event-stream")


def _sources(nodes) -> list[dict]:
    """Search results keep one entry per chunk, with its score; a summary read
    every chunk of every document, so it is listed once per document instead."""
    out, summarized = [], set()
    for n in nodes:
        title = n.node.metadata.get("title", "")
        if n.score is None:
            if title in summarized:
                continue
            summarized.add(title)
        out.append({
            "title": title,
            "snippet": n.node.get_content()[:300],
            "score": round(n.score, 3) if n.score is not None else None,
        })
    return out


async def _answer(messages: list[Message], user: str):
    from llama_index.core.agent.workflow import AgentStream
    from llama_index.core.llms import ChatMessage

    started = time.monotonic()
    # Off the event loop: the first build reads every document from R2 for the summary.
    run = await asyncio.to_thread(engine().build_agent)
    history = [ChatMessage(role=m.role, content=m.content) for m in messages[:-1]]
    handler = run.agent.run(
        user_msg=messages[-1].content,
        chat_history=history,
        max_iterations=engine().MAX_AGENT_ITERATIONS,
        early_stopping_method="generate",
    )
    status = "ok"
    try:
        async with asyncio.timeout(TIMEOUT_S):
            async for event in handler.stream_events():
                # Text deltas of the final answer. While the agent is only choosing a
                # tool, deltas are empty.
                if isinstance(event, AgentStream) and event.delta:
                    yield sse({"type": "token", "text": event.delta})
            await handler

        yield sse({"type": "sources", "sources": _sources(run.sources)})
        yield sse({
            "type": "done",
            "usage": {
                "input_tokens": run.tokens.prompt_llm_token_count,
                "output_tokens": run.tokens.completion_llm_token_count,
            },
        })
    except TimeoutError:
        status = "timeout"
        yield sse({"type": "error", "message": "The answer took too long"})
    except Exception as e:  # terse to the caller; details go to the logs
        status = "error"
        log("chat_error", user=user, error=type(e).__name__, detail=str(e)[:300])
        yield sse({"type": "error", "message": "The demo could not answer right now"})
    finally:
        # Timeout, error, or the hub aborting (the user left): stop the agent so an
        # answer nobody will read stops spending tokens.
        if not handler.done():
            status = "cancelled" if status == "ok" else status
            await handler.cancel_run()
        log("chat", user=user, status=status, messages=len(messages), tools=run.tools_used,
            sources=len(run.sources), input_tokens=run.tokens.prompt_llm_token_count,
            output_tokens=run.tokens.completion_llm_token_count,
            ms=int((time.monotonic() - started) * 1000))

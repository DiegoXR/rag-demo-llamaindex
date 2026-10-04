"""HTTP client for the bridge (src/bridge.js): Workers AI, Vectorize and R2.

Inside Cloudflare, requests to http://bridge.internal never leave the machine: the
Worker intercepts them and answers with its bindings. Nothing here holds a key.
"""

from dataclasses import dataclass
from urllib.parse import quote, unquote

import httpx

from app.settings import BRIDGE_URL, TIMEOUT_S

_client = httpx.Client(base_url=BRIDGE_URL, timeout=TIMEOUT_S)


@dataclass
class StoredDoc:
    doc_id: str
    title: str
    chunks: int
    text: str


def _ok(response: httpx.Response) -> httpx.Response:
    response.raise_for_status()
    return response


def embed(texts: list[str]) -> list[list[float]]:
    return _ok(_client.post("/embed", json={"texts": texts})).json()["vectors"]


def upsert_vectors(vectors: list[dict]) -> None:
    # Vectorize accepts up to 1000 vectors per call.
    for i in range(0, len(vectors), 500):
        _ok(_client.post("/vectors/upsert", json={"vectors": vectors[i : i + 500]}))


def query_vectors(vector: list[float], top_k: int, filter: dict | None = None) -> list[dict]:
    body = {"vector": vector, "topK": top_k, "filter": filter}
    return _ok(_client.post("/vectors/query", json=body)).json()["matches"]


def delete_vectors(ids: list[str]) -> None:
    for i in range(0, len(ids), 500):
        _ok(_client.post("/vectors/delete", json={"ids": ids[i : i + 500]}))


def list_docs() -> list[dict]:
    docs = _ok(_client.get("/docs")).json()["docs"]
    for doc in docs:
        doc["title"] = unquote(doc["title"])
    return docs


def get_doc(doc_id: str) -> StoredDoc | None:
    response = _client.get(f"/docs/{doc_id}")
    if response.status_code == 404:
        return None
    _ok(response)
    return StoredDoc(
        doc_id=doc_id,
        title=unquote(response.headers.get("x-doc-title", "")),
        chunks=int(response.headers.get("x-doc-chunks", "0")),
        text=response.text,
    )


def put_doc(doc_id: str, title: str, chunks: int, text: str) -> None:
    # Percent-encoded: HTTP header values cannot carry accents as-is.
    headers = {"x-doc-title": quote(title), "x-doc-chunks": str(chunks), "content-type": "text/markdown"}
    _ok(_client.put(f"/docs/{doc_id}", content=text.encode("utf-8"), headers=headers))


def delete_doc(doc_id: str) -> None:
    _ok(_client.delete(f"/docs/{doc_id}"))

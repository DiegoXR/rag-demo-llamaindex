"""LlamaIndex vector store backed by Cloudflare Vectorize, through the bridge.

Each chunk is one Vectorize vector. Its text and title travel in the vector's
metadata, so no separate docstore is needed: a query returns everything required
to build the answer and its sources.

Vector ids are deterministic, "<doc_id>:<n>", so a document's vectors can be
deleted knowing only its id and chunk count (kept with the original in R2) —
Vectorize has no delete-by-metadata.
"""

from typing import Any

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from app import bridge_client
from app.settings import MAX_METADATA_TEXT_BYTES


def vector_ids(doc_id: str, chunks: int) -> list[str]:
    return [f"{doc_id}:{n}" for n in range(chunks)]


class VectorizeVectorStore(BasePydanticVectorStore):
    stores_text: bool = True

    @property
    def client(self) -> None:
        return None

    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        vectors = []
        for node in nodes:
            text = node.get_content().encode("utf-8")[:MAX_METADATA_TEXT_BYTES].decode("utf-8", "ignore")
            vectors.append(
                {
                    "id": node.node_id,
                    "values": node.get_embedding(),
                    "metadata": {
                        "doc_id": node.metadata["doc_id"],
                        "title": node.metadata["title"],
                        "text": text,
                    },
                }
            )
        bridge_client.upsert_vectors(vectors)
        return [v["id"] for v in vectors]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Deletes a document's vectors. Needs `chunks`, the count stored in R2."""
        bridge_client.delete_vectors(vector_ids(ref_doc_id, delete_kwargs.get("chunks", 0)))

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.filters:
            raise NotImplementedError("VectorizeVectorStore does not support metadata filters")
        matches = bridge_client.query_vectors(query.query_embedding, query.similarity_top_k)
        nodes, scores, ids = [], [], []
        for match in matches:
            meta = match.get("metadata") or {}
            nodes.append(
                TextNode(
                    id_=match["id"],
                    text=meta.get("text", ""),
                    metadata={"doc_id": meta.get("doc_id", ""), "title": meta.get("title", "")},
                )
            )
            scores.append(match["score"])
            ids.append(match["id"])
        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

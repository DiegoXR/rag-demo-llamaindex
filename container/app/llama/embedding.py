"""LlamaIndex embedding model backed by Workers AI, through the bridge.

The model itself is chosen by the Worker (EMBED_MODEL in wrangler.jsonc); this
class only sends text and receives vectors. The same model embeds documents and
questions, which is what makes their vectors comparable.
"""

import asyncio

from llama_index.core.base.embeddings.base import BaseEmbedding

from app import bridge_client


class WorkersAIEmbedding(BaseEmbedding):
    model_name: str = "workers-ai-via-bridge"

    def _get_text_embedding(self, text: str) -> list[float]:
        return bridge_client.embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return bridge_client.embed(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        return bridge_client.embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

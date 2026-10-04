"""The RAG pipeline: ingesting documents and answering questions with LlamaIndex.

Ingest:  markdown -> MarkdownNodeParser (by heading) -> SentenceSplitter
         -> Workers AI embeddings -> Vectorize; the original goes to R2.
Answer:  a RouterQueryEngine picks one of two tools for each question —
           "vector":  top-k chunks from Vectorize, for specific questions;
           "summary": every document read from R2, for "summarize ..." requests.
"""

import threading

import tiktoken
from llama_index.core import Document, SummaryIndex, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

from app import bridge_client
from app.llama.embedding import WorkersAIEmbedding
from app.llama.vectorize_store import VectorizeVectorStore, vector_ids
from app.settings import (
    BRIDGE_URL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LLM_MODEL,
    MAX_SUMMARY_CHARS,
    MAX_TOKENS,
    TIMEOUT_S,
    TOP_K,
)

embed_model = WorkersAIEmbedding()
vector_store = VectorizeVectorStore()

SYSTEM_PROMPT = (
    "You answer questions about the documents provided as context. "
    "Answer in the language of the question. If the context does not contain the "
    "answer, say so instead of guessing."
)


def _splitters():
    return [MarkdownNodeParser(), SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)]


# --- Ingest -------------------------------------------------------------------


def ingest(doc_id: str, title: str, text: str) -> int:
    """Indexes one document, replacing any previous version with the same id."""
    previous = bridge_client.get_doc(doc_id)

    document = Document(id_=doc_id, text=text, metadata={"doc_id": doc_id, "title": title})
    # Keep doc_id out of what gets embedded; the title stays in, it helps retrieval.
    document.excluded_embed_metadata_keys = ["doc_id"]
    nodes = IngestionPipeline(transformations=[*_splitters(), embed_model]).run(documents=[document])
    for n, node in enumerate(nodes):
        node.id_ = f"{doc_id}:{n}"

    vector_store.add(nodes)
    # A shorter new version leaves stale tail vectors behind; remove them.
    if previous and previous.chunks > len(nodes):
        bridge_client.delete_vectors(vector_ids(doc_id, previous.chunks)[len(nodes) :])
    bridge_client.put_doc(doc_id, title, len(nodes), text)
    _invalidate_summary()
    return len(nodes)


def delete(doc_id: str) -> bool:
    stored = bridge_client.get_doc(doc_id)
    if stored is None:
        return False
    vector_store.delete(doc_id, chunks=stored.chunks)
    bridge_client.delete_doc(doc_id)
    _invalidate_summary()
    return True


# --- Summary index, cached until the documents change ------------------------

_summary_lock = threading.Lock()
_summary_nodes = None


def _invalidate_summary() -> None:
    global _summary_nodes
    with _summary_lock:
        _summary_nodes = None


def _summary_index_nodes():
    """Nodes for the summary engine: every document from R2, up to MAX_SUMMARY_CHARS."""
    global _summary_nodes
    with _summary_lock:
        if _summary_nodes is None:
            documents, budget = [], MAX_SUMMARY_CHARS
            for meta in bridge_client.list_docs():
                if budget <= 0:
                    break
                stored = bridge_client.get_doc(meta["doc_id"])
                if stored is None:
                    continue
                text = stored.text[:budget]
                budget -= len(text)
                documents.append(
                    Document(text=text, metadata={"doc_id": stored.doc_id, "title": stored.title})
                )
            _summary_nodes = IngestionPipeline(transformations=_splitters()).run(documents=documents)
        return _summary_nodes


# --- Answer -------------------------------------------------------------------


def build_query_engine():
    """A router over both tools, with its own token counter.

    Built per request (it is cheap: no data is loaded except the cached summary
    nodes) so concurrent requests never share a token count.
    """
    token_counter = TokenCountingHandler(tokenizer=tiktoken.encoding_for_model(LLM_MODEL).encode)
    callbacks = CallbackManager([token_counter])
    llm = OpenAI(
        model=LLM_MODEL,
        # The bridge adds the real key on its way to api.openai.com.
        api_base=f"{BRIDGE_URL}/openai",
        api_key="added-by-the-bridge",
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_S,
        max_retries=1,
        system_prompt=SYSTEM_PROMPT,
        callback_manager=callbacks,
    )

    vector_engine = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model, callback_manager=callbacks
    ).as_query_engine(llm=llm, similarity_top_k=TOP_K, streaming=True)

    summary_engine = SummaryIndex(
        _summary_index_nodes(), callback_manager=callbacks
    ).as_query_engine(llm=llm, response_mode="tree_summarize", streaming=True)

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[
            QueryEngineTool.from_defaults(
                query_engine=vector_engine,
                name="vector",
                description="Answers specific questions about details in the documents.",
            ),
            QueryEngineTool.from_defaults(
                query_engine=summary_engine,
                name="summary",
                description="Summarizes the documents or gives an overview of everything in them.",
            ),
        ],
        llm=llm,
    )
    # RouterQueryEngine resets the LLM's callback manager to the global default;
    # every engine above shares this one LLM object, so restoring it here is enough.
    llm.callback_manager = callbacks
    return router, token_counter

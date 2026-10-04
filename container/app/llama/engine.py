"""The RAG pipeline: ingesting documents and answering questions with LlamaIndex.

Ingest:  markdown -> MarkdownNodeParser (by heading) -> SentenceSplitter
         -> Workers AI embeddings -> Vectorize; the original goes to R2.
Answer:  a FunctionAgent with the conversation history and two tools —
           search_documents:    top-k chunks from Vectorize, for specific questions;
           summarize_documents: a SummaryIndex over every document read from R2.
         The agent decides which tool to call (or none, for a greeting), and
         rewrites follow-ups like "and for sale items?" using the history.
"""

import asyncio
import threading
from dataclasses import dataclass, field

import tiktoken
from llama_index.core import Document, SummaryIndex, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import FunctionTool
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
    "You are the assistant of a document collection uploaded by the user. "
    "Use search_documents for questions about specific facts, and summarize_documents "
    "when asked for a summary or an overview. Answer only from what the tools return; "
    "if they do not contain the answer, say so instead of guessing. "
    "Always answer in the language of the user's last message."
)

# Tool-call rounds per answer. One is the norm; the cap stops a confused agent from
# looping (and spending) — on reaching it the agent is forced to answer.
MAX_AGENT_ITERATIONS = 3


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


@dataclass
class AgentRun:
    """One answer's agent plus what its tools found, for the `sources` event and logs."""

    agent: FunctionAgent
    tokens: TokenCountingHandler
    sources: list[NodeWithScore] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


def build_agent() -> AgentRun:
    """A FunctionAgent over both tools, with its own token counter and sources list.

    Built per request (cheap: nothing is loaded except the cached summary nodes) so
    concurrent requests never share a token count or a list of sources.
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
        callback_manager=callbacks,
    )

    retriever = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model).as_retriever(
        similarity_top_k=TOP_K
    )
    summary_engine = SummaryIndex(_summary_index_nodes()).as_query_engine(
        llm=llm, response_mode="tree_summarize"
    )

    run = AgentRun(agent=None, tokens=token_counter)  # type: ignore[arg-type]

    # Tools return text for the agent and record their nodes as the answer's sources.
    # The work is blocking HTTP through the bridge, so it runs off the event loop.
    async def search_documents(query: str) -> str:
        """Search the documents for passages relevant to a specific question.

        `query` must be a standalone question, with any context from the
        conversation already filled in."""
        run.tools_used.append("search_documents")
        nodes = await asyncio.to_thread(retriever.retrieve, query)
        run.sources.extend(nodes)
        if not nodes:
            return "No documents matched."
        return "\n\n".join(f"[{n.node.metadata.get('title', '')}]\n{n.node.get_content()}" for n in nodes)

    async def summarize_documents(focus: str = "") -> str:
        """Summarize every uploaded document, optionally around a focus topic."""
        run.tools_used.append("summarize_documents")
        response = await asyncio.to_thread(
            summary_engine.query, f"Summarize the documents. Focus: {focus}" if focus else "Summarize the documents."
        )
        run.sources.extend(response.source_nodes)
        return str(response)

    run.agent = FunctionAgent(
        tools=[
            FunctionTool.from_defaults(async_fn=search_documents),
            FunctionTool.from_defaults(async_fn=summarize_documents),
        ],
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    # FunctionAgent resets the LLM's callback manager to the global default; the
    # agent and the summary engine share this one LLM, so restoring it here is enough.
    run.agent.llm.callback_manager = callbacks
    return run

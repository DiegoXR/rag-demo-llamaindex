# rag-demo-llamaindex

Upload Markdown documents and chat with them. A **LlamaIndex** agent, written in Python and running in a **Cloudflare Container**, remembers the conversation. It uses vector search for specific questions and a summary of every document for overviews. Every answer cites the passages it was grounded on.

This is one demo in a portfolio of RAG exercises. The live demo has **no public URL**: it is reachable only from the hub site's `/playground` page, which sits behind Cloudflare Access. Access is granted on request.

## Stack

Python 3.12 · FastAPI · LlamaIndex · OpenAI (`gpt-4o-mini`) · Workers AI embeddings (`@cf/baai/bge-m3`) · AI Gateway · Vectorize · R2 · Secrets Store · Cloudflare Containers · Durable Objects

## Architecture

```
hubworld (/playground, behind Cloudflare Access)
   │ service binding, x-hub-token
   ▼
Worker ─ gateway.js ──► RagContainer (Durable Object) ──► container: FastAPI + LlamaIndex :8080
   ▲                                                          │ http://resources.internal
   └──────────── resources.js ◄───────────────────────────────┘
                    │
                    ├─ AI Gateway ─┬─ Workers AI  (embeddings)
                    │              └─ OpenAI      (chat completions, key from the Secrets Store)
                    ├─ Vectorize    (vectors + chunk text)
                    └─ R2           (original documents)
```

The Worker has two roles, one module each:

- **Gateway** (`src/gateway.js`), the way in. It accepts only requests carrying the hub's `x-hub-token`, and rejects oversized uploads, *before* the container is woken. Then it forwards the request to the container's single instance.
- **Resource provider** (`src/resources.js`), the way out. Workers AI, Vectorize, R2 and the Secrets Store exist only as Worker bindings. When the container calls `http://resources.internal/...`, the Container class intercepts the request (`outboundByHost` in `src/container.js`), and the Worker answers it with its bindings. The container has **no internet access and no secrets**: the OpenAI key is added on the way out, and only `chat/completions` with the configured model is allowed through.

Every model call, embeddings and chat alike, goes through **AI Gateway** (`AI_GATEWAY_ID`). It logs each call with its tokens and cost, and rate-limits them. The calls use the `AI` binding, which authenticates to the gateway on its own, so the gateway can require authentication without the Worker holding a gateway token.

The container stores nothing. Vectors and chunk text live in Vectorize, and the original documents live in R2. It can sleep after 5 idle minutes without losing anything.

## How it answers

- **Ingest (`POST /documents`):** Markdown → `MarkdownNodeParser` → `SentenceSplitter` (512/50) → Workers AI embeddings → Vectorize. Vector ids are deterministic (`<doc_id>:<n>`), so a document can be replaced or deleted knowing only its id and chunk count, which are kept in R2.
- **Chat (`POST /chat`):** a `FunctionAgent` with the conversation history and two tools:
  - `search_documents` retrieves the top 4 chunks from Vectorize.
  - `summarize_documents` runs a `SummaryIndex` over every document.

  The answer streams back as Server-Sent Events: `token…`, then `sources`, then `done` with token usage.

## API

| Route | |
|---|---|
| `POST /chat` | `{messages: [{role: "user"\|"assistant", content}]}` → SSE `token` … `sources` · `done` \| `error` |
| `POST /documents?title=` | Markdown body (≤ 200 KB) → `201 {doc_id, title, chunks, replaced}` |
| `GET /documents` | `{documents: [{doc_id, title, chunks, size, uploaded}]}` |
| `DELETE /documents/{doc_id}` | `{deleted}` |
| `GET /health` · `GET /meta` | readiness · description |

Cost limits are enforced inside the demo, whoever the caller is:

- ≤ 10 messages and 4,000 characters per request.
- 512 output tokens per answer.
- `top_k` 4.
- 3 agent rounds.
- A 30 s timeout per answer.
- At most 20 documents.
- One container instance.

## Project layout

```
src/
  index.js          wires the modules; routes local dev requests to the provider
  gateway.js        the way in: token, size check, forward to the container
  container.js      RagContainer: port, sleep, no internet, outbound interception
  resources.js      the way out: Workers AI and OpenAI (via AI Gateway), Vectorize, R2
container/
  Dockerfile        python:3.12-slim + uv; downloads tokenizer data at build time
  app/main.py       FastAPI routes and the SSE stream
  app/llama/        LlamaIndex pipeline, Vectorize vector store, Workers AI embeddings
  app/resources_client.py   HTTP client for the resource provider
  app/settings.py   every limit and model name
  scripts/chat.py   terminal chat client for local development (not in the image)
wrangler.example.jsonc  template for wrangler.jsonc: container, Durable Object, bindings,
                        secrets — no public URL
```

## Set up your own copy

The Cloudflare configuration is published as a template, without account or Secrets Store IDs. Copy it and fill in your own:

```bash
cp wrangler.example.jsonc wrangler.jsonc     # git-ignored: your IDs stay on your machine
npx wrangler secrets-store store list --remote   # your Secrets Store ID → replace <YOUR_SECRETS_STORE_ID>
```

Optionally, uncomment `account_id` and set it to yours (`npx wrangler whoami`).

## Run locally

Containers do not run under `wrangler dev` on Windows, so locally the two halves run as separate processes. FastAPI runs on uvicorn, and the Worker only serves the resource provider. The Python code is the same; only `RESOURCES_URL` changes.

Prerequisites:

- Node.js and [uv](https://docs.astral.sh/uv/).
- `npx wrangler login`.
- The OpenAI key in the **local** Secrets Store, which `wrangler dev` cannot read from production. Create it once:

  ```bash
  npx wrangler secrets-store secret create <YOUR_SECRETS_STORE_ID> --name OPENAI_TOKEN --scopes workers
  ```

Then, in three terminals:

```bash
npm install
npm run dev:resources   # Worker on :8787, lends its bindings at /__resources
npm run dev:api         # FastAPI + LlamaIndex on :8080 (uvicorn --reload)
npm run chat            # terminal chat client: /docs, /upload <file.md>, /delete <id>, /reset
```

Local development uses the **real** Workers AI, Vectorize and R2 (`remote: true`), so documents uploaded locally are the same ones the live demo sees.

## Deploy

Requirements:

- Docker running.
- A wrangler login allowed to deploy containers.
- These resources, already existing in the account:
  - A Vectorize index `rag-demo-llamaindex` with 1024 dimensions, to match `bge-m3`.
  - An R2 bucket `rag-demo-llamaindex-docs`.
  - The Secrets Store secrets `OPENAI_TOKEN` and `HUB_TOKEN`.
  - An AI Gateway `rag-demo-llamaindex`, created on the dashboard (AI → AI Gateway), with logs and a rate limit.

```bash
npm run deploy   # builds the image, pushes it, deploys the Worker
npm run tail     # live logs, including the container's JSON lines
```

## License

[MIT](LICENSE)

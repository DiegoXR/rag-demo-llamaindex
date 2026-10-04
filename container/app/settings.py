"""Every limit and model name in one place.

The cost limits are fixed here on purpose: the demo never trusts the caller to be
reasonable (see "Cost limits inside the demo" in the portfolio plan).
"""

import os

# The bridge: the Worker intercepts requests to this host and serves them with its
# bindings (src/bridge.js). Locally it points at `wrangler dev` instead.
BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://bridge.internal").rstrip("/")

# Passed by the Worker at start time from wrangler.jsonc `vars`, so a model change
# never needs a new image.
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Chat limits.
MAX_TOKENS = 512          # output cap per answer
TOP_K = 4                 # chunks retrieved per question
MAX_INPUT_CHARS = 4_000   # total characters across the messages sent
MAX_MESSAGES = 10         # messages per request
TIMEOUT_S = 30            # whole answer, retrieval included

# Document limits.
MAX_DOC_BYTES = 200_000   # same value checked by the Worker before waking the container
MAX_DOCS = 20             # documents kept at once
MAX_TITLE_CHARS = 100
CHUNK_SIZE = 512          # tokens per chunk sent to the embedding model
CHUNK_OVERLAP = 50

# The summary engine reads whole documents; this caps what one summary can cost.
MAX_SUMMARY_CHARS = 60_000

# Vectorize allows 10 KiB of metadata per vector; the chunk text travels there.
MAX_METADATA_TEXT_BYTES = 8_000

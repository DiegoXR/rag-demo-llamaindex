// rag-demo-llamaindex — the Worker half: the only way into the Python container.
//
// The container has no public address, and neither does this Worker
// (workers_dev: false). hubworld reaches it through a service binding; the Worker
// checks the hub's token, then asks a Durable Object (the Container class below)
// for an instance and forwards the request. If the instance is asleep,
// `container.fetch()` is what wakes it.

import { Container } from '@cloudflare/containers';
import { handleBridge } from './bridge.js';

// Required by @cloudflare/containers for outbound interception (the bridge).
export { ContainerProxy } from '@cloudflare/containers';

const BRIDGE_HOST = 'bridge.internal';
const LOCAL_BRIDGE_PREFIX = '/__bridge';
// Same limit as MAX_DOC_BYTES in container/app/settings.py, checked here too so an
// oversized upload never wakes the container.
const MAX_DOC_BYTES = 200_000;

export class RagContainer extends Container {
	// The port uvicorn listens on inside the image — see container/Dockerfile.
	defaultPort = 8080;

	// Idle time before the container stops and billing stops.
	sleepAfter = '5m';

	// The container talks to nothing but the bridge below.
	enableInternet = false;

	// Plain config only — secrets stay in the Worker and are added by the bridge.
	envVars = { LLM_MODEL: this.env.LLM_MODEL };
}

// Assigned, not declared as `static outboundByHost = …`: a JS static field defines
// its own property and skips the library's setter, which is what registers the
// handler — the container would then get 520 "Origin is disallowed".
RagContainer.outboundByHost = {
	[BRIDGE_HOST]: (request, env) => handleBridge(request, env, new URL(request.url).pathname)
};

export default {
	async fetch(request, env) {
		const url = new URL(request.url);

		// Local development only: lets the FastAPI app running on the laptop use the
		// real bindings through `wrangler dev`. LOCAL_BRIDGE is passed only by the
		// `dev:bridge` script (--var), never in wrangler.jsonc, so it cannot exist
		// in production.
		if (env.LOCAL_BRIDGE === '1' && url.pathname.startsWith(LOCAL_BRIDGE_PREFIX)) {
			return handleBridge(request, env, url.pathname.slice(LOCAL_BRIDGE_PREFIX.length));
		}

		// Trust only the hub — checked before the container is touched, so a rejected
		// request never wakes it.
		const expected = await env.HUB_TOKEN.get();
		if (!expected || request.headers.get('x-hub-token') !== expected) {
			return Response.json({ error: 'unauthorized' }, { status: 401 });
		}

		if (url.pathname === '/documents' && request.method === 'POST') {
			const length = Number(request.headers.get('content-length') ?? 0);
			if (!length || length > MAX_DOC_BYTES) {
				return Response.json({ error: `document must be 1–${MAX_DOC_BYTES} bytes` }, { status: 413 });
			}
		}

		// Logged for attribution only; the hub enforces quotas.
		console.log(JSON.stringify({ path: url.pathname, user: request.headers.get('x-hub-user') }));

		// One named instance for everyone.
		const container = env.RAG_CONTAINER.getByName('main');
		return container.fetch(request);
	}
};

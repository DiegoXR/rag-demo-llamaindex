// rag-demo-llamaindex — the Worker half, in front of a Python container.
//
// The Worker has two roles, one module each:
//   gateway.js   — the way in: hubworld → token check → container
//   resources.js — the way out: container → Workers AI, Vectorize, R2, OpenAI
// and container.js describes the container itself. This file only wires them.
//
// Neither the Worker nor the container has a public address (workers_dev: false):
// hubworld reaches the Worker through a service binding.

import { handleGateway } from './gateway.js';
import { handleResourceRequest, LOCAL_RESOURCES_PREFIX } from './resources.js';

// Durable Object classes must be exported from the main module.
export { RagContainer } from './container.js';
// Required by @cloudflare/containers for outbound interception (outboundByHost).
export { ContainerProxy } from '@cloudflare/containers';

const LOCAL_HOSTNAMES = new Set(['localhost', '127.0.0.1']);

export default {
	async fetch(request, env) {
		const url = new URL(request.url);

		// Local development only: lets the FastAPI app running on the laptop use the
		// real bindings through `wrangler dev`. LOCAL_RESOURCES is passed only by the
		// `dev:resources` script (--var), never in wrangler.jsonc, so it cannot exist
		// in production. It is checked before the gateway's token, so the hostname must
		// be the laptop's too: deploying with the var by mistake (`wrangler deploy
		// --var`, a dashboard variable) still opens nothing — hubworld calls
		// https://demo/..., never localhost.
		if (
			env.LOCAL_RESOURCES === '1' &&
			LOCAL_HOSTNAMES.has(url.hostname) &&
			url.pathname.startsWith(LOCAL_RESOURCES_PREFIX)
		) {
			return handleResourceRequest(request, env, url.pathname.slice(LOCAL_RESOURCES_PREFIX.length));
		}

		return handleGateway(request, env);
	}
};

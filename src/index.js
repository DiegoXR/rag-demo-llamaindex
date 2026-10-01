// rag-demo-llamaindex — the Worker half: the only way into the Python container.
//
// The container has no public address, and neither does this Worker
// (workers_dev: false). hubworld reaches it through a service binding; the Worker
// asks a Durable Object (the Container class below) for an instance and forwards
// the request. If the instance is asleep, `container.fetch()` is what wakes it.

import { Container } from '@cloudflare/containers';

export class RagContainer extends Container {
	// The port uvicorn listens on inside the image — see container/Dockerfile.
	defaultPort = 8080;

	// Idle time before the container stops and billing stops.
	sleepAfter = '5m';
}

export default {
	async fetch(request, env) {
		// One named instance for everyone.
		const container = env.RAG_CONTAINER.getByName('main');
		return container.fetch(request);
	}
};

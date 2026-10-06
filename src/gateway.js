// The gateway: the only way into the container.
//
// hubworld reaches this Worker through a service binding; the gateway checks the
// hub's token, then asks the RagContainer Durable Object for an instance and
// forwards the request. If the instance is asleep, `container.fetch()` is what
// wakes it — so everything rejected here is rejected before the container runs.

// Same limit as MAX_DOC_BYTES in container/app/settings.py, checked here too so an
// oversized upload never wakes the container.
const MAX_DOC_BYTES = 200_000;

export async function handleGateway(request, env) {
	const url = new URL(request.url);

	// Trust only the hub — checked before the container is touched, so a rejected
	// request never wakes it.
	const expected = await env.HUB_TOKEN.get();
	if (!expected || request.headers.get('x-hub-token') !== expected) {
		// The only request the container never sees, so it is logged here. Accepted
		// requests are logged by the container itself, with more detail.
		console.log(JSON.stringify({ event: 'unauthorized', path: url.pathname, method: request.method }));
		return Response.json({ error: 'unauthorized' }, { status: 401 });
	}

	if (url.pathname === '/documents' && request.method === 'POST') {
		const length = Number(request.headers.get('content-length') ?? 0);
		if (!length || length > MAX_DOC_BYTES) {
			return Response.json({ error: `document must be 1–${MAX_DOC_BYTES} bytes` }, { status: 413 });
		}
	}

	// One named instance for everyone.
	const container = env.RAG_CONTAINER.getByName('main');
	return container.fetch(request);
}

// The resource provider: how the Python container reaches Cloudflare resources it
// cannot hold.
//
// Workers AI, Vectorize, R2 and the Secrets Store exist only as bindings of this
// Worker. The container calls http://resources.internal/..., the Container class
// intercepts that outbound request (outboundByHost in container.js) and hands it to
// handleResourceRequest() together with the Worker's env. No secret ever enters the
// container: the OpenAI key is added here, on the way out.
//
// Routes (paths are relative to the provider root):
//   POST   /embed            {texts: string[]}            -> {vectors: number[][]}
//   POST   /vectors/upsert   {vectors: [{id, values, metadata}]} -> {count}
//   POST   /vectors/query    {vector, topK}                -> {matches}
//   POST   /vectors/delete   {ids: string[]}               -> {count}
//   GET    /docs                                           -> {docs: [{doc_id, title, chunks, size, uploaded}]}
//   PUT    /docs/:id         markdown body, x-doc-title (percent-encoded), x-doc-chunks headers
//   GET    /docs/:id         -> markdown body
//   DELETE /docs/:id
//   POST   /openai/chat/completions  (model = LLM_MODEL only) -> OpenAI with the key added
//
// Every model call — embeddings and chat — goes through AI Gateway (AI_GATEWAY_ID in
// wrangler.jsonc): one place to see each call's tokens and cost, and to rate-limit.

// Where the container finds the provider in production — see container.js.
export const RESOURCES_HOST = 'resources.internal';
// Where the FastAPI app on the laptop finds it under `wrangler dev` — see index.js.
export const LOCAL_RESOURCES_PREFIX = '/__resources';

const EMBED_BATCH = 50;

export async function handleResourceRequest(request, env, path) {
	const { method } = request;

	if (path.startsWith('/openai/')) return proxyOpenAI(request, env, path.slice('/openai'.length));

	if (method === 'POST' && path === '/embed') {
		const { texts } = await request.json();
		const vectors = [];
		for (let i = 0; i < texts.length; i += EMBED_BATCH) {
			const out = await env.AI.run(
				env.EMBED_MODEL,
				{ text: texts.slice(i, i + EMBED_BATCH) },
				{ gateway: { id: env.AI_GATEWAY_ID } }
			);
			vectors.push(...out.data);
		}
		return Response.json({ vectors });
	}

	if (method === 'POST' && path === '/vectors/upsert') {
		const { vectors } = await request.json();
		await env.VECTORIZE.upsert(vectors);
		return Response.json({ count: vectors.length });
	}

	if (method === 'POST' && path === '/vectors/query') {
		const { vector, topK } = await request.json();
		const result = await env.VECTORIZE.query(vector, { topK, returnMetadata: 'all' });
		return Response.json({ matches: result.matches });
	}

	if (method === 'POST' && path === '/vectors/delete') {
		const { ids } = await request.json();
		if (ids.length) await env.VECTORIZE.deleteByIds(ids);
		return Response.json({ count: ids.length });
	}

	if (path === '/docs' && method === 'GET') {
		const listed = await env.DOCS.list({ prefix: 'docs/', include: ['customMetadata'] });
		const docs = listed.objects.map((o) => ({
			doc_id: o.key.slice('docs/'.length, -'.md'.length),
			title: o.customMetadata?.title ?? '',
			chunks: Number(o.customMetadata?.chunks ?? 0),
			size: o.size,
			uploaded: o.uploaded
		}));
		return Response.json({ docs });
	}

	const docMatch = path.match(/^\/docs\/([a-z0-9-]{1,50})$/);
	if (docMatch) {
		const key = `docs/${docMatch[1]}.md`;
		if (method === 'PUT') {
			await env.DOCS.put(key, request.body, {
				httpMetadata: { contentType: 'text/markdown; charset=utf-8' },
				customMetadata: {
					title: request.headers.get('x-doc-title') ?? '',
					chunks: request.headers.get('x-doc-chunks') ?? '0'
				}
			});
			return Response.json({ ok: true });
		}
		if (method === 'GET') {
			const obj = await env.DOCS.get(key);
			if (!obj) return new Response('not found', { status: 404 });
			return new Response(obj.body, {
				headers: {
					'content-type': 'text/markdown; charset=utf-8',
					'x-doc-title': obj.customMetadata?.title ?? '',
					'x-doc-chunks': obj.customMetadata?.chunks ?? '0'
				}
			});
		}
		if (method === 'DELETE') {
			await env.DOCS.delete(key);
			return Response.json({ ok: true });
		}
	}

	return new Response('unknown resource route', { status: 404 });
}

// Forwards an OpenAI chat completion through AI Gateway, adding the key from the
// Secrets Store. The response (including an SSE stream) is returned as-is.
//
// The key can do far more than this demo needs (other models, files, images…), so
// only what the demo uses is let through: chat completions with LLM_MODEL. Even if
// code inside the container were compromised, it could not spend the key elsewhere.
async function proxyOpenAI(request, env, rest) {
	if (request.method !== 'POST' || rest !== '/chat/completions') {
		return Response.json({ error: `OpenAI route not allowed: ${request.method} ${rest}` }, { status: 403 });
	}
	let query;
	try {
		query = await request.json();
	} catch {
		return Response.json({ error: 'body must be JSON' }, { status: 400 });
	}
	if (query.model !== env.LLM_MODEL) {
		return Response.json({ error: `model not allowed: ${query.model}` }, { status: 403 });
	}

	// Through the binding, not fetch(): the call is already authenticated to the
	// gateway, so no AI Gateway token is needed. The gateway logs every call with
	// its tokens and cost, and applies the rate limit set on the dashboard.
	const key = await env.OPENAI_API_KEY.get();
	return env.AI.gateway(env.AI_GATEWAY_ID).run({
		provider: 'openai',
		endpoint: 'chat/completions',
		headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
		query
	});
}

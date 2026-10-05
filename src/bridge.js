// The bridge: how the Python container reaches Cloudflare bindings it cannot hold.
//
// Workers AI, Vectorize, R2 and the Secrets Store exist only as bindings of this
// Worker. The container calls http://bridge.internal/..., the Container class
// intercepts that outbound request (outboundByHost in index.js) and hands it to
// handleBridge() together with the Worker's env. No secret ever enters the
// container: the OpenAI key is added here, on the way out.
//
// Routes (paths are relative to the bridge root):
//   POST   /embed            {texts: string[]}            -> {vectors: number[][]}
//   POST   /vectors/upsert   {vectors: [{id, values, metadata}]} -> {count}
//   POST   /vectors/query    {vector, topK}                -> {matches}
//   POST   /vectors/delete   {ids: string[]}               -> {count}
//   GET    /docs                                           -> {docs: [{doc_id, title, chunks, size, uploaded}]}
//   PUT    /docs/:id         markdown body, x-doc-title (percent-encoded), x-doc-chunks headers
//   GET    /docs/:id         -> markdown body
//   DELETE /docs/:id
//   *      /openai/*         -> https://api.openai.com/v1/* with the key added

const EMBED_BATCH = 50;

export async function handleBridge(request, env, path) {
	const { method } = request;

	if (path.startsWith('/openai/')) return proxyOpenAI(request, env, path.slice('/openai'.length));

	if (method === 'POST' && path === '/embed') {
		const { texts } = await request.json();
		const vectors = [];
		for (let i = 0; i < texts.length; i += EMBED_BATCH) {
			const out = await env.AI.run(env.EMBED_MODEL, { text: texts.slice(i, i + EMBED_BATCH) });
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

	return new Response('unknown bridge route', { status: 404 });
}

// Forwards an OpenAI API call, adding the key from the Secrets Store. Only the
// headers OpenAI needs are passed on; the response (including an SSE stream) is
// returned as-is.
async function proxyOpenAI(request, env, rest) {
	const key = await env.OPENAI_API_KEY.get();
	const headers = new Headers({ authorization: `Bearer ${key}` });
	for (const name of ['content-type', 'accept']) {
		const value = request.headers.get(name);
		if (value) headers.set(name, value);
	}
	const url = new URL(request.url);
	return fetch(`https://api.openai.com/v1${rest}${url.search}`, {
		method: request.method,
		headers,
		body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body
	});
}

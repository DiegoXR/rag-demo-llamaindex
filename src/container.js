// The container: which image runs, on which port, for how long, and what it may
// reach. RagContainer is a Durable Object — the remote control that starts the
// container, forwards requests to it and puts it to sleep.

import { Container } from '@cloudflare/containers';
import { handleResourceRequest, RESOURCES_HOST } from './resources.js';

export class RagContainer extends Container {
	// The port uvicorn listens on inside the image — see container/Dockerfile.
	defaultPort = 8080;

	// Idle time before the container stops and billing stops.
	sleepAfter = '5m';

	// The container talks to nothing but the resource provider below.
	enableInternet = false;

	// Plain config only — secrets stay in the Worker and are added by the provider.
	envVars = { LLM_MODEL: this.env.LLM_MODEL };
}

// Assigned, not declared as `static outboundByHost = …`: a JS static field defines
// its own property and skips the library's setter, which is what registers the
// handler — the container would then get 520 "Origin is disallowed".
RagContainer.outboundByHost = {
	[RESOURCES_HOST]: (request, env) => handleResourceRequest(request, env, new URL(request.url).pathname)
};

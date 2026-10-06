export async function sha256(text: string): Promise<string> {
  const buffer = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buffer)].map(value => value.toString(16).padStart(2, '0')).join('');
}

export async function fetchReplayText(path: string, signal?: AbortSignal): Promise<string> {
  // No absolute URLs, traversal, queries or redirects; only same-origin replay assets.
  if (!/^\/replay-v01\/(manifest\.json|cases\/index\.json|cases\/case-0(?:0[1-9]|1[0-9]|2[0-2])\/(signals|prediction-windows|events)\.json)$/.test(path)) {
    throw new Error('Replay asset unavailable.');
  }
  signal?.throwIfAborted();
  const response = await fetch(path, { cache: 'no-cache', redirect: 'error', signal });
  if (!response.ok) throw new Error('Replay asset unavailable.');
  const text = await response.text();
  signal?.throwIfAborted();
  return text;
}

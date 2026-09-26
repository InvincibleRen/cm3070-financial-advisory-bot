

const BASE = "/api";

async function request(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") {
      url.searchParams.set(k, v);
    }
  });
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export function fetchUniverse() {
  return request(`${BASE}/universe`);
}

export function fetchPresets() {
  return request(`${BASE}/presets`);
}

export function fetchSelection({ tickers, start, top_n, model } = {}) {
  return request(`${BASE}/selection`, {
    tickers: tickers ? tickers.join(",") : undefined,
    start,
    top_n,
    model,
  });
}

export function fetchStock(ticker, { period } = {}) {
  return request(`${BASE}/stock/${encodeURIComponent(ticker)}`, { period });
}

export function fetchSentiment(ticker) {
  return request(`${BASE}/sentiment/${encodeURIComponent(ticker)}`);
}

export function fetchDirection(ticker, { horizon } = {}) {
  return request(`${BASE}/direction/${encodeURIComponent(ticker)}`, { horizon });
}

export function fetchCompare(ticker, { period } = {}) {
  return request(`${BASE}/compare/${encodeURIComponent(ticker)}`, { period });
}

export function fetchEvaluation(ticker, { period, train } = {}) {
  return request(`${BASE}/evaluate/${encodeURIComponent(ticker)}`, { period, train });
}

export function fetchEvidence({ signal } = {}) {
  return request(`${BASE}/evidence`, { signal });
}

export function fetchResearch() {
  return request(`${BASE}/research`);
}

export function fetchProfileGuidance(profile) {
  return request(`${BASE}/profile`, profile);
}

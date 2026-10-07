/* Shared auth helpers for multi-page site. */
window.KEOAuth = (() => {
  const TOKEN_KEY = "keo_pref_token_v2";

  function token() {
    return localStorage.getItem(TOKEN_KEY) || "";
  }

  function setToken(t) {
    localStorage.setItem(TOKEN_KEY, t);
  }

  function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
  }

  function headers(json = false) {
    const h = {};
    if (json) h["Content-Type"] = "application/json";
    const t = token();
    if (t) h.Authorization = `Bearer ${t}`;
    return h;
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.error || res.statusText || "request failed");
      err.status = res.status;
      throw err;
    }
    return data;
  }

  async function logout() {
    const t = token();
    if (t) {
      try {
        await fetch("/api/logout", { method: "POST", headers: headers() });
      } catch (_) {}
    }
    clearToken();
  }

  return { TOKEN_KEY, token, setToken, clearToken, headers, api, logout };
})();

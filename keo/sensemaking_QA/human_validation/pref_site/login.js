(() => {
  const { api, setToken, headers, token } = window.KEOAuth;
  const form = document.getElementById("login-form");
  const err = document.getElementById("login-error");

  if (token()) {
    api("/api/me", { headers: headers() })
      .then((me) => {
        location.href = me.role === "adjudicator" ? "adjudicate.html" : "rate.html";
      })
      .catch(() => {
        window.KEOAuth.clearToken();
      });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    err.hidden = true;
    const btn = form.querySelector('button[type="submit"]');
    const who = (form.querySelector('input[name="annotator"]:checked') || {}).value || "1";
    const pw = document.getElementById("password").value;
    btn.disabled = true;
    btn.textContent = "Signing in…";
    try {
      const data = await api("/api/login", {
        method: "POST",
        headers: headers(true),
        body: JSON.stringify({ annotator: who, password: pw }),
      });
      setToken(data.token);
      location.href = data.role === "adjudicator" ? "adjudicate.html" : "rate.html";
    } catch (ex) {
      err.textContent = ex.message || "Login failed";
      err.hidden = false;
      btn.disabled = false;
      btn.textContent = "Continue →";
    }
  });
})();

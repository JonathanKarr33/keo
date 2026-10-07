(() => {
  const { api, headers, token, logout, clearToken } = window.KEOAuth;
  const INDEX_KEY = "keo_adj_index_v1";

  if (!token()) {
    location.href = "index.html";
    return;
  }

  const progressLabel = document.getElementById("progress-label");
  const progressBar = document.getElementById("progress-bar");
  const saveFlash = document.getElementById("save-flash");
  const itemStatus = document.getElementById("item-status");
  const itemMeta = document.getElementById("item-meta");
  const questionEl = document.getElementById("question");
  const chipsA = document.getElementById("chips-a");
  const chipsB = document.getElementById("chips-b");
  const bodyA = document.getElementById("body-a");
  const bodyB = document.getElementById("body-b");
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  const autoNext = document.getElementById("auto-next");
  const noteEl = document.getElementById("note");
  const dots = document.getElementById("dots");
  const card = document.getElementById("card");
  const doneOverlay = document.getElementById("done-overlay");
  const gate = document.getElementById("gate");
  const adjMain = document.getElementById("adj-main");
  const prior1 = document.getElementById("prior-1");
  const prior2 = document.getElementById("prior-2");

  let items = [];
  let index = 0;
  let saving = false;

  function flash(msg) {
    saveFlash.textContent = msg;
    saveFlash.hidden = false;
    clearTimeout(flash._t);
    flash._t = setTimeout(() => {
      saveFlash.hidden = true;
    }, 1800);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function chipHtml(words) {
    if (!words || !words.length) return "<em>similar</em>";
    return words.map((w) => `<span class="chip">${escapeHtml(w)}</span>`).join(" ");
  }

  function openCount() {
    return items.filter((it) => !it.prior_pref).length;
  }

  function paintDots() {
    dots.innerHTML = items
      .map((it, i) => {
        const done = Boolean(it.prior_pref);
        const cls = ["dot", i === index ? "current" : "", done ? "done" : ""]
          .filter(Boolean)
          .join(" ");
        const title = `#${it.item_num}${done ? " · resolved" : " · open"}`;
        return `<button type="button" class="${cls}" data-i="${i}" title="${title}"></button>`;
      })
      .join("");
  }

  function paintProgress() {
    const open = openCount();
    const n = items.length;
    progressLabel.textContent = n ? `${open} open / ${n} contested` : "0 contested";
    progressBar.style.width = n ? `${(100 * (n - open)) / n}%` : "0%";
    paintDots();
    if (n && open === 0) doneOverlay.hidden = false;
    else doneOverlay.hidden = true;
  }

  function currentItem() {
    return items[index] || null;
  }

  function resolution() {
    const el = document.querySelector('input[name="resolution"]:checked');
    return (el && el.value) || "adjudicator";
  }

  function showItem() {
    const it = currentItem();
    if (!it) return;
    localStorage.setItem(INDEX_KEY, String(index));
    btnPrev.disabled = index <= 0;
    btnNext.disabled = index >= items.length - 1;

    card.classList.remove("enter");
    void card.offsetWidth;
    card.classList.add("enter");

    itemMeta.textContent = `Item ${it.item_num} · contested ${index + 1}/${items.length}`;
    questionEl.textContent = it.question || "";
    chipsA.innerHTML = chipHtml(it.chips_a);
    chipsB.innerHTML = chipHtml(it.chips_b);
    bodyA.innerHTML = it.answer_a_html || "";
    bodyB.innerHTML = it.answer_b_html || "";
    prior1.textContent = `Ann1: ${it.ann1_label || "—"}`;
    prior2.textContent = `Ann2: ${it.ann2_label || "—"}`;
    noteEl.value = it.prior_note || "";

    if (it.prior_resolution === "conference") {
      document.querySelector('input[name="resolution"][value="conference"]').checked = true;
    } else {
      document.querySelector('input[name="resolution"][value="adjudicator"]').checked = true;
    }

    const pref = it.prior_pref;
    if (pref) {
      itemStatus.textContent = pref === "neither" ? "neither" : `prefer ${pref}`;
      itemStatus.classList.add("rated");
    } else {
      itemStatus.textContent = "open";
      itemStatus.classList.remove("rated");
    }
    document.querySelectorAll(".vote button").forEach((btn) => {
      btn.classList.toggle("selected", Boolean(pref) && btn.dataset.pref === pref);
    });

    paintProgress();
    bodyA.scrollTop = 0;
    bodyB.scrollTop = 0;
  }

  function go(delta) {
    if (!items.length) return;
    index = Math.max(0, Math.min(items.length - 1, index + delta));
    showItem();
  }

  function nextOpen(afterIdx) {
    for (let i = afterIdx + 1; i < items.length; i++) {
      if (!items[i].prior_pref) return i;
    }
    for (let i = 0; i <= afterIdx; i++) {
      if (!items[i].prior_pref) return i;
    }
    return -1;
  }

  async function resolve(pref) {
    if (saving) return;
    const it = currentItem();
    if (!it) return;
    saving = true;
    document.querySelectorAll(".vote button").forEach((b) => {
      b.disabled = true;
    });
    try {
      const data = await api("/api/adjudicate", {
        method: "POST",
        headers: headers(true),
        body: JSON.stringify({
          item_num: it.item_num,
          preference_ab: pref,
          resolution: resolution(),
          note: noteEl.value.trim(),
        }),
      });
      it.prior_pref = pref;
      it.prior_resolution = resolution();
      it.prior_note = noteEl.value.trim();
      showItem();
      flash(`Saved · ${data.n_open_disagreements} open`);
      if (autoNext.checked) {
        const ni = nextOpen(index);
        if (ni >= 0 && ni !== index) {
          index = ni;
          showItem();
        }
      }
    } catch (err) {
      flash(err.message || "Save failed");
      if (err.status === 401) {
        await logout();
        location.href = "index.html";
      }
    } finally {
      saving = false;
      document.querySelectorAll(".vote button").forEach((b) => {
        b.disabled = false;
      });
    }
  }

  async function boot() {
    try {
      const me = await api("/api/me", { headers: headers() });
      if (me.role !== "adjudicator") {
        location.href = "rate.html";
        return;
      }
      const data = await api("/api/disagreements", { headers: headers() });
      items = data.items || [];

      if (!data.primaries_complete && items.length === 0) {
        gate.hidden = false;
        adjMain.hidden = true;
        document.getElementById("gate-title").textContent = "Waiting on primary raters";
        document.getElementById("gate-body").textContent =
          "Both Annotator 1 and 2 should finish all 40 items. Contested pairs appear here automatically.";
        return;
      }

      if (data.primaries_complete && items.length === 0) {
        gate.hidden = false;
        adjMain.hidden = true;
        document.getElementById("gate-title").textContent = "Full agreement";
        document.getElementById("gate-body").textContent =
          "Annotators 1 and 2 agree on every overlapping item — nothing to adjudicate.";
        return;
      }

      gate.hidden = true;
      adjMain.hidden = false;

      const saved = Number(localStorage.getItem(INDEX_KEY) || "0");
      index = Number.isFinite(saved) ? Math.max(0, Math.min(items.length - 1, saved)) : 0;
      const firstOpen = items.findIndex((it) => !it.prior_pref);
      if (firstOpen >= 0) index = firstOpen;
      showItem();
    } catch (err) {
      clearToken();
      location.href = "index.html";
    }
  }

  btnPrev.addEventListener("click", () => go(-1));
  btnNext.addEventListener("click", () => go(1));
  dots.addEventListener("click", (e) => {
    const btn = e.target.closest(".dot");
    if (!btn) return;
    index = Number(btn.dataset.i) || 0;
    showItem();
  });
  document.querySelectorAll(".vote button").forEach((btn) => {
    btn.addEventListener("click", () => resolve(btn.dataset.pref));
  });
  document.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA"].includes((e.target || {}).tagName)) return;
    if (e.key === "ArrowLeft") go(-1);
    if (e.key === "ArrowRight") go(1);
    if (e.key === "1") resolve("A");
    if (e.key === "2") resolve("neither");
    if (e.key === "3") resolve("B");
    if (e.key === "Escape" && !doneOverlay.hidden) doneOverlay.hidden = true;
  });
  document.getElementById("btn-logout").addEventListener("click", async () => {
    await logout();
    location.href = "index.html";
  });
  document.getElementById("btn-review").addEventListener("click", () => {
    doneOverlay.hidden = true;
    index = 0;
    showItem();
  });

  boot();
})();

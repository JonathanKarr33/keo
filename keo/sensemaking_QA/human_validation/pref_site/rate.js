(() => {
  const { api, headers, token, logout, clearToken } = window.KEOAuth;
  const INDEX_KEY = "keo_pref_index_v2";

  if (!token()) {
    location.href = "index.html";
    return;
  }

  const whoLabel = document.getElementById("who-label");
  const orderNote = document.getElementById("order-note");
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
  const syncScroll = document.getElementById("sync-scroll");
  const dots = document.getElementById("dots");
  const card = document.getElementById("card");
  const doneOverlay = document.getElementById("done-overlay");

  let annotator = null;
  let items = [];
  let votes = {};
  let index = 0;
  let voting = false;
  let syncing = false;

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

  function ratedCount() {
    return items.reduce((n, it) => n + (votes[String(it.item_num)] ? 1 : 0), 0);
  }

  function paintDots() {
    dots.innerHTML = items
      .map((it, i) => {
        const pref = votes[String(it.item_num)];
        const cls = [
          "dot",
          i === index ? "current" : "",
          pref ? "done" : "",
          pref === "neither" ? "neither" : "",
        ]
          .filter(Boolean)
          .join(" ");
        const title = pref
          ? `#${it.item_num} · ${pref === "neither" ? "neither" : "prefer " + pref}`
          : `#${it.item_num} · unrated`;
        return `<button type="button" class="${cls}" data-i="${i}" title="${title}" aria-label="${title}"></button>`;
      })
      .join("");
  }

  function paintProgress() {
    const n = ratedCount();
    progressLabel.textContent = `${n} / ${items.length}`;
    progressBar.style.width = `${(100 * n) / Math.max(items.length, 1)}%`;
    paintDots();
    if (n >= items.length && items.length) {
      doneOverlay.hidden = false;
    } else {
      doneOverlay.hidden = true;
    }
  }

  function currentItem() {
    return items[index] || null;
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

    const pref = votes[String(it.item_num)];
    itemMeta.textContent = `Item ${it.item_num} of ${items.length}`;
    questionEl.textContent = it.question || "";
    chipsA.innerHTML = chipHtml(it.chips_a);
    chipsB.innerHTML = chipHtml(it.chips_b);
    bodyA.innerHTML = it.answer_a_html || "";
    bodyB.innerHTML = it.answer_b_html || "";

    if (pref) {
      itemStatus.textContent = pref === "neither" ? "neither" : `prefer ${pref}`;
      itemStatus.classList.add("rated");
    } else {
      itemStatus.textContent = "unrated";
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
    index = Math.max(0, Math.min(items.length - 1, index + delta));
    showItem();
  }

  function nextUnrated(afterNum) {
    const start = items.findIndex((it) => it.item_num === afterNum);
    for (let i = start + 1; i < items.length; i++) {
      if (!votes[String(items[i].item_num)]) return i;
    }
    for (let i = 0; i <= start; i++) {
      if (!votes[String(items[i].item_num)]) return i;
    }
    return -1;
  }

  async function vote(pref) {
    if (voting) return;
    const it = currentItem();
    if (!it) return;
    voting = true;
    document.querySelectorAll(".vote button").forEach((b) => {
      b.disabled = true;
    });
    try {
      const data = await api("/api/vote", {
        method: "POST",
        headers: headers(true),
        body: JSON.stringify({ item_num: it.item_num, preference_ab: pref }),
      });
      votes[String(it.item_num)] = pref;
      showItem();
      flash(`Saved · ${data.n_rated}/${items.length}`);
      if (autoNext.checked) {
        const ni = nextUnrated(it.item_num);
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
      voting = false;
      document.querySelectorAll(".vote button").forEach((b) => {
        b.disabled = false;
      });
    }
  }

  function bindSyncScroll() {
    const sync = (src, dst) => {
      if (!syncScroll.checked || syncing) return;
      syncing = true;
      const maxS = src.scrollHeight - src.clientHeight;
      const maxD = dst.scrollHeight - dst.clientHeight;
      const ratio = maxS > 0 ? src.scrollTop / maxS : 0;
      dst.scrollTop = ratio * maxD;
      requestAnimationFrame(() => {
        syncing = false;
      });
    };
    bodyA.addEventListener("scroll", () => sync(bodyA, bodyB));
    bodyB.addEventListener("scroll", () => sync(bodyB, bodyA));
  }

  async function boot() {
    try {
      const me = await api("/api/me", { headers: headers() });
      if (me.role === "adjudicator") {
        location.href = "adjudicate.html";
        return;
      }
      const data = await api("/api/items", { headers: headers() });
      annotator = data.annotator;
      items = data.items || [];
      votes = data.votes || {};
      whoLabel.textContent = `Annotator ${annotator}`;
      orderNote.textContent =
        "Sides are randomized — systems stay hidden.";

      const saved = Number(localStorage.getItem(INDEX_KEY) || "0");
      index = Number.isFinite(saved) ? Math.max(0, Math.min(items.length - 1, saved)) : 0;
      const firstOpen = items.findIndex((it) => !votes[String(it.item_num)]);
      if (firstOpen >= 0) index = firstOpen;
      showItem();
      bindSyncScroll();
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
    btn.addEventListener("click", () => vote(btn.dataset.pref));
  });

  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes((e.target || {}).tagName)) return;
    if (e.key === "ArrowLeft") go(-1);
    if (e.key === "ArrowRight") go(1);
    if (e.key === "1") vote("A");
    if (e.key === "2") vote("neither");
    if (e.key === "3") vote("B");
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

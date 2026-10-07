(() => {
  const { api, headers, token, logout } = window.KEOAuth;
  if (!token()) {
    location.href = "index.html";
    return;
  }

  const CLASSES = [
    "catastrophic",
    "hazardous",
    "major",
    "minor",
    "none",
  ];
  const SHORT = {
    catastrophic: "Catastrophic",
    hazardous: "Hazardous",
    major: "Major",
    minor: "Minor",
    none: "No safety effect",
  };
  const DEFS = {
    catastrophic:
      "Prevents continued safe flight/landing; multiple fatalities or loss of aircraft plausible.",
    hazardous:
      "Large cut in safety margin; crew may be unable to perform tasks; serious injury plausible.",
    major:
      "Significant cut in margins or rise in crew workload; physical distress / injuries possible.",
    minor:
      "Slight cut in margins or slight workload increase; discomfort; routine crew response.",
    none:
      "No effect on safety, capability, or crew workload beyond ordinary inconvenience.",
  };

  let me = null;
  let items = [];
  let votes = {};
  let contested = [];
  let idx = 0; // index into `queue`
  let queue = [];

  const el = {
    who: document.getElementById("who-label"),
    progress: document.getElementById("progress-label"),
    bar: document.getElementById("progress-bar"),
    flash: document.getElementById("save-flash"),
    mode: document.getElementById("mode-label"),
    num: document.getElementById("item-num"),
    instructions: document.getElementById("instructions"),
    head: document.getElementById("head-text"),
    rel: document.getElementById("rel-text"),
    tail: document.getElementById("tail-text"),
    grid: document.getElementById("sev-grid"),
    adj: document.getElementById("adj-panel"),
    ann1: document.getElementById("ann1-class"),
    ann2: document.getElementById("ann2-class"),
    note: document.getElementById("adj-note"),
    err: document.getElementById("error"),
    prev: document.getElementById("btn-prev"),
    next: document.getElementById("btn-next"),
    logout: document.getElementById("btn-logout"),
  };

  function flash(msg) {
    el.flash.textContent = msg;
    el.flash.hidden = false;
    setTimeout(() => {
      el.flash.hidden = true;
    }, 1200);
  }

  function ratedCount() {
    return Object.values(votes).filter(Boolean).length;
  }

  function updateProgress() {
    const n = me && me.role === "adjudicator" ? contested.length : items.length;
    const r =
      me && me.role === "adjudicator"
        ? contested.filter((c) => votes[c]).length
        : ratedCount();
    el.progress.textContent = `${r} / ${n || 0}`;
    const pct = n ? Math.round((100 * r) / n) : 0;
    el.bar.style.width = `${pct}%`;
  }

  function current() {
    return queue[idx] || null;
  }

  function renderGrid(selected) {
    el.grid.innerHTML = CLASSES.map((c, i) => {
      const active = selected === c ? "active" : "";
      return `<button type="button" class="sev-btn ${active}" data-class="${c}" title="${DEFS[c]}">
        <kbd>${i + 1}</kbd>
        <strong>${SHORT[c]}</strong>
        <small class="sev-def">${DEFS[c]}</small>
      </button>`;
    }).join("");
    el.grid.querySelectorAll(".sev-btn").forEach((btn) => {
      btn.addEventListener("click", () => vote(btn.dataset.class));
    });
  }

  function render() {
    el.err.hidden = true;
    const it = current();
    if (!it) {
      el.num.textContent = "—";
      el.head.textContent = me && me.role === "adjudicator"
        ? "No contested severity items (or primaries incomplete)."
        : "No items.";
      el.rel.textContent = "";
      el.tail.textContent = "";
      el.adj.hidden = true;
      renderGrid(null);
      updateProgress();
      return;
    }
    const num = String(it.item_num);
    el.num.textContent = String(Number(num) + 1); // 1-based display
    el.head.textContent = it.head;
    el.rel.textContent = it.relation;
    el.tail.textContent = it.tail;
    if (me.role === "adjudicator") {
      el.adj.hidden = false;
      // fetch labels from status payload cached on vote objects — use API consensus if needed
      el.ann1.textContent = it._ann1 || "—";
      el.ann2.textContent = it._ann2 || "—";
    } else {
      el.adj.hidden = true;
    }
    renderGrid(votes[num] || null);
    updateProgress();
  }

  async function vote(sevClass) {
    const it = current();
    if (!it) return;
    const num = String(it.item_num);
    const body = {
      item_num: Number(num),
      severity_class: sevClass,
    };
    if (me.role === "adjudicator") {
      const res = document.querySelector('input[name="resolution"]:checked');
      body.resolution = (res && res.value) || "adjudicator";
      body.note = el.note.value || "";
    }
    try {
      const data = await api("/api/severity/vote", {
        method: "POST",
        headers: headers(true),
        body: JSON.stringify(body),
      });
      votes[num] = sevClass;
      flash("Saved");
      updateProgress();
      renderGrid(sevClass);
      if (me.role !== "adjudicator" && idx < queue.length - 1) {
        idx += 1;
        render();
      } else if (me.role === "adjudicator") {
        // drop resolved from local contested queue view after save
        contested = contested.filter((c) => c !== num || true);
        // stay; refresh contested list lightly
        if (data.n_open_disagreements === 0) {
          flash("All severity conflicts done");
        }
        if (idx < queue.length - 1) {
          idx += 1;
          render();
        }
      }
    } catch (ex) {
      el.err.textContent = ex.message || "Save failed";
      el.err.hidden = false;
    }
  }

  async function boot() {
    try {
      me = await api("/api/me", { headers: headers() });
      el.who.textContent =
        me.role === "adjudicator" ? "Adjudicator" : `Annotator ${me.annotator}`;

      const data = await api("/api/severity/items", { headers: headers() });
      items = data.items || [];
      votes = data.votes || {};
      contested = (data.contested || []).map(String);
      el.instructions.textContent = data.instructions || "";

      if (me.role === "adjudicator") {
        el.mode.textContent =
          "Severity adjudication · contested classes only · prior hidden";
        // attach primary labels
        const labels = await api("/api/severity/labels");
        const byItem = {};
        for (const row of labels.rows || []) {
          if (!byItem[row.item_num]) byItem[row.item_num] = {};
          byItem[row.item_num][row.annotator] = row.severity_class;
        }
        queue = contested.map((num) => {
          const base = items.find((it) => String(it.item_num) === String(num));
          return {
            ...(base || { item_num: Number(num), head: "?", relation: "?", tail: "?" }),
            _ann1: (byItem[num] || {})["1"] || "—",
            _ann2: (byItem[num] || {})["2"] || "—",
          };
        });
      } else {
        el.mode.textContent =
          "Risk / severity triples · lexicon prior hidden";
        queue = items.slice();
        // resume at first unrated
        const firstOpen = queue.findIndex(
          (it) => !votes[String(it.item_num)]
        );
        idx = firstOpen >= 0 ? firstOpen : 0;
      }
      render();
    } catch (ex) {
      if (ex.status === 401) {
        location.href = "index.html";
        return;
      }
      el.err.textContent =
        (ex.status === 404
          ? "Severity API missing — restart serve.py (old server still running)."
          : ex.message) || "Load failed";
      el.err.hidden = false;
      // still show class buttons so the page isn't blank chrome
      renderGrid(null);
    }
  }

  el.prev.addEventListener("click", () => {
    if (idx > 0) {
      idx -= 1;
      render();
    }
  });
  el.next.addEventListener("click", () => {
    if (idx < queue.length - 1) {
      idx += 1;
      render();
    }
  });
  el.logout.addEventListener("click", async () => {
    await logout();
    location.href = "index.html";
  });

  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA"))
      return;
    if (e.key === "ArrowLeft") {
      el.prev.click();
    } else if (e.key === "ArrowRight") {
      el.next.click();
    } else if (e.key >= "1" && e.key <= "5") {
      vote(CLASSES[Number(e.key) - 1]);
    }
  });

  boot();
})();

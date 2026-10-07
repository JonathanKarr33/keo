(() => {
  const summary = document.getElementById("summary");
  const agree = document.getElementById("agree");
  const tbody = document.querySelector("#labels-table tbody");
  const sevSummary = document.getElementById("sev-summary");
  const sevAgree = document.getElementById("sev-agree");
  const sevBody = document.querySelector("#sev-table tbody");
  const paths = document.getElementById("paths");

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function loadPrefs() {
    return fetch("/api/labels")
      .then((r) => r.json())
      .then((data) => {
        const by = (data.summary && data.summary.by_annotator) || {};
        const cons = (data.summary && data.summary.consensus) || {};
        const nDisc = (data.summary && data.summary.disagreements) || 0;

        summary.innerHTML = ["1", "2", "3"]
          .map((who) => {
            const row = by[who] || {};
            const n = row.n_rated ?? 0;
            const max = who === "3" ? Math.max(nDisc, 1) : 40;
            const pct = who === "3"
              ? (nDisc ? Math.round((100 * n) / nDisc) : 100)
              : Math.round((100 * n) / 40);
            const title = who === "3" ? "Adjudicator" : `Annotator ${who}`;
            const denom = who === "3" ? `${nDisc} contested` : "40";
            return `<div class="status-card">
              <h2>${title}</h2>
              <p class="big">${n} <span>/ ${denom}</span></p>
              <div class="mini-bar"><span style="width:${Math.min(pct, 100)}%"></span></div>
              <p class="sub">Updated ${esc(row.updated_at || "—")}</p>
            </div>`;
          })
          .join("");

        agree.hidden = false;
        agree.innerHTML = `<strong>Prefs consensus</strong>
          agree ${cons.n_agree ?? 0} ·
          adjudicated ${cons.n_adjudicated ?? 0} ·
          conference ${cons.n_conference ?? 0} ·
          <span class="agree-pct">${cons.n_open ?? 0} open</span>
          <span class="agree-hint">${nDisc} disagreements (system preference)</span>`;

        const rows = data.rows || [];
        tbody.innerHTML = rows.length
          ? rows
              .map((r) => {
                const sys = String(r.preference_system || "");
                const badge =
                  sys === "tie" || sys === "neither"
                    ? "badge-neither"
                    : sys === "keo"
                      ? "badge-keo"
                      : "badge-tok";
                const res = r.resolution ? ` · ${esc(r.resolution)}` : "";
                return `<tr>
              <td>${esc(r.annotator)}${r.role === "adjudicator" ? " (adj)" : ""}</td>
              <td>${esc(r.item_num)}</td>
              <td><code>${esc(r.item_id)}</code></td>
              <td>${esc(r.preference_ab)}${res}</td>
              <td><span class="badge ${badge}">${esc(r.preference_system)}</span></td>
              <td class="when">${esc(r.updated_at)}</td>
            </tr>`;
              })
              .join("")
          : `<tr><td colspan="6" class="empty">No preference votes yet — start on Prefs.</td></tr>`;
      });
  }

  function loadSeverity() {
    return fetch("/api/severity/labels")
      .then((r) => r.json())
      .then((data) => {
        const by = (data.summary && data.summary.by_annotator) || {};
        const cons = (data.summary && data.summary.consensus) || {};
        const nDisc = (data.summary && data.summary.disagreements) || 0;
        const nItems = (data.summary && data.summary.n_items) || 100;

        sevSummary.innerHTML = ["1", "2", "3"]
          .map((who) => {
            const row = by[who] || {};
            const n = row.n_rated ?? 0;
            const denom = who === "3" ? Math.max(nDisc, 1) : nItems;
            const pct = who === "3"
              ? (nDisc ? Math.round((100 * n) / nDisc) : 100)
              : Math.round((100 * n) / nItems);
            const title = who === "3" ? "Adjudicator" : `Annotator ${who}`;
            const denLabel = who === "3" ? `${nDisc} contested` : String(nItems);
            return `<div class="status-card">
              <h2>${title}</h2>
              <p class="big">${n} <span>/ ${denLabel}</span></p>
              <div class="mini-bar"><span style="width:${Math.min(pct, 100)}%"></span></div>
              <p class="sub">Updated ${esc(row.updated_at || "—")}</p>
            </div>`;
          })
          .join("");

        sevAgree.hidden = false;
        sevAgree.innerHTML = `<strong>Severity consensus</strong>
          agree ${cons.n_agree ?? 0} ·
          adjudicated ${cons.n_adjudicated ?? 0} ·
          conference ${cons.n_conference ?? 0} ·
          <span class="agree-pct">${cons.n_open ?? 0} open</span>
          <span class="agree-hint">${nDisc} class disagreements · prior never shown</span>`;

        const rows = data.rows || [];
        sevBody.innerHTML = rows.length
          ? rows
              .map((r) => {
                const trip = `${esc(r.head)} · ${esc(r.relation)} · ${esc(r.tail)}`;
                const res = r.resolution ? ` · ${esc(r.resolution)}` : "";
                return `<tr>
              <td>${esc(r.annotator)}${r.role === "adjudicator" ? " (adj)" : ""}</td>
              <td>${esc(Number(r.item_num) + 1)}</td>
              <td>${trip}</td>
              <td><span class="badge badge-tok">${esc(r.severity_class)}${res}</span></td>
              <td class="when">${esc(r.updated_at)}</td>
            </tr>`;
              })
              .join("")
          : `<tr><td colspan="5" class="empty">No severity votes yet — open Severity.</td></tr>`;

        paths.textContent =
          "Prefs: human_pref_votes_annotator{1,2,3}.json · Severity: severity_votes_annotator{1,2,3}.json · severity_consensus.json (prior key stays offline)";
      });
  }

  function load() {
    Promise.all([loadPrefs(), loadSeverity()]).catch((err) => {
      summary.innerHTML = `<p class="error">Could not load: ${esc(err.message)}</p>`;
    });
  }

  document.getElementById("btn-refresh").addEventListener("click", load);
  load();
})();

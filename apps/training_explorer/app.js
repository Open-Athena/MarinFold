const state = {
  view: "latest",
  data: null,
  selected: null,
  filter: "All",
  query: "",
  viewer: null,
  fallbackCleanup: null,
  serial: 0,
  viewSerial: 0,
};
const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const format = (n) => Number(n).toLocaleString();
const formatE = (value) =>
  Number(value) === 0
    ? "<1e-300"
    : Number(value) < 0.001
      ? Number(value).toExponential(1)
      : Number(value).toPrecision(2);

async function loadView(view) {
  const viewTicket = ++state.viewSerial;
  state.view = view;
  state.query = "";
  state.filter = "All";
  $("search").value = "";
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.view === (view === "eval" ? "eval" : "latest");
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $("corpus-toggle").checked = view === "original";
  $("corpus-toggle-wrap").hidden = view === "eval";
  $("catalog-list").innerHTML =
    '<div class="empty-state">Loading proteins…</div>';
  try {
    const response = await fetch(`data/${view}.json`, { cache: "no-store" });
    if (!response.ok)
      throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    if (viewTicket !== state.viewSerial) return;
    if (!Array.isArray(data.proteins))
      throw new Error("Snapshot has no proteins array");
    state.data = data;
    $("view-description").textContent = data.description;
    $("provenance").textContent = data.provenance || "";
    renderInsights(data);
    renderEvalAnalysis(data, view);
    $("catalog-title").textContent =
      view === "eval" ? "Eval proteins" : "Sampled proteins";
    $("neighbor-caption").textContent = data.neighborsComplete
      ? `10 highest scoring reported local alignments across the ${data.neighborCorpus === "original" ? "original" : "latest"} corpus. Weak supplemental matches are marked.`
      : "Sequence search has not completed for this snapshot.";
    renderFilters();
    state.selected = data.proteins[0] || null;
    renderList();
    renderDetail();
  } catch (error) {
    if (viewTicket !== state.viewSerial) return;
    state.data = null;
    $("view-description").textContent = "Snapshot unavailable";
    $("provenance").textContent = "";
    $("insights").hidden = true;
    $("eval-analysis").hidden = true;
    $("catalog-list").innerHTML =
      `<div class="empty-state">${esc(error.message)}<br>Run the data preparation pipeline in PIPELINE.md.</div>`;
    $("neighbor-list").innerHTML = "";
    $("neighbor-caption").textContent = "No snapshot";
  }
}

function renderEvalAnalysis(data, view) {
  const box = $("eval-analysis");
  if (view !== "eval") {
    box.hidden = true;
    return;
  }
  if (!data.neighborsComplete) {
    box.innerHTML =
      '<span class="eyebrow">EVAL OVERLAP PROFILE</span><p>The complete 116-protein sequence comparison will appear when the latest-corpus search finishes.</p>';
    box.hidden = false;
    return;
  }
  const labels = [
    "No standard hit",
    "<20%",
    "20–30%",
    "30–50%",
    "50–70%",
    "≥70%",
  ];
  const colors = [
    "#a6b2aa",
    "#bddad2",
    "#87c1ad",
    "#50a685",
    "#2d8066",
    "#174f43",
  ];
  const bins = (proteins) => {
    const counts = Array(6).fill(0);
    for (const protein of proteins) {
      const identity = protein.reportedAtE10
        ? protein.neighbors?.[0]?.identity
        : undefined;
      const index =
        identity === undefined
          ? 0
          : identity < 0.2
            ? 1
            : identity < 0.3
              ? 2
              : identity < 0.5
                ? 3
                : identity < 0.7
                  ? 4
                  : 5;
      counts[index]++;
    }
    return counts;
  };
  const rows = ["eval-val", "eval-denovo"]
    .map((subset) => {
      const proteins = data.proteins.filter((p) => p.subset === subset);
      const counts = bins(proteins);
      return `<div class="profile-row"><span>${esc(subset)} <small>n=${proteins.length}</small></span><div class="profile-bar" role="img" aria-label="${esc(subset)} training similarity distribution">${counts.map((count, i) => `<i title="${labels[i]}: ${count}" style="width:${(100 * count) / proteins.length}%;background:${colors[i]}"></i>`).join("")}</div><span class="profile-nohit">${counts[0]} no standard hit</span></div>`;
    })
    .join("");
  const broadMatches = ["eval-val", "eval-denovo"]
    .map((subset) => {
      const proteins = data.proteins.filter((p) => p.subset === subset);
      const count = proteins.filter((p) =>
        p.neighbors.some(
          (hit) =>
            hit.identity >= 0.3 &&
            hit.queryCoverage >= 0.5 &&
            !hit.weakFallback,
        ),
      ).length;
      return `${count}/${proteins.length} ${subset}`;
    })
    .join(" · ");
  const qualifyingHits = data.proteins.flatMap((p) =>
    p.neighbors.filter(
      (hit) =>
        hit.identity >= 0.3 && hit.queryCoverage >= 0.5 && !hit.weakFallback,
    ),
  );
  const sourceNote = !qualifyingHits.length
    ? ""
    : qualifyingHits.every((hit) => hit.source.startsWith("MPNN"))
      ? "All qualifying displayed hits are synthetic redesigns."
      : "Qualifying hits span native and synthetic sources.";
  box.innerHTML = `<div class="profile-title"><span class="eyebrow">EVAL OVERLAP PROFILE</span><p>Identity of each protein’s highest bit-score standard-search hit, shown by evaluation set. “No standard hit” can still have weak supplemental matches.</p></div><div class="profile-body">${rows}<div class="profile-legend">${labels.map((label, i) => `<span><i style="background:${colors[i]}"></i>${esc(label)}</span>`).join("")}</div><p class="profile-threshold">Among the displayed ten: ${esc(broadMatches)} have a hit with ≥30% identity and ≥50% query coverage. ${esc(sourceNote)} Identity is over the aligned region.</p></div>`;
  box.hidden = false;
}

function renderInsights(data) {
  const groups = data.proteins.reduce((result, p) => {
    const key = p.subset || p.source;
    result[key] = (result[key] || 0) + 1;
    return result;
  }, {});
  const totalHits = data.proteins.filter((p) => p.neighbors?.length).length;
  const best = data.proteins
    .map((p) => p.neighbors?.[0]?.identity)
    .filter((x) => Number.isFinite(x));
  const median = best.length
    ? [...best].sort((a, b) => a - b)[Math.floor(best.length / 2)]
    : null;
  const cards = [
    [
      "RECORDS",
      format(data.proteins.length),
      `of ${format(data.population)} proteins`,
    ],
    [
      "COMPOSITION",
      Object.entries(groups)
        .map(([name, count]) => `${name}: ${count}`)
        .join(" · "),
      "sampled documents / eval units",
    ],
    [
      "SEQUENCE NEIGHBORS",
      data.neighborsComplete
        ? `${totalHits}/${data.proteins.length}`
        : "Searching",
      data.neighborsComplete
        ? `${median === null ? "—" : `${(median * 100).toFixed(1)}%`} median best-hit identity`
        : "full-corpus search in progress",
    ],
  ];
  $("insights").innerHTML = cards
    .map(
      ([label, value, note]) =>
        `<div class="insight"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`,
    )
    .join("");
  $("insights").hidden = false;
}

function renderFilters() {
  const values = [
    "All",
    ...new Set(state.data.proteins.map((p) => p.subset || p.source)),
  ];
  $("filters").innerHTML = values
    .map(
      (value) =>
        `<button class="filter ${value === state.filter ? "active" : ""}" data-filter="${esc(value)}">${esc(value)}</button>`,
    )
    .join("");
  $("filters")
    .querySelectorAll("button")
    .forEach((button) =>
      button.addEventListener("click", () => {
        state.filter = button.dataset.filter;
        renderFilters();
        renderList();
      }),
    );
}

function renderList() {
  if (!state.data) return;
  const needle = state.query.toLowerCase();
  const proteins = state.data.proteins.filter(
    (p) =>
      (state.filter === "All" || (p.subset || p.source) === state.filter) &&
      (!needle ||
        `${p.id} ${p.sequence} ${p.source} ${p.title || ""}`
          .toLowerCase()
          .includes(needle)),
  );
  $("list-count").textContent = format(proteins.length);
  $("catalog-list").innerHTML = proteins.length
    ? proteins
        .map(
          (p, i) => `
    <button class="protein-row ${p.id === state.selected?.id ? "active" : ""}" data-id="${esc(p.id)}" role="option" aria-selected="${p.id === state.selected?.id}">
      <span class="row-number">${String(i + 1).padStart(2, "0")}</span><span class="row-text"><strong>${esc(p.label || p.id)}</strong><small>${esc(p.subset || p.source)}</small></span><span class="row-length">${format(p.length || p.sequence?.length)} aa</span>
    </button>`,
        )
        .join("")
    : '<div class="empty-state">No proteins match this filter.</div>';
  $("catalog-list")
    .querySelectorAll(".protein-row")
    .forEach((row) =>
      row.addEventListener("click", () => {
        state.selected = state.data.proteins.find(
          (p) => p.id === row.dataset.id,
        );
        renderList();
        renderDetail();
      }),
    );
}

function renderDetail() {
  const p = state.selected;
  if (!p) return;
  $("protein-source").textContent = p.subset || p.source;
  $("protein-name").textContent = p.label || p.id;
  $("protein-title").textContent =
    p.title || p.organism || "Training corpus protein";
  $("protein-length").textContent = format(p.length || p.sequence?.length);
  $("sequence").textContent = p.sequence || "Sequence unavailable";
  $("meta-set").textContent = p.subset || state.data.title;
  $("meta-origin").textContent = p.source;
  $("meta-structure").textContent = p.structureUrl
    ? p.structureNote || "Predicted / experimental"
    : "Unavailable";
  renderNeighbors(p);
  loadStructure(p);
}

function renderNeighbors(p) {
  const hits = p.neighbors || [];
  $("neighbor-caption").textContent = p.lowComplexitySearch
    ? "This sequence is strongly low complexity. Its ten ranked matches required an unmasked search and should not be read as evidence of homology."
    : `10 highest scoring reported local alignments across the ${state.data.neighborCorpus === "original" ? "original" : "latest"} corpus. Weak supplemental matches are marked.`;
  $("neighbor-count").textContent = String(Math.min(10, hits.length));
  $("neighbor-list").innerHTML = hits.length
    ? hits
        .slice(0, 10)
        .map((hit, i) => {
          const identity = Number(hit.identity || 0);
          return `<div class="neighbor-row"><div class="neighbor-top"><strong title="${esc(hit.id)}">${String(i + 1).padStart(2, "0")} · ${esc(hit.label || hit.id)}</strong><span>${(100 * identity).toFixed(1)}%</span></div><div class="neighbor-bar"><i style="width:${Math.min(100, Math.max(0, 100 * identity))}%"></i></div><div class="neighbor-sub"><span>${esc(hit.source)} · ${format(hit.length)} aa${hit.weakFallback ? ' · <em title="Supplemental permissive search; weak similarity does not establish homology">WEAK</em>' : ""}</span><span>B ${hit.bitscore} · Q ${(100 * Number(hit.queryCoverage || 0)).toFixed(0)}% · E≈${formatE(hit.evalue)}</span></div></div>`;
        })
        .join("")
    : `<div class="empty-state">${state.data.neighborsComplete ? "No sequence matches were reported at the search threshold." : "Neighbor search pending. The full corpus has not been searched yet."}</div>`;
}

async function loadStructure(p) {
  const ticket = ++state.serial;
  $("viewer-status").textContent = p.structureUrl
    ? "Loading 3D structure…"
    : "Structure not available for this entry";
  if (state.viewer) {
    state.viewer.dispose();
    state.viewer = null;
  }
  if (state.fallbackCleanup) {
    state.fallbackCleanup();
    state.fallbackCleanup = null;
  }
  $("molstar").replaceChildren();
  if (!p.structureUrl) return;
  try {
    const probe = document.createElement("canvas");
    if (!probe.getContext("webgl") && !probe.getContext("experimental-webgl")) {
      state.fallbackCleanup = await window.createBackboneFallback(
        $("molstar"),
        p,
        () => ticket === state.serial,
      );
      if (ticket === state.serial)
        $("viewer-status").textContent = "Interactive 3D backbone preview";
      return;
    }
    if (!window.molstar?.Viewer) throw new Error("Mol* library unavailable");
    const viewer = await window.molstar.Viewer.create("molstar", {
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      viewportShowExpand: false,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
    });
    if (ticket !== state.serial) {
      viewer.dispose();
      return;
    }
    if (!viewer.plugin.canvas3d) {
      viewer.dispose();
      throw new Error("Mol* WebGL canvas unavailable");
    }
    state.viewer = viewer;
    await viewer.loadStructureFromUrl(
      p.structureUrl,
      p.structureFormat || "mmcif",
      false,
    );
    if (ticket === state.serial)
      $("viewer-status").textContent =
        p.structureNote || "Interactive 3D structure";
  } catch (error) {
    if (ticket !== state.serial) return;
    try {
      state.fallbackCleanup = await window.createBackboneFallback(
        $("molstar"),
        p,
        () => ticket === state.serial,
      );
      $("viewer-status").textContent = "Interactive 3D backbone preview";
    } catch (fallbackError) {
      $("viewer-status").textContent =
        `Structure load failed: ${fallbackError.message}`;
    }
  }
}

document
  .querySelectorAll(".tab")
  .forEach((tab) =>
    tab.addEventListener("click", () => loadView(tab.dataset.view)),
  );
$("corpus-toggle").addEventListener("change", (event) =>
  loadView(event.target.checked ? "original" : "latest"),
);
$("search").addEventListener("input", (event) => {
  state.query = event.target.value;
  renderList();
});
$("copy-sequence").addEventListener("click", async () => {
  if (!state.selected?.sequence) return;
  await navigator.clipboard.writeText(state.selected.sequence);
  $("copy-sequence").textContent = "Copied";
  setTimeout(() => {
    $("copy-sequence").textContent = "Copy";
  }, 1500);
});
loadView("latest");

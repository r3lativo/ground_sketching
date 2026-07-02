const DATA_DIR = "data";

const state = {
  index: [],       // [{id, category, label, num_messages}, ...]
  conv: null,      // {id, category, rows: [...]}
  selectedPos: 0,  // position of the selected row within conv.rows
  sideOverride: { A: null, B: null }, // manual thumbnail preview per side
  searchMatches: [],
  searchCursor: -1,
};

const el = (id) => document.getElementById(id);

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}`);
  return res.json();
}

async function loadIndex() {
  state.index = await fetchJSON(`${DATA_DIR}/index.json`);
}

async function loadConversation(id) {
  state.conv = await fetchJSON(`${DATA_DIR}/${id}.json`);
  state.sideOverride = { A: null, B: null };
  closeSearch();
  const chatPositions = getChatPositions();
  state.selectedPos = chatPositions.length ? chatPositions[0] : 0;
  render();
  el("chatScroll").scrollTop = 0;
}

function getChatPositions() {
  return state.conv.rows.reduce((acc, r, i) => {
    if (r.mtype === "text") acc.push(i);
    return acc;
  }, []);
}

function getAnnotationRows() {
  return state.conv.rows.filter((r) => r.mtype && r.mtype !== "text");
}

// The one rule that drives the image + metadata on each side panel: nearest
// row at or before `uptoPos` where `character` produced an image.
function findSideRow(character, uptoPos) {
  const rows = state.conv.rows;
  for (let i = uptoPos; i >= 0; i--) {
    if (rows[i].character === character && rows[i].img_path) return rows[i];
  }
  return null;
}

// Every distinct frame this character has ever had, each at its latest
// (most-evolved) image, in frame-number order. Not bounded by the current
// selection — the whole set is always shown, since this is an overview of
// the character's whole visual history, not a "so far" list.
function getGlobalFrames(character) {
  const latestByFrame = new Map();
  state.conv.rows.forEach((row, pos) => {
    if (row.character === character && row.frame_id && row.img_path) {
      latestByFrame.set(row.frame_id, { row, pos }); // later rows overwrite -> keeps the latest
    }
  });
  return Array.from(latestByFrame.entries())
    .map(([frameId, { row, pos }]) => {
      const match = frameId.match(/(\d+)\s*$/);
      return { frameId, num: match ? parseInt(match[1], 10) : null, row, pos };
    })
    .sort((a, b) => (a.num ?? 0) - (b.num ?? 0));
}

// The character's most recent turn at all, image or not. SKIP turns have no
// img_path (nothing to show), but the badge still needs to say "SKIP" when
// that's genuinely the last thing this character did.
function findLatestTurn(character, uptoPos) {
  const rows = state.conv.rows;
  for (let i = uptoPos; i >= 0; i--) {
    if (rows[i].character === character) return rows[i];
  }
  return null;
}

function render() {
  renderConvLabel();
  renderChat();
  renderQA();
  renderSide("A");
  renderSide("B");
  renderNavButtons();
}

function renderConvLabel() {
  el("convLabel").textContent = state.conv
    ? `${state.conv.category.replace("_VA", "")} — ${state.conv.id}`
    : "";
}

function renderChat() {
  const list = el("chatList");
  list.innerHTML = "";
  const rows = state.conv.rows;
  const chatPositions = getChatPositions();
  if (!chatPositions.length) return;

  const query = el("searchInput").value.trim().toLowerCase();
  state.searchMatches = [];

  chatPositions.forEach((pos) => {
    const row = rows[pos];
    const bubble = document.createElement("div");
    // A is always on the left (matches the A-side panel), B always on the right.
    bubble.className = "bubble " + (row.character === "B" ? "align-right" : "align-left");
    if (pos === state.selectedPos) bubble.classList.add("selected");

    const who = document.createElement("span");
    who.className = "who";
    who.textContent = `${row.character}:`;
    bubble.appendChild(who);
    bubble.appendChild(document.createTextNode(row.text));

    if (query) {
      const isMatch = row.text.toLowerCase().includes(query);
      if (isMatch) {
        bubble.classList.add("search-match");
        state.searchMatches.push(pos);
      } else {
        bubble.classList.add("search-dim");
      }
    }

    bubble.addEventListener("click", () => selectPos(pos));
    list.appendChild(bubble);
  });

  if (query) {
    el("searchCount").textContent = state.searchMatches.length
      ? `${state.searchMatches.length} match${state.searchMatches.length > 1 ? "es" : ""}`
      : "no matches";
  } else {
    el("searchCount").textContent = "";
  }
}

function renderQA() {
  const list = el("qaList");
  list.innerHTML = "";
  const rows = getAnnotationRows();
  if (!rows.length) {
    list.innerHTML = '<div class="qa-item">Nothing to show.</div>';
    return;
  }
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "qa-item";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = row.mtype;
    item.appendChild(tag);
    item.appendChild(document.createTextNode(`${row.character || "?"}: ${row.text}`));
    list.appendChild(item);
  });
}

function metaField(label, value) {
  const wrap = document.createElement("div");
  wrap.className = "meta-item";
  const l = document.createElement("span");
  l.className = "label";
  l.textContent = label;
  const v = document.createElement("span");
  v.className = "value";
  v.textContent = value && value.length ? value : "-";
  wrap.appendChild(l);
  wrap.appendChild(v);
  return wrap;
}

// Frame Choice drives a color-coded badge shown right under the image:
// green = a new frame was started, blue = the existing frame was continued,
// orange = this turn didn't touch the image.
function frameChoiceBadge(rawValue) {
  const clean = (rawValue || "").replace(/[\[\]]/g, "").trim().toUpperCase();
  if (!clean) return null;
  const classByChoice = { NEW: "fc-new", CONTINUE: "fc-continue", SKIP: "fc-skip" };
  return { text: clean, cls: classByChoice[clean] || "fc-other" };
}

function renderGlobalTracker(character) {
  const wrap = el(`globalTracker${character}`);
  wrap.innerHTML = "";
  const activeRow = findSideRow(character, state.selectedPos);
  const activeFrameId = activeRow ? activeRow.frame_id : null;

  getGlobalFrames(character).forEach(({ frameId, num, row, pos }) => {
    const thumb = document.createElement("div");
    thumb.className = "global-thumb" + (frameId === activeFrameId ? " global-thumb-active" : "");
    thumb.title = frameId;

    const img = document.createElement("img");
    img.src = row.img_path;
    thumb.appendChild(img);

    const label = document.createElement("span");
    label.className = "global-thumb-num";
    label.textContent = num !== null ? num : "?";
    thumb.appendChild(label);

    thumb.addEventListener("click", () => selectPos(pos));
    wrap.appendChild(thumb);
  });
}

function renderSide(character) {
  const row = findSideRow(character, state.selectedPos);
  const img = el(`image${character}`);
  const noImage = el(`noImage${character}`);
  const thumbStrip = el(`thumbs${character}`);
  const badgeWrap = el(`badge${character}`);
  const promptCaption = el(`prompt${character}`);
  const metaGrid = el(`meta${character}`);

  thumbStrip.innerHTML = "";
  badgeWrap.innerHTML = "";
  metaGrid.innerHTML = "";
  renderGlobalTracker(character);

  // The badge reflects this character's most recent turn, even a SKIP one
  // (which has no image of its own) — that's the case the badge matters most.
  const latestTurn = findLatestTurn(character, state.selectedPos);
  const badge = latestTurn ? frameChoiceBadge(latestTurn.frame_choice) : null;
  if (badge) {
    const span = document.createElement("span");
    span.className = `frame-badge ${badge.cls}`;
    span.textContent = badge.text;
    badgeWrap.appendChild(span);
  }

  if (!row) {
    img.classList.add("hidden");
    noImage.classList.remove("hidden");
    promptCaption.textContent = "";
    state.sideOverride[character] = null;
    return;
  }

  const sequence = row.sequence || [];
  const override = state.sideOverride[character];
  const displayed = (override && sequence.some((s) => s.path === override.path))
    ? override
    : sequence.find((s) => s.is_current) || { path: row.img_path, prompt: "" };

  img.src = displayed.path;
  img.classList.remove("hidden");
  noImage.classList.add("hidden");
  promptCaption.textContent = displayed.prompt ? displayed.prompt : "No sub-prompt for this step.";

  sequence
    .filter((s) => s.path !== (sequence.find((x) => x.is_current) || {}).path)
    .forEach((s) => {
      const thumb = document.createElement("img");
      thumb.src = s.path;
      thumb.title = s.seq_id;
      if (s.path === displayed.path) thumb.classList.add("thumb-active");
      thumb.addEventListener("click", () => {
        state.sideOverride[character] = s;
        renderSide(character);
      });
      thumbStrip.appendChild(thumb);
    });

  metaGrid.appendChild(metaField("Frame Meta", row.frame_meta));
  metaGrid.appendChild(metaField("Relation", row.relation));
  metaGrid.appendChild(metaField("Extracted Triplets", row.extracted_triplets));
  metaGrid.appendChild(metaField("Frame ID", row.frame_id));
  metaGrid.appendChild(metaField("Imagery", row.imagery));
  metaGrid.appendChild(metaField("Final Prompt", row.final_prompt));
}

function selectPos(pos) {
  state.selectedPos = pos;
  state.sideOverride = { A: null, B: null };
  render();
  const selected = document.querySelector(".bubble.selected");
  if (selected) selected.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function renderNavButtons() {
  const chatPositions = getChatPositions();
  const cursor = chatPositions.indexOf(state.selectedPos);
  el("btnFirst").disabled = cursor <= 0;
  el("btnPrev").disabled = cursor <= 0;
  el("btnNext").disabled = cursor === -1 || cursor >= chatPositions.length - 1;
  el("btnLast").disabled = cursor === -1 || cursor >= chatPositions.length - 1;
}

function stepChat(delta) {
  const chatPositions = getChatPositions();
  const cursor = chatPositions.indexOf(state.selectedPos);
  const next = Math.min(Math.max(cursor + delta, 0), chatPositions.length - 1);
  selectPos(chatPositions[next]);
}

function jumpChat(toEnd) {
  const chatPositions = getChatPositions();
  if (!chatPositions.length) return;
  selectPos(toEnd ? chatPositions[chatPositions.length - 1] : chatPositions[0]);
}

// --- Menu (conversation picker) ---
function renderMenu() {
  const list = el("menuList");
  list.innerHTML = "";
  state.index.forEach((entry) => {
    const btn = document.createElement("button");
    btn.className = "menu-item" + (state.conv && state.conv.id === entry.id ? " active" : "");
    btn.textContent = `${entry.label} (${entry.num_messages} messages)`;
    btn.addEventListener("click", async () => {
      await loadConversation(entry.id);
      closeMenu();
    });
    list.appendChild(btn);
  });
}

function openMenu() {
  renderMenu();
  el("menuOverlay").classList.add("open");
}

function closeMenu() {
  el("menuOverlay").classList.remove("open");
}

// --- Search ---
function openSearch() {
  el("searchBar").classList.add("open");
  el("searchInput").focus();
}

function closeSearch() {
  el("searchBar").classList.remove("open");
  el("searchInput").value = "";
  state.searchMatches = [];
  state.searchCursor = -1;
  renderChat();
}

function searchNext() {
  if (!state.searchMatches.length) return;
  state.searchCursor = (state.searchCursor + 1) % state.searchMatches.length;
  selectPos(state.searchMatches[state.searchCursor]);
}

// --- Q&A panel (sticky, toggled independently of the chat scroll) ---
function toggleQA() {
  const open = el("qaPanel").classList.toggle("open");
  el("btnQA").classList.toggle("active", open);
}

function closeQA() {
  el("qaPanel").classList.remove("open");
  el("btnQA").classList.remove("active");
}

// --- Random ---
async function loadRandomConversation() {
  const others = state.index.filter((e) => !state.conv || e.id !== state.conv.id);
  const pool = others.length ? others : state.index;
  const pick = pool[Math.floor(Math.random() * pool.length)];
  await loadConversation(pick.id);
}

function wireUp() {
  el("btnMenu").addEventListener("click", openMenu);
  el("menuClose").addEventListener("click", closeMenu);
  el("menuOverlay").addEventListener("click", (e) => {
    if (e.target.id === "menuOverlay") closeMenu();
  });

  el("btnSearch").addEventListener("click", () => {
    if (el("searchBar").classList.contains("open")) closeSearch();
    else openSearch();
  });
  el("searchInput").addEventListener("input", renderChat);
  el("searchInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") searchNext();
    if (e.key === "Escape") closeSearch();
  });

  el("btnRandom").addEventListener("click", loadRandomConversation);

  el("btnQA").addEventListener("click", toggleQA);
  el("qaClose").addEventListener("click", closeQA);

  el("btnFirst").addEventListener("click", () => jumpChat(false));
  el("btnPrev").addEventListener("click", () => stepChat(-1));
  el("btnNext").addEventListener("click", () => stepChat(1));
  el("btnLast").addEventListener("click", () => jumpChat(true));
}

async function main() {
  wireUp();
  await loadIndex();
  if (state.index.length) {
    await loadConversation(state.index[0].id);
  }
}

main();

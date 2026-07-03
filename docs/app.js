const DATA_DIR = "data";

const state = {
  index: [],       // [{id, category, label, num_messages}, ...]
  conv: null,      // {id, category, rows: [...]}
  selectedPos: 0,  // position of the selected row within conv.rows
  sideOverride: { A: null, B: null }, // manual thumbnail preview per side
  searchMatches: [],
  searchCursor: -1,
  tutorialOpen: false,
  tutorialStep: 0,
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

// The single rule that drives everything about a side panel: this
// character's most recent chat turn at or before `uptoPos`, image or not.
// Selecting one of A's own bubbles resolves to that exact turn (uptoPos
// itself matches immediately); selecting a B bubble resolves to A's latest
// turn *as of then* -- which may itself be a SKIP with no image, and that's
// shown honestly rather than borrowing an older image. Restricted to text
// rows so Question/Answer annotation rows (which share the same
// `character` value but carry no frame_choice) don't shadow the real turn.
function findLatestTurn(character, uptoPos) {
  const rows = state.conv.rows;
  for (let i = uptoPos; i >= 0; i--) {
    if (rows[i].character === character && rows[i].mtype === "text") return rows[i];
  }
  return null;
}

// The chat position of the turn that produced a given sequence image, so
// clicking a sequence-strip thumb can jump to its chat bubble the same way
// the global tracker does.
//
// Usually the turn's own `img_path` points exactly at this file. But a
// single turn can use "$$$" in its final_prompt to describe two edits in
// one go (e.g. "...$$$..."), which produces two sequence images while only
// the *last* one becomes that row's own `img_path` -- the intermediate one
// has no row of its own. In that case, fall back to the earliest turn for
// this frame whose own resulting seq number has caught up to (or passed)
// this one, since that's the turn that generated it as a byproduct.
function findPosForImage(character, path, targetSeqNum, frameId) {
  const rows = state.conv.rows;
  const exact = rows.findIndex((r) => r.character === character && r.img_path === path);
  if (exact !== -1) return exact;
  if (targetSeqNum == null || !frameId) return -1;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (r.character !== character || r.frame_id !== frameId || !r.img_path) continue;
    const m = r.img_path.match(/_seq(\d+)\./);
    const rSeq = m ? parseInt(m[1], 10) : null;
    if (rSeq !== null && rSeq >= targetSeqNum) return i;
  }
  return -1;
}

function render() {
  renderConvLabel();
  renderChat();
  renderQA();
  renderSide("A");
  renderSide("B");
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
    // A is always on the left (matches the A-side panel), B always on the
    // right. The bubble's own background carries its frame-choice color
    // directly instead of a separate badge.
    const badge = frameChoiceBadge(row.frame_choice);
    bubble.className = "bubble " + (row.character === "B" ? "align-right" : "align-left") +
      (badge ? ` ${badge.cls}` : "");
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

function metaField(label, value, spanClass) {
  const wrap = document.createElement("div");
  wrap.className = "meta-item" + (spanClass ? ` ${spanClass}` : "");
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

// Shared thumb renderer used by both the global frame tracker and the
// per-frame sequence strip -- they're the same visual language, just
// different sizes and different sets of entries.
function renderThumb(container, { src, label, active, sizeClass, title, onClick }) {
  const thumb = document.createElement("div");
  thumb.className = `global-thumb ${sizeClass}` + (active ? " global-thumb-active" : "");
  if (title) thumb.title = title;

  const img = document.createElement("img");
  img.src = src;
  thumb.appendChild(img);

  const labelEl = document.createElement("span");
  labelEl.className = "global-thumb-num";
  labelEl.textContent = label;
  thumb.appendChild(labelEl);

  thumb.addEventListener("click", onClick);
  container.appendChild(thumb);
}

function renderGlobalTracker(character) {
  const wrap = el(`globalTracker${character}`);
  wrap.innerHTML = "";
  const activeRow = findLatestTurn(character, state.selectedPos);
  const activeFrameId = activeRow ? activeRow.frame_id : null;

  getGlobalFrames(character).forEach(({ frameId, num, row, pos }) => {
    renderThumb(wrap, {
      src: row.img_path,
      label: num !== null ? num : "?",
      active: frameId === activeFrameId,
      sizeClass: "thumb-size-tracker",
      title: frameId,
      onClick: () => selectPos(pos),
    });
  });
}

function extractSeqNum(seqId) {
  const m = (seqId || "").match(/(\d+)/);
  return m ? parseInt(m[1], 10) : 1;
}

function extractFrameNum(frameId) {
  const m = (frameId || "").match(/(\d+)\s*$/);
  return m ? parseInt(m[1], 10) : null;
}

function renderSide(character) {
  // One rule for both sides (see findLatestTurn): this character's own turn
  // when it's the one selected, or its latest turn as of the selection
  // otherwise. Whatever that turn actually has -- image or none -- is what
  // gets shown; no borrowing an image or metadata from a different turn.
  const row = findLatestTurn(character, state.selectedPos);
  const img = el(`image${character}`);
  const noImage = el(`noImage${character}`);
  const thumbStrip = el(`thumbs${character}`);
  const badgeWrap = el(`badge${character}`);
  const metaGrid = el(`meta${character}`);

  thumbStrip.innerHTML = "";
  badgeWrap.innerHTML = "";
  metaGrid.innerHTML = "";
  renderGlobalTracker(character);

  const badge = row ? frameChoiceBadge(row.frame_choice) : null;
  if (badge) {
    const span = document.createElement("span");
    span.className = `frame-badge ${badge.cls}`;
    span.textContent = badge.text;
    badgeWrap.appendChild(span);
  }

  if (!row) {
    img.classList.add("hidden");
    noImage.classList.remove("hidden");
    state.sideOverride[character] = null;
    return;
  }

  let promptText = "";

  if (row.img_path) {
    const sequence = row.sequence || [];
    const currentEntry = sequence.find((s) => s.is_current) || { path: row.img_path, prompt: "", seq_id: "Final" };
    const override = state.sideOverride[character];
    const displayed = (override && sequence.some((s) => s.path === override.path))
      ? override
      : currentEntry;

    img.src = displayed.path;
    img.classList.remove("hidden");
    noImage.classList.add("hidden");

    // The prompt for *this specific image*: when a turn's final_prompt used
    // "$$$" to describe two edits at once, `displayed.prompt` is already
    // just the one segment that produced this exact image (see
    // resolve_sequence() in build_site.py). Falls back to the turn's whole
    // final_prompt for the common case of a single, unsplit prompt.
    promptText = displayed.prompt || row.final_prompt || "";

    const frameNum = extractFrameNum(row.frame_id);
    sequence.forEach((s) => {
      const seqNum = extractSeqNum(s.seq_id);
      renderThumb(thumbStrip, {
        src: s.path,
        label: frameNum !== null ? `${frameNum}.${seqNum}` : seqNum,
        active: s.path === displayed.path,
        sizeClass: "thumb-size-seq",
        title: s.prompt ? `${s.seq_id}: ${s.prompt}` : s.seq_id,
        onClick: () => {
          // Navigate to the turn that produced this file (for context), then
          // force the preview to this exact sequence entry -- otherwise an
          // intermediate "$$$" step (see findPosForImage) would land you on
          // its owning turn but show that turn's own *final* image instead of
          // the intermediate one you actually clicked.
          const pos = findPosForImage(character, s.path, seqNum, row.frame_id);
          if (pos !== -1) selectPos(pos);
          state.sideOverride[character] = s;
          renderSide(character);
        },
      });
    });
  } else {
    // A genuine SKIP: no image, and no fields below borrowed from an
    // earlier turn either -- metaField already renders "-" for each blank
    // one, which is the honest state of *this* turn.
    img.classList.add("hidden");
    noImage.classList.remove("hidden");
    state.sideOverride[character] = null;
  }

  metaGrid.appendChild(metaField("Prompt", promptText, "meta-span-all"));
  metaGrid.appendChild(metaField("Frame Meta", row.frame_meta, "meta-span-third"));
  metaGrid.appendChild(metaField("Frame ID", row.frame_id, "meta-span-third"));
  metaGrid.appendChild(metaField("Extracted Triplets", row.extracted_triplets, "meta-span-third"));
  metaGrid.appendChild(metaField("Relation", row.relation, "meta-span-half"));
  metaGrid.appendChild(metaField("Imagery", row.imagery, "meta-span-half"));
}

function selectPos(pos) {
  state.selectedPos = pos;
  state.sideOverride = { A: null, B: null };
  render();
  const selected = document.querySelector(".bubble.selected");
  if (selected) selected.scrollIntoView({ block: "nearest", behavior: "smooth" });
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

// --- Tutorial ---
// Anchored to this specific conversation so every step has real, curated
// data behind it (see anchorPos below) -- opening the tour always loads it,
// regardless of what the visitor was previously browsing.
const TUTORIAL_CONV_ID = "262_163_290_15";

const TUTORIAL_STEPS = [
  {
    title: "Welcome",
    text: "Two human participants, A and B, try to find each other in a virtual environment. We let an Agentic system build imagery of scenes through conversation alone. Each message can start a new sketch, edit the existing one, or leave it untouched: let's walk through how that's shown here.",
    selector: null,
    anchorPos: 13,
  },
  {
    title: "The conversation",
    text: "This is the dialogue between A and B. Click any message to jump straight to that turn. The panels on either side always reflect what was true at that exact moment.",
    selector: ".chat-col",
    placement: "right",
  },
  {
    title: "Each side's sketch",
    text: "This panel shows what A pictured as of the selected turn: A's own state if it's A's turn, or A's most recent state otherwise.",
    selector: "#sideA",
    placement: "right",
  },
  {
    title: "Skipped turns",
    text: "Not every turn produces a change to the picture. B's last turn here was skipped, so B's panel shows “No image available.”",
    selector: "#sideB",
    anchorPos: 8,
    placement: "left",
  },
  {
    title: "Global frame history",
    text: "Every distinct scene sketched from A's POV in this conversation, each at its latest version. Click any thumbnail to preview it and jump to the corresponding message in the conversation.",
    selector: "#globalTrackerA",
    anchorPos: 13,
  },
  {
    title: "Edit history",
    text: "Every edit made to the current scene.",
    selector: "#thumbsA",
  },
  {
    title: "Prompt & metadata",
    text: "The exact prompt that generated the current image, plus the structured data extracted from that turn (frame metadata, relations, imagery).",
    selector: "#metaA",
  },
  {
    title: "Color legend",
    text: "Green marks a brand-new sketch, purple an edit to the existing one, orange a skipped turn.",
    selector: ".footer-legend",
  },
  {
    title: "More tools",
    text: "Switch conversations, search within one, jump to a random example, or inspect the underlying Q&A annotations from here.",
    selector: ".footer-left",
  },
  {
    title: "That's it",
    text: "Click any message to start exploring. You can reopen this tour anytime from the “Show tutorial” button.",
    selector: null,
  },
];

let tutorialShineTimeout = null;

async function openTutorial() {
  closeSearch();
  closeMenu();
  closeQA();
  if (!state.conv || state.conv.id !== TUTORIAL_CONV_ID) {
    await loadConversation(TUTORIAL_CONV_ID);
  }
  state.tutorialOpen = true;
  state.tutorialStep = 0;
  el("tutorialBox").classList.add("open");
  window.addEventListener("resize", positionCurrentTutorialStep);
  window.addEventListener("scroll", positionCurrentTutorialStep, true);
  renderTutorialStep();
}

function closeTutorial() {
  clearTimeout(tutorialShineTimeout);
  el("tutorialNext").classList.remove("tutorial-next-shine");
  state.tutorialOpen = false;
  el("tutorialBackdrop").classList.remove("open");
  el("tutorialSpotlight").classList.remove("open");
  el("tutorialBox").classList.remove("open");
  window.removeEventListener("resize", positionCurrentTutorialStep);
  window.removeEventListener("scroll", positionCurrentTutorialStep, true);
}

function renderTutorialStep() {
  const step = TUTORIAL_STEPS[state.tutorialStep];
  if (step.anchorPos !== undefined) selectPos(step.anchorPos);

  el("tutorialStepCount").textContent = `${state.tutorialStep + 1} / ${TUTORIAL_STEPS.length}`;
  el("tutorialTitle").textContent = step.title;
  el("tutorialText").textContent = step.text;
  el("tutorialPrev").disabled = state.tutorialStep === 0;
  el("tutorialNext").textContent = state.tutorialStep === TUTORIAL_STEPS.length - 1 ? "Done" : "Next";

  // Nudge toward the next step if this one sits unread for a while.
  clearTimeout(tutorialShineTimeout);
  el("tutorialNext").classList.remove("tutorial-next-shine");
  tutorialShineTimeout = setTimeout(() => {
    el("tutorialNext").classList.add("tutorial-next-shine");
  }, 5000);

  if (!step.selector) {
    el("tutorialBackdrop").classList.add("open");
    positionSpotlight(null);
    positionTutorialBox(null);
    return;
  }

  el("tutorialBackdrop").classList.remove("open");
  const target = document.querySelector(step.selector);
  if (!target) {
    positionSpotlight(null);
    positionTutorialBox(null);
    return;
  }
  target.scrollIntoView({ block: "center", behavior: "smooth" });
  // Give the smooth scroll a moment to settle before measuring/positioning.
  setTimeout(() => {
    const rect = target.getBoundingClientRect();
    positionSpotlight(rect);
    positionTutorialBox(rect, step.placement);
  }, 300);
}

// Reposition-only pass for resize/scroll while the tour is open -- doesn't
// re-select a position or re-run the scroll choreography.
function positionCurrentTutorialStep() {
  const step = TUTORIAL_STEPS[state.tutorialStep];
  if (!step.selector) {
    positionSpotlight(null);
    positionTutorialBox(null);
    return;
  }
  const target = document.querySelector(step.selector);
  if (target) {
    const rect = target.getBoundingClientRect();
    positionSpotlight(rect);
    positionTutorialBox(rect, step.placement);
  }
}

// Sizes/moves the cutout to match the target's box exactly (plus a little
// padding) -- everywhere outside of it is dimmed by its own box-shadow.
function positionSpotlight(rect) {
  const spotlight = el("tutorialSpotlight");
  if (!rect) {
    spotlight.classList.remove("open");
    return;
  }
  const pad = 4;
  spotlight.style.left = `${rect.left - pad}px`;
  spotlight.style.top = `${rect.top - pad}px`;
  spotlight.style.width = `${rect.width + pad * 2}px`;
  spotlight.style.height = `${rect.height + pad * 2}px`;
  spotlight.classList.add("open");
}

// `placement` of "left"/"right" puts the box beside the target (vertically
// centered on it) instead of below/above -- for tall targets like the chat
// column or a side panel, below/above would land on top of their own
// content instead of next to it.
function positionTutorialBox(rect, placement) {
  const box = el("tutorialBox");
  const margin = 12;
  const boxWidth = box.offsetWidth || 320;
  const boxHeight = box.offsetHeight || 140;

  if (!rect) {
    box.style.left = `${(window.innerWidth - boxWidth) / 2}px`;
    box.style.top = `${(window.innerHeight - boxHeight) / 2}px`;
    return;
  }

  let top, left;

  if (placement === "left" || placement === "right") {
    top = rect.top + rect.height / 2 - boxHeight / 2;
    top = Math.max(margin, Math.min(top, window.innerHeight - boxHeight - margin));

    if (placement === "right") {
      left = rect.right + margin;
      if (left + boxWidth > window.innerWidth - margin) left = rect.left - boxWidth - margin;
    } else {
      left = rect.left - boxWidth - margin;
      if (left < margin) left = rect.right + margin;
    }
    left = Math.max(margin, Math.min(left, window.innerWidth - boxWidth - margin));
  } else {
    top = rect.bottom + margin;
    if (top + boxHeight > window.innerHeight - margin) {
      top = rect.top - boxHeight - margin;
    }
    top = Math.max(margin, Math.min(top, window.innerHeight - boxHeight - margin));

    left = rect.left + rect.width / 2 - boxWidth / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - boxWidth - margin));
  }

  box.style.top = `${top}px`;
  box.style.left = `${left}px`;
}

function tutorialNext() {
  if (state.tutorialStep >= TUTORIAL_STEPS.length - 1) {
    closeTutorial();
    return;
  }
  state.tutorialStep += 1;
  renderTutorialStep();
}

function tutorialPrev() {
  if (state.tutorialStep <= 0) return;
  state.tutorialStep -= 1;
  renderTutorialStep();
}

// First-ever visit: open the tour by default instead of waiting to be
// clicked. Marked as seen right away so a reload (or any later visit)
// doesn't keep forcing it back open -- the button remains available to
// reopen it manually at any time.
function maybeAutoOpenTutorial() {
  if (localStorage.getItem("gs_tutorial_prompted")) return;
  localStorage.setItem("gs_tutorial_prompted", "true");
  openTutorial();
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

  el("btnTutorial").addEventListener("click", openTutorial);
  el("tutorialNext").addEventListener("click", tutorialNext);
  el("tutorialPrev").addEventListener("click", tutorialPrev);
  el("tutorialSkip").addEventListener("click", closeTutorial);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.tutorialOpen) closeTutorial();
  });
}

async function main() {
  wireUp();
  await loadIndex();
  if (state.index.length) {
    await loadConversation(state.index[0].id);
  }
  maybeAutoOpenTutorial();
}

main();
